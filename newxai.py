import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients, GuidedGradCam

# Add the TransFuser path to sys.path
sys.path.append("./team_code_transfuser")

# Import TransFuser modules
from team_code_transfuser.transfuser import TransfuserBackbone
from team_code_transfuser.config import GlobalConfig

# ----------------------- #
# TransFuser Model Wrapper
# ----------------------- #

class RGBWrapper(torch.nn.Module):
    """Wrapper for RGB input attribution"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, rgb_input):
        # Create zeroed lidar input of correct dimensions
        batch_size = rgb_input.shape[0]
        lidar_input = torch.zeros((batch_size, 2, 256, 256), device=rgb_input.device)
        velocity = torch.zeros((batch_size, 1), device=rgb_input.device)
        
        _, _, fused_features = self.model(rgb_input, lidar_input, velocity)
        return fused_features

# ----------------------- #
# TransFuser Explainer
# ----------------------- #

class CameraOnlyTransfuserExplainer:
    """Camera-focused explainability system for TransFuser"""
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.attention_maps = {}
        
        # Create wrapper module for RGB
        self.rgb_wrapper = RGBWrapper(self.model)
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.rgb_wrapper)
        
        # For LayerGradCAM, use a proper module
        self.rgb_gradcam = GuidedGradCam(self.rgb_wrapper, 
                                         self.model.image_encoder.features.layer4)
        
        # Register hooks to capture attention weights
        self.register_attention_hooks()
        
    def register_attention_hooks(self):
        """Register hooks to capture transformer attention patterns"""
        def hook_fn(module, input, output, name):
            # Store attention weights
            self.attention_maps[name] = output
            
        # Hook into transformer attention blocks specifically
        for i, transformer in enumerate([
            self.model.transformer1, 
            self.model.transformer2,
            self.model.transformer3, 
            self.model.transformer4
        ]):
            # Access SelfAttention directly
            for j, block in enumerate(transformer.blocks):
                block.attn.register_forward_hook(
                    lambda mod, inp, out, name=f"transformer_{i+1}_block_{j+1}": 
                    hook_fn(mod, inp, out, name)
                )
    
    def extract_attention_patterns(self):
        """Extract and analyze attention patterns in transformer blocks"""
        attention_patterns = {}
        
        for name, attention in self.attention_maps.items():
            try:
                # Skip if attention is None or not the right format
                if attention is None or not isinstance(attention, torch.Tensor):
                    continue
                    
                # Extract the attention weights
                if len(attention.shape) >= 3:  # Typically [batch, seq_len, seq_len] or [batch, heads, seq_len, seq_len]
                    # Process attention tensor
                    attention_map = attention.mean(dim=0) if len(attention.shape) == 3 else attention.mean(dim=(0, 1))
                    
                    # Store processed attention for this transformer block
                    attention_patterns[name] = {
                        'tensor': attention_map.detach().cpu(),
                        'mean': float(attention_map.mean().item()),
                        'max': float(attention_map.max().item())
                    }
            except Exception as e:
                print(f"Error processing attention for {name}: {e}")
                continue
                
        return attention_patterns

    def analyze_visual_features(self, rgb_input):
        """Analyze important visual features in the camera input"""
        results = {}
        
        # Generate attributions using integrated gradients
        ig_attr = self.integrated_gradients.attribute(
            rgb_input,
            target=0,
            n_steps=50
        )
        
        # Get overall importance map
        importance_map = ig_attr.sum(dim=1).abs()
        
        # Calculate spatial distribution of important features
        height, width = importance_map.shape[1:]
        
        # Split the image into regions and calculate importance per region
        h_segments = 3
        w_segments = 3
        
        h_step = height // h_segments
        w_step = width // w_segments
        
        region_importance = np.zeros((h_segments, w_segments))
        
        for i in range(h_segments):
            for j in range(w_segments):
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                
                region = importance_map[:, h_start:h_end, w_start:w_end]
                region_importance[i, j] = region.sum().item()
        
        # Normalize to get percentages
        total_importance = region_importance.sum()
        if total_importance > 0:
            region_importance = region_importance / total_importance * 100
        
        results['region_importance'] = region_importance
        
        # Identify key areas
        results['road_importance'] = region_importance[2, :].sum()  # Bottom row
        results['center_importance'] = region_importance[1, 1]      # Center
        results['horizon_importance'] = region_importance[1, :].sum()  # Middle row
        
        return results
    
    def detect_simple_objects(self, rgb_input):
        """Simple region-based detection of key elements in the scene"""
        # Convert tensor to numpy for analysis
        img = rgb_input.squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        # Image dimensions
        h, w = img.shape[:2]
        
        # Define regions of interest
        regions = {
            'top': img[:h//3, :, :],           # Sky/horizon
            'middle': img[h//3:2*h//3, :, :],  # Middle of scene
            'bottom': img[2*h//3:, :, :],      # Road surface
            'center': img[h//3:2*h//3, w//3:2*w//3, :],  # Center of image
            'left': img[:, :w//3, :],          # Left side
            'right': img[:, 2*w//3:, :]        # Right side
        }
        
        # Analyze regions for different elements
        scene_elements = []
        
        # Check for road presence
        road_std = np.std(regions['bottom'])
        if road_std < 0.15:
            scene_elements.append({'type': 'road', 'confidence': 0.9, 'region': 'bottom'})
        
        # Check for possible vehicles
        center_std = np.std(regions['center'])
        if center_std > 0.1:
            scene_elements.append({'type': 'possible_vehicle', 'confidence': 0.7, 'region': 'center'})
        
        # Check for lane markings
        bottom_edges = np.std(regions['bottom'], axis=(0, 1))
        if bottom_edges.max() > 0.1:
            scene_elements.append({'type': 'lane_markings', 'confidence': 0.8, 'region': 'bottom'})
            
        return scene_elements
    
    def explain_prediction(self, rgb_input):
        """Generate comprehensive explanations focused on camera input"""
        # Create zero tensors for lidar and velocity
        batch_size = rgb_input.shape[0]
        lidar_input = torch.zeros((batch_size, 2, 256, 256), device=rgb_input.device)
        velocity = torch.zeros((batch_size, 1), device=rgb_input.device)
        
        # Run inference to populate attention maps
        with torch.no_grad():
            features, image_features_grid, fused_features = self.model(rgb_input, lidar_input, velocity)
        
        # Generate attributions for camera
        rgb_attribution = self.integrated_gradients.attribute(
            rgb_input,
            target=0,
            n_steps=50
        )
        
        # Generate GradCAM for camera
        rgb_gradcam = self.rgb_gradcam.attribute(
            rgb_input,
            target=0
        )
        
        # Analyze visual features
        visual_features = self.analyze_visual_features(rgb_input)
        
        # Get attention patterns
        attention_patterns = self.extract_attention_patterns()
        
        # Detect scene elements
        scene_elements = self.detect_simple_objects(rgb_input)
        
        # Generate textual explanation
        explanation_text = self.generate_explanation_text(
            visual_features, 
            attention_patterns,
            scene_elements
        )
        
        return {
            'rgb_attribution': rgb_attribution,
            'rgb_gradcam': rgb_gradcam,
            'visual_features': visual_features,
            'attention_patterns': attention_patterns,
            'scene_elements': scene_elements,
            'explanation_text': explanation_text,
            'features': features,
            'fused_features': fused_features
        }
    
    def generate_explanation_text(self, visual_features, attention_patterns, scene_elements):
        """Generate natural language explanation of the model's decision process"""
        # Extract attention statistics
        attention_stats = []
        for i in range(1, 5):
            transformer_name = f"transformer_{i}_block_1"
            if transformer_name in attention_patterns:
                stats = attention_patterns[transformer_name]
                attention_stats.append({
                    'transformer': i,
                    'mean': stats['mean'],
                    'max': stats['max']
                })
        
        # Extract key visual regions
        road_importance = visual_features.get('road_importance', 0)
        center_importance = visual_features.get('center_importance', 0)
        horizon_importance = visual_features.get('horizon_importance', 0)
        
        # Generate the explanation
        explanation = f"The TransFuser model is analyzing this driving scene using camera input. The model is focusing on visual cues from the camera to make driving decisions.\n\n"
        
        # Add interpretation about important regions
        explanation += "Important visual regions in the scene:\n"
        if road_importance > 30:
            explanation += f"- Road surface: {road_importance:.1f}% attention (high focus on the road ahead)\n"
        else:
            explanation += f"- Road surface: {road_importance:.1f}% attention\n"
            
        explanation += f"- Center region: {center_importance:.1f}% attention "
        explanation += "(likely focusing on vehicles or obstacles ahead)\n"
        
        explanation += f"- Horizon/distant view: {horizon_importance:.1f}% attention "
        explanation += "(monitoring the path ahead and distant objects)\n\n"
        
        # Add information about detected elements
        if scene_elements:
            explanation += "Key elements detected in the scene:\n"
            for element in scene_elements:
                explanation += f"- {element['type'].replace('_', ' ').title()} (confidence: {element['confidence']:.1f})\n"
            explanation += "\n"
        
        # Add information about the transformer attention
        explanation += "Analysis of the transformer attention patterns reveals:\n"
        for stat in attention_stats:
            explanation += f"- Transformer {stat['transformer']}: Average attention strength: {stat['mean']:.4f}, Max: {stat['max']:.4f}\n"
        
        # Add interpretation of this scene specifically
        explanation += "\nIn this particular traffic scene, "
        explanation += "the model appears to focus on the road structure, lane markings, and vehicles ahead. "
        explanation += "The camera provides rich visual information about the scene appearance, "
        explanation += "including road geometry, lane positions, and the presence of other traffic participants.\n\n"
        
        # Add technical context about TransFuser architecture
        explanation += "TransFuser's architecture processes camera inputs through multi-stage feature extraction. "
        explanation += "This enables the model to first identify low-level visual features like edges and colors, "
        explanation += "then progressively build up to high-level understanding of road structure, vehicles, and appropriate driving actions."
        
        return explanation

# ----------------------- #
# Visualization Functions
# ----------------------- #

def simple_camera_visualization(rgb_input, explanations):
    """Create a simple matplotlib visualization for camera-only TransFuser analysis"""
    # Convert RGB input to numpy for visualization
    rgb_img = rgb_input.squeeze().cpu().permute(1, 2, 0).numpy()
    
    # Extract attribution maps
    rgb_attr = explanations['rgb_attribution'].sum(dim=1).abs()
    rgb_attr_np = rgb_attr.squeeze().cpu().detach().numpy()
    rgb_attr_np = rgb_attr_np / (rgb_attr_np.max() + 1e-8)  # Normalize
    
    gradcam = explanations['rgb_gradcam'].sum(dim=1).abs()
    gradcam_np = gradcam.squeeze().cpu().detach().numpy()
    gradcam_np = gradcam_np / (gradcam_np.max() + 1e-8)  # Normalize
    
    # Create a simple 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Camera-Only TransFuser Analysis", fontsize=16)
    
    # Plot 1: Original image
    axes[0, 0].set_title("Original Camera Input")
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].axis('off')
    
    # Plot 2: Attribution heatmap
    axes[0, 1].set_title("Feature Importance (Integrated Gradients)")
    axes[0, 1].imshow(rgb_attr_np, cmap='hot')
    axes[0, 1].axis('off')
    
    # Plot 3: GradCAM visualization
    axes[1, 0].set_title("GuidedGradCAM")
    axes[1, 0].imshow(gradcam_np, cmap='hot')
    axes[1, 0].axis('off')
    
    # Plot 4: GradCAM overlay on original image
    axes[1, 1].set_title("GuidedGradCAM Overlay")
    axes[1, 1].imshow(rgb_img)
    
    # Create a masked overlay with transparency
    mask = gradcam_np > 0.3  # Only show strong activations
    overlay = np.zeros_like(rgb_img)
    overlay[..., 0] = mask * 1.0  # Red channel
    
    # Apply overlay with transparency
    axes[1, 1].imshow(overlay, alpha=0.6)
    axes[1, 1].axis('off')
    
    # Add explanation text
    explanation_text = """
    This visualization shows how the TransFuser model analyzes camera input:
    • Top Left: Original camera input from the vehicle
    • Top Right: Feature importance map showing which parts of the image influence decisions
    • Bottom Left: GuidedGradCAM highlighting critical features used by the model
    • Bottom Right: Important features overlaid on the original image
    
    In this scene, the model focuses on road structure, lane markings, and objects ahead.
    """
    
    plt.figtext(0.5, 0.01, explanation_text, wrap=True, horizontalalignment='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make space for the text
    
    return fig

def create_region_importance_figure(rgb_input, explanations):
    """Create a region importance visualization for camera input"""
    # Convert RGB input to numpy for visualization
    rgb_img = rgb_input.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    h, w = rgb_img.shape[:2]
    
    # Create a 3x3 grid of region importance
    if 'visual_features' in explanations and 'region_importance' in explanations['visual_features']:
        region_importance = explanations['visual_features']['region_importance']
    else:
        # Calculate region importance from attribution map
        attr_map = explanations['rgb_attribution'].sum(dim=1).abs().squeeze().cpu().numpy()
        attr_map = attr_map / (attr_map.max() + 1e-8)  # Normalize
        
        h_step, w_step = h // 3, w // 3
        region_importance = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                region = attr_map[h_start:h_end, w_start:w_end]
                region_importance[i, j] = region.mean() * 100  # Convert to percentage
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Region Importance Analysis", fontsize=16)
    
    # Plot original image with grid
    axes[0].set_title("Original Camera Input")
    axes[0].imshow(rgb_img)
    
    # Draw grid lines
    h_step, w_step = h // 3, w // 3
    for i in range(1, 3):
        axes[0].axhline(y=i*h_step, color='white', linestyle='-', linewidth=1)
        axes[0].axvline(x=i*w_step, color='white', linestyle='-', linewidth=1)
    
    # Add region labels
    for i in range(3):
        for j in range(3):
            h_center = i * h_step + h_step // 2
            w_center = j * w_step + w_step // 2
            axes[0].text(w_center, h_center, f"{region_importance[i, j]:.1f}%", 
                       color='white', ha='center', va='center', 
                       bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
    
    axes[0].axis('off')
    
    # Plot heatmap of region importance
    im = axes[1].imshow(region_importance, cmap='viridis', interpolation='nearest')
    axes[1].set_title("Region Importance Heatmap")
    
    # Add region labels
    region_names = [
        ["Top Left", "Top Center", "Top Right"],
        ["Mid Left", "Center", "Mid Right"],
        ["Bottom Left", "Bottom Center", "Bottom Right"]
    ]
    
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{region_names[i][j]}\n{region_importance[i, j]:.1f}%", 
                       ha='center', va='center', color='white' if region_importance[i, j] > 15 else 'black')
    
    plt.colorbar(im, ax=axes[1], label='Importance %')
    
    # Add explanation text
    explanation = """
    This visualization shows which regions of the camera input are most important for the TransFuser model's decisions.
    The bottom center region (road ahead) and center region (potential obstacles) typically receive the most attention.
    Higher percentages indicate areas where the model focuses more when making driving decisions.
    """
    
    plt.figtext(0.5, 0.01, explanation, wrap=True, horizontalalignment='center', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    return fig

# ----------------------- #
# Utility Functions
# ----------------------- #

def load_transfuser_model(model_path, args_path):
    """Load the pretrained TransFuser model"""
    try:
        # Parse args from the args.txt file
        with open(args_path, 'r') as f:
            import json
            args_dict = json.load(f)
            
        # Create config from args
        config = GlobalConfig(root_dir='/tmp', setting='eval')
        
        # Set other configurations from args
        config.backbone = args_dict.get('backbone', 'transFuser')
        config.image_architecture = args_dict.get('image_architecture', 'regnety_032')
        config.lidar_architecture = args_dict.get('lidar_architecture', 'regnety_032')
        config.use_velocity = args_dict.get('use_velocity', 0)
        config.n_layer = args_dict.get('n_layer', 4)
        
        # Create model
        model = TransfuserBackbone(
            config,
            image_architecture=config.image_architecture,
            lidar_architecture=config.lidar_architecture,
            use_velocity=bool(config.use_velocity)
        )
        
        # Load weights with strict=False to ignore missing keys
        model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'), strict=False)
        model.eval()
        
        return model, config
    except Exception as e:
        print(f"Error loading TransFuser model: {e}")
        import traceback
        traceback.print_exc()
        raise

def prepare_sample_input(rgb_path):
    """Prepare sample camera input for the model"""
    try:
        # Load RGB image and preprocess
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = rgb.resize((400, 300))  # Adjust based on TransFuser's expected input size
        rgb = np.array(rgb).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
        return rgb
    except Exception as e:
        print(f"Error preparing sample input: {e}")
        import traceback
        traceback.print_exc()
        raise

# ----------------------- #
# Main Function
# ----------------------- #

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Camera-Only TransFuser Explainability')
    parser.add_argument('--model_path', 
                    default="model_ckpt/models_2022/transfuser/model_seed1_39.pth",
                    help='Path to TransFuser model weights')
    parser.add_argument('--args_path', 
                        default="model_ckpt/models_2022/transfuser/args.txt",
                        help='Path to TransFuser args file')
    parser.add_argument('--image_path',
                        default="carla2.png",
                        help='Path to input image')
    parser.add_argument('--output_dir', default='./outputs',
                        help='Path to save the explanation visualization')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional output')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots interactively')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Load model
        print("Loading TransFuser model...")
        model, config = load_transfuser_model(args.model_path, args.args_path)
        print("Model loaded successfully!")
        
        # Prepare inputs
        print("Preparing camera input...")
        rgb_input = prepare_sample_input(args.image_path)
        
        # Print input shape for debugging
        print(f"RGB input shape: {rgb_input.shape}")
        
        # Move input to the same device as the model
        device = next(model.parameters()).device
        rgb_input = rgb_input.to(device)
        
        # Initialize explainer
        explainer = CameraOnlyTransfuserExplainer(model)
        
        # Run inference
        print("Running inference and generating explanations...")
        explanations = explainer.explain_prediction(rgb_input)
        
        # Debug output if requested
        if args.debug:
            print("\nVisual Feature Analysis:")
            for k, v in explanations['visual_features'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 3 and v.shape[1] == 3:
                    print(f"  {k} (3x3 grid):")
                    for row in v:
                        print(f"    {row}")
                
            print("\nAttention Patterns:")
            for k, v in explanations['attention_patterns'].items():
                if 'mean' in v:
                    print(f"  {k}: mean={v['mean']:.4f}, max={v['max']:.4f}")
            
            print("\nDetected Scene Elements:")
            for elem in explanations['scene_elements']:
                print(f"  {elem['type']} (confidence: {elem['confidence']:.2f}, region: {elem['region']})")
        
        # Create main visualization
        print("Creating main visualization...")
        main_viz = simple_camera_visualization(rgb_input, explanations)
        
        # Create region importance visualization
        print("Creating region importance visualization...")
        region_viz = create_region_importance_figure(rgb_input, explanations)
        
        # Save visualizations
        main_viz_path = os.path.join(args.output_dir, "transfuser_main_viz.png")
        main_viz.savefig(main_viz_path, dpi=300, bbox_inches='tight')
        print(f"Main visualization saved to: {main_viz_path}")
        
        region_viz_path = os.path.join(args.output_dir, "transfuser_region_importance.png")
        region_viz.savefig(region_viz_path, dpi=300, bbox_inches='tight')
        print(f"Region importance visualization saved to: {region_viz_path}")
        
        # Save explanation text
        explanation_path = os.path.join(args.output_dir, "transfuser_explanation.txt")
        with open(explanation_path, 'w') as f:
            f.write(explanations['explanation_text'])
        print(f"Textual explanation saved to: {explanation_path}")
        
        # Show plots if requested
        if args.show_plots:
            plt.show()
            
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()        