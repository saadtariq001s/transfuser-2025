import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import argparse
from captum.attr import IntegratedGradients, GuidedGradCam, LayerAttribution

# Add the TransFuser path to sys.path
sys.path.append("./team_code_transfuser")

# Import TransFuser modules
from team_code_transfuser.transfuser import TransfuserBackbone
from team_code_transfuser.config import GlobalConfig

class RGBWrapper(torch.nn.Module):
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

class TransfuserExplainer:
    def __init__(self, model):
        """
        Advanced explainability system for TransFuser
        
        Args:
            model: TransfuserBackbone model instance
        """
        self.model = model
        self.model.eval()
        self.attention_maps = {}
        
        # Register hooks to capture attention weights
        self.register_attention_hooks()
        
        # Create proper wrapper modules
        self.rgb_wrapper = RGBWrapper(self.model)
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.rgb_wrapper)
        self.lidar_integrated_gradients = IntegratedGradients(self.forward_wrapper_lidar)
        
        # For LayerGradCAM, we need to use a proper module
        # Instead of transformer4 which might be complex, use a simpler target layer
        self.guided_gradcam = GuidedGradCam(self.rgb_wrapper, 
                                            self.model.image_encoder.features.layer4)

    # def forward_wrapper_rgb(self, rgb_input):
    #     """Wrapper for RGB input attribution"""
    #     # Create zeroed lidar input of correct dimensions
    #     batch_size = rgb_input.shape[0]
    #     lidar_input = torch.zeros((batch_size, 2, 256, 256), device=rgb_input.device)
    #     velocity = torch.zeros((batch_size, 1), device=rgb_input.device)
        
    #     _, _, fused_features = self.model(rgb_input, lidar_input, velocity)
    #     return fused_features

    def forward_wrapper_lidar(self, lidar_input):
        """Wrapper for LiDAR input attribution"""
        # Create zeroed RGB input of correct dimensions
        batch_size = lidar_input.shape[0]
        rgb_input = torch.zeros((batch_size, 3, 300, 400), device=lidar_input.device)
        velocity = torch.zeros((batch_size, 1), device=lidar_input.device)
        
        _, _, fused_features = self.model(rgb_input, lidar_input, velocity)
        return fused_features
    
    def register_attention_hooks(self):
        """Register forward hooks to capture attention patterns"""
        def hook_fn(module, input, output, name):
            # Store attention weights
            self.attention_maps[name] = output.detach()
            
        # Hook into each transformer's attention blocks
        transformers = [self.model.transformer1, self.model.transformer2, 
                        self.model.transformer3, self.model.transformer4]
        
        for i, transformer in enumerate(transformers):
            for j, block in enumerate(transformer.blocks):
                block.attn.register_forward_hook(
                    lambda mod, inp, out, name=f"transformer_{i+1}_block_{j+1}": 
                    hook_fn(mod, inp, out, name)
                )

    def explain_prediction(self, rgb_input, lidar_input, velocity):
        """
        Generate comprehensive explanations for TransFuser's decisions
        
        Args:
            rgb_input: Camera input tensor
            lidar_input: LiDAR BEV input tensor
            velocity: Velocity input tensor
        
        Returns:
            Dictionary containing various explanations
        """
        # Run inference to populate attention maps
        with torch.no_grad():
            features, image_features_grid, fused_features = self.model(rgb_input, lidar_input, velocity)
        
        # Generate integrated gradients attribution for RGB input
        rgb_attribution = self.integrated_gradients.attribute(
            rgb_input,
            target=0,  # Target first output dimension
            n_steps=50
        )
        
        # Generate integrated gradients attribution for LiDAR input
        lidar_attribution = self.lidar_integrated_gradients.attribute(
            lidar_input,
            target=0,  # Target first output dimension
            n_steps=50
        )
        
        # Generate GuidedGradCAM for RGB
        guided_gradcam_attr = self.guided_gradcam.attribute(
            rgb_input,
            target=0
        )
        
        # Get modality importance
        modality_importance = self.analyze_modality_importance(rgb_input, lidar_input, velocity)
        
        # Get attention visualizations
        attention_analysis = self.analyze_attention_patterns()
        
        return {
            'rgb_attribution': rgb_attribution,
            'lidar_attribution': lidar_attribution,
            'guided_gradcam': guided_gradcam_attr,
            'modality_importance': modality_importance,
            'attention_analysis': attention_analysis,
            'features': features,
            'fused_features': fused_features
        }
    
    def analyze_modality_importance(self, rgb_input, lidar_input, velocity):
        """Analyze relative importance of RGB vs LiDAR inputs"""
        # Original prediction
        with torch.no_grad():
            _, _, fused_features = self.model(rgb_input, lidar_input, velocity)
            
        # Prediction with zeroed RGB input
        rgb_zeroed = torch.zeros_like(rgb_input)
        with torch.no_grad():
            _, _, lidar_only_features = self.model(rgb_zeroed, lidar_input, velocity)
            
        # Prediction with zeroed LiDAR input
        lidar_zeroed = torch.zeros_like(lidar_input)
        with torch.no_grad():
            _, _, rgb_only_features = self.model(rgb_input, lidar_zeroed, velocity)
            
        # Calculate contribution percentages
        rgb_contribution = torch.abs(fused_features - lidar_only_features).sum()
        lidar_contribution = torch.abs(fused_features - rgb_only_features).sum()
        total = rgb_contribution + lidar_contribution
        
        rgb_percent = (rgb_contribution / total * 100).item()
        lidar_percent = (lidar_contribution / total * 100).item()
        
        return {
            'rgb_contribution': rgb_percent,
            'lidar_contribution': lidar_percent
        }
    
    def analyze_attention_patterns(self):
        """Extract and analyze attention patterns from transformer blocks"""
        # This will contain attention analysis results
        results = {}
        
        # For each transformer
        for i in range(1, 5):
            transformer_name = f"transformer_{i}"
            block_results = []
            
            # For each block in the transformer
            for j in range(1, 5):  # Assume up to 4 blocks
                block_name = f"{transformer_name}_block_{j}"
                
                if block_name in self.attention_maps:
                    attn_map = self.attention_maps[block_name]
                    
                    # Process attention map
                    # This is a simplified analysis - you may need to adapt based on exact structure
                    avg_attention = attn_map.mean(dim=0)  # Average across batch
                    
                    # Store results
                    block_results.append({
                        'name': block_name,
                        'attention_stat': {
                            'mean': float(avg_attention.mean().item()),
                            'max': float(avg_attention.max().item()),
                            'min': float(avg_attention.min().item())
                        }
                    })
            
            results[transformer_name] = block_results
        
        return results
    
    def generate_explanation_text(self, explanations):
        """Generate natural language explanation of the model's decision process"""
        modality_importance = explanations['modality_importance']
        
        # Determine the primary modality
        primary_modality = "camera" if modality_importance['rgb_contribution'] > modality_importance['lidar_contribution'] else "LiDAR"
        secondary_modality = "LiDAR" if primary_modality == "camera" else "camera"
        
        primary_percent = modality_importance['rgb_contribution'] if primary_modality == "camera" else modality_importance['lidar_contribution']
        secondary_percent = modality_importance['lidar_contribution'] if primary_modality == "camera" else modality_importance['rgb_contribution']
        
        # Generate explanation text
        explanation = f"The TransFuser model's decision is primarily based on {primary_modality} input "
        explanation += f"({primary_percent:.1f}% contribution), with secondary influence from {secondary_modality} data "
        explanation += f"({secondary_percent:.1f}% contribution).\n\n"
        
        # Add attention-based details
        explanation += "The model's transformer attention patterns show:\n"
        
        for i in range(1, 5):
            transformer_name = f"transformer_{i}"
            if transformer_name in explanations['attention_analysis']:
                blocks = explanations['attention_analysis'][transformer_name]
                if blocks:
                    block = blocks[0]  # Just use the first block for simplicity
                    explanation += f"- Transformer {i}: Average attention strength {block['attention_stat']['mean']:.4f}\n"
        
        # Add suggestions for potential improvements or insights
        explanation += "\nFor optimal driving decisions, the model fuses information from both sensors, "
        explanation += "allowing it to leverage the complementary strengths of visual data (appearance, color, texture) "
        explanation += "and LiDAR data (precise distance measurement, 3D structure)."
        
        return explanation


def visualize_explanations(explanations, rgb_input, lidar_input):
    """
    Create comprehensive visualization of TransFuser explanations
    
    Args:
        explanations: Dictionary from explain_prediction method
        rgb_input: Original RGB input
        lidar_input: Original LiDAR input
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original RGB image
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    ax1.set_title("Original Camera Input")
    ax1.imshow(rgb_input.squeeze().cpu().permute(1, 2, 0).numpy())
    ax1.axis('off')
    
    # Plot 2: Original LiDAR image
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax2.set_title("Original LiDAR Input")
    # Sum across channels to make visualization simpler
    lidar_vis = lidar_input.squeeze().sum(dim=0).cpu().numpy()
    ax2.imshow(lidar_vis, cmap='viridis')
    ax2.axis('off')
    
    # Plot 3: RGB Attribution (Integrated Gradients)
    ax3 = plt.subplot2grid((3, 4), (1, 0))
    ax3.set_title("Camera Input Attribution\n(Integrated Gradients)")
    rgb_attr = explanations['rgb_attribution'].sum(dim=1).abs()
    rgb_attr = rgb_attr / (rgb_attr.max() + 1e-8)
    ax3.imshow(rgb_attr.squeeze().cpu().numpy(), cmap='hot')
    ax3.axis('off')
    
    # Plot 4: LiDAR Attribution
    ax4 = plt.subplot2grid((3, 4), (1, 1))
    ax4.set_title("LiDAR Input Attribution\n(Integrated Gradients)")
    lidar_attr = explanations['lidar_attribution'].sum(dim=1).abs()
    lidar_attr = lidar_attr / (lidar_attr.max() + 1e-8)
    ax4.imshow(lidar_attr.squeeze().cpu().numpy(), cmap='hot')
    ax4.axis('off')
    
    # Plot 5: GuidedGradCAM visualization
    ax5 = plt.subplot2grid((3, 4), (1, 2))
    ax5.set_title("Camera GuidedGradCAM")
    # Normalize for visualization
    guided_gradcam = explanations['guided_gradcam'].sum(dim=1).abs()
    guided_gradcam = guided_gradcam / (guided_gradcam.max() + 1e-8)
    ax5.imshow(guided_gradcam.squeeze().cpu().numpy(), cmap='hot')
    ax5.axis('off')
    
    # Plot 6: Modality importance pie chart
    ax6 = plt.subplot2grid((3, 4), (1, 3))
    ax6.set_title("Modality Contribution")
    mod_imp = explanations['modality_importance']
    ax6.pie([mod_imp['rgb_contribution'], mod_imp['lidar_contribution']], 
            labels=['Camera', 'LiDAR'], autopct='%1.1f%%', 
            startangle=90, colors=['#ff9999', '#66b3ff'])
    ax6.axis('equal')
    
    # Plot 7: Textual explanation
    ax7 = plt.subplot2grid((3, 4), (2, 0), colspan=4)
    ax7.axis('off')
    explanation_text = explanations.get('explanation_text', 
                                       "Explanation text not available.")
    ax7.text(0.5, 0.5, explanation_text, 
             ha='center', va='center', wrap=True, fontsize=12,
             bbox=dict(boxstyle="round,pad=1", facecolor='#f0f0f0'))
    
    plt.tight_layout()
    return fig


def load_transfuser_model(model_path, args_path):
    """
    Load the pretrained TransFuser model
    """
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
    
    # Create model - note the parameters needed for TransfuserBackbone
    model = TransfuserBackbone(
        config,
        image_architecture=config.image_architecture,
        lidar_architecture=config.lidar_architecture,
        use_velocity=bool(config.use_velocity)
    )
    
    # Load weights with strict=False to ignore missing keys
    model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True), strict=False)
    model.eval()
    
    return model, config


def prepare_sample_input(rgb_path, lidar_path=None):
    """
    Prepare sample inputs for the model
    """
    # Load RGB image and preprocess
    rgb = Image.open(rgb_path).convert('RGB')
    rgb = rgb.resize((400, 300))  # Adjust based on TransFuser's expected input size
    rgb = np.array(rgb).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # If no LiDAR input provided, create a dummy one
    if lidar_path is None:
        # Create dummy LiDAR input with correct dimensions
        lidar_bev = torch.zeros((1, 2, 256, 256))  # Using 2 channels as expected by the model
    else:
        # Load actual LiDAR data if available
        lidar_bev = torch.load(lidar_path)
        
    # Also create dummy velocity input since the model expects it
    velocity = torch.zeros((1, 1))
        
    return rgb, lidar_bev, velocity


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='TransFuser Explainability')
    parser.add_argument('--model_path', default="model_ckpt/models_2022/transfuser/model_seed1_39.pth",
                        help='Path to TransFuser model weights')
    parser.add_argument('--args_path', default="model_ckpt/models_2022/transfuser/args.txt",
                        help='Path to TransFuser args file')
    parser.add_argument('--image_path', default="carla.png",
                        help='Path to input image')
    parser.add_argument('--output_path', default="transfuser_explanation.png",
                        help='Path to save the explanation visualization')
    args = parser.parse_args()
    
    # Load model
    print("Loading TransFuser model...")
    model, config = load_transfuser_model(args.model_path, args.args_path)
    print("Model loaded successfully!")
    
    # Prepare inputs
    print("Preparing sample inputs...")
    rgb_input, lidar_input, velocity = prepare_sample_input(args.image_path)
    
    # Print input shapes for debugging
    print(f"RGB input shape: {rgb_input.shape}")
    print(f"LiDAR input shape: {lidar_input.shape}")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    rgb_input = rgb_input.to(device)
    lidar_input = lidar_input.to(device)
    velocity = velocity.to(device)
    
    # Initialize explainer
    explainer = TransfuserExplainer(model)
    
    # Run inference
    print("Running inference and generating explanations...")
    with torch.no_grad():
        explanations = explainer.explain_prediction(rgb_input, lidar_input, velocity)
    
    # Generate textual explanation
    explanations['explanation_text'] = explainer.generate_explanation_text(explanations)
    
    # Create visualization
    print("Creating visualization...")
    fig = visualize_explanations(explanations, rgb_input, lidar_input)
    
    # Save or display visualization
    if args.output_path:
        fig.savefig(args.output_path)
        print(f"Explanation saved to {args.output_path}")
    
    plt.show()
    print("Done!")


if __name__ == "__main__":
    main()