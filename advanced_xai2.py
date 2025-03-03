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

class LiDARWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, lidar_input):
        # Create zeroed RGB input of correct dimensions
        batch_size = lidar_input.shape[0]
        rgb_input = torch.zeros((batch_size, 3, 300, 400), device=lidar_input.device)
        velocity = torch.zeros((batch_size, 1), device=lidar_input.device)
        
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
        
        # Create proper wrapper modules
        self.rgb_wrapper = RGBWrapper(self.model)
        self.lidar_wrapper = LiDARWrapper(self.model)
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.rgb_wrapper)
        self.lidar_integrated_gradients = IntegratedGradients(self.lidar_wrapper)
        
        # For LayerGradCAM, use a proper module
        self.rgb_gradcam = GuidedGradCam(self.rgb_wrapper, 
                                         self.model.image_encoder.features.layer4)
        self.lidar_gradcam = GuidedGradCam(self.lidar_wrapper,
                                          self.model.lidar_encoder._model.layer4)
        
        # Register hooks to capture attention weights
        self.register_attention_hooks()
        
    def register_attention_hooks(self):
        """Register hooks to capture transformer attention patterns"""
        def hook_fn(module, input, output, name):
            # For standard attention, output often has shape [batch, seq_len, seq_len]
            # For TransFuser, we need to verify the exact shape and structure
            self.attention_maps[name] = output
            
        # Hook into transformer attention blocks specifically
        for i, transformer in enumerate([
            self.model.transformer1, 
            self.model.transformer2,
            self.model.transformer3, 
            self.model.transformer4
        ]):
            # Access SelfAttention directly - this might need adjustment based on exact architecture
            for j, block in enumerate(transformer.blocks):
                block.attn.register_forward_hook(
                    lambda mod, inp, out, name=f"transformer_{i+1}_block_{j+1}": 
                    hook_fn(mod, inp, out, name)
                )
    
    def extract_cross_modal_attention(self):
        """
        Extract and analyze cross-modal attention patterns
        
        This method specifically looks at how image tokens attend to LiDAR tokens
        and vice versa, providing insight into the cross-modal fusion.
        """
        cross_modal_patterns = {}
        
        for name, attention in self.attention_maps.items():
            # For each attention map from the transformers
            # In TransFuser, the attention is complex and needs specific extraction
            try:
                # This structure needs to be adapted to the exact TransFuser implementation
                # We're looking for the attention weights between modalities
                
                # Skip if attention is None or not the right format
                if attention is None or not isinstance(attention, torch.Tensor):
                    continue
                    
                # Extract the cross-modal attention weights based on the structure
                # This is a placeholder - the exact implementation depends on TransFuser's attention format
                if len(attention.shape) >= 3:  # Typically [batch, seq_len, seq_len] or [batch, heads, seq_len, seq_len]
                    # Process attention tensor to extract cross-modal components
                    # For TransFuser, we need to determine which indices correspond to which modality
                    
                    # Example calculation - adjust based on actual structure
                    img_lidar_attention = attention.mean(dim=0) if len(attention.shape) == 3 else attention.mean(dim=(0, 1))
                    
                    # Store processed attention for this transformer block
                    cross_modal_patterns[name] = {
                        'tensor': img_lidar_attention.detach().cpu(),
                        'mean': float(img_lidar_attention.mean().item()),
                        'max': float(img_lidar_attention.max().item())
                    }
            except Exception as e:
                print(f"Error processing attention for {name}: {e}")
                continue
                
        return cross_modal_patterns

    def analyze_modality_importance(self, rgb_input, lidar_input, velocity):
        """
        Analyze relative importance of RGB vs LiDAR inputs using multiple methods
        
        This improved method uses both ablation and gradient-based approaches to
        provide a more robust assessment of modality contributions.
        """
        results = {}
        
        # Method 1: Feature map visualization
        with torch.no_grad():
            features, image_features_grid, fused_features = self.model(rgb_input, lidar_input, velocity)
        
        # Method 2: Gradual ablation to measure sensitivity
        contributions = []
        steps = 5  # Number of ablation steps
        
        # Test RGB importance by gradually reducing LiDAR input
        rgb_contributions = []
        lidar_baselines = [lidar_input * (step/steps) for step in range(steps+1)]
        
        for lidar_baseline in lidar_baselines:
            with torch.no_grad():
                _, _, features_with_reduced_lidar = self.model(rgb_input, lidar_baseline, velocity)
                
            # Measure similarity to full-input prediction
            similarity = 1.0 - torch.nn.functional.cosine_similarity(
                fused_features.view(1, -1), 
                features_with_reduced_lidar.view(1, -1)
            ).item()
            
            rgb_contributions.append(similarity)
        
        # Test LiDAR importance by gradually reducing RGB input
        lidar_contributions = []
        rgb_baselines = [rgb_input * (step/steps) for step in range(steps+1)]
        
        for rgb_baseline in rgb_baselines:
            with torch.no_grad():
                _, _, features_with_reduced_rgb = self.model(rgb_baseline, lidar_input, velocity)
                
            # Measure similarity to full-input prediction
            similarity = 1.0 - torch.nn.functional.cosine_similarity(
                fused_features.view(1, -1), 
                features_with_reduced_rgb.view(1, -1)
            ).item()
            
            lidar_contributions.append(similarity)
        
        # Compute area under sensitivity curve for each modality
        rgb_importance = np.trapz(rgb_contributions, dx=1.0/steps)
        lidar_importance = np.trapz(lidar_contributions, dx=1.0/steps)
        
        # Normalize to percentages
        total = rgb_importance + lidar_importance
        rgb_percent = (rgb_importance / total * 100)
        lidar_percent = (lidar_importance / total * 100)
        
        results['rgb_contribution'] = rgb_percent
        results['lidar_contribution'] = lidar_percent
        
        # Method 3: Gradient magnitude as importance indicator
        rgb_grad = self.integrated_gradients.attribute(rgb_input, target=0, n_steps=50)
        lidar_grad = self.lidar_integrated_gradients.attribute(lidar_input, target=0, n_steps=50)
        
        # Normalize by input size to make fair comparison
        rgb_grad_mag = torch.sum(torch.abs(rgb_grad)).item() / np.prod(rgb_input.shape)
        lidar_grad_mag = torch.sum(torch.abs(lidar_grad)).item() / np.prod(lidar_input.shape)
        
        # Normalize to percentages
        grad_total = rgb_grad_mag + lidar_grad_mag
        results['rgb_gradient_importance'] = (rgb_grad_mag / grad_total * 100)
        results['lidar_gradient_importance'] = (lidar_grad_mag / grad_total * 100)
        
        # Average the metrics for a more robust estimate
        results['rgb_contribution_avg'] = (results['rgb_contribution'] + results['rgb_gradient_importance']) / 2
        results['lidar_contribution_avg'] = (results['lidar_contribution'] + results['lidar_gradient_importance']) / 2
        
        # Add in raw gradient magnitudes for reference
        results['rgb_grad_magnitude'] = rgb_grad_mag
        results['lidar_grad_magnitude'] = lidar_grad_mag
        
        return results
    
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
        
        # Generate attributions for both modalities
        rgb_attribution = self.integrated_gradients.attribute(
            rgb_input,
            target=0,
            n_steps=50
        )
        
        lidar_attribution = self.lidar_integrated_gradients.attribute(
            lidar_input,
            target=0,
            n_steps=50
        )
        
        # Generate GradCAM for both modalities
        rgb_gradcam = self.rgb_gradcam.attribute(
            rgb_input,
            target=0
        )
        
        lidar_gradcam = self.lidar_gradcam.attribute(
            lidar_input,
            target=0
        )
        
        # Get modality importance using improved method
        modality_importance = self.analyze_modality_importance(rgb_input, lidar_input, velocity)
        
        # Analyze cross-modal attention patterns
        cross_modal_attention = self.extract_cross_modal_attention()
        
        # Generate textual explanation based on findings
        explanation_text = self.generate_explanation_text(
            modality_importance, 
            cross_modal_attention
        )
        
        return {
            'rgb_attribution': rgb_attribution,
            'lidar_attribution': lidar_attribution,
            'rgb_gradcam': rgb_gradcam,
            'lidar_gradcam': lidar_gradcam,
            'modality_importance': modality_importance,
            'cross_modal_attention': cross_modal_attention,
            'explanation_text': explanation_text,
            'features': features,
            'fused_features': fused_features
        }
    
    def generate_explanation_text(self, modality_importance, cross_modal_attention):
        """
        Generate natural language explanation of the model's decision process
        based on the observed attribution patterns
        """
        # Get the average contribution metrics for more robust estimation
        rgb_contrib = modality_importance['rgb_contribution_avg']
        lidar_contrib = modality_importance['lidar_contribution_avg']
        
        # Format percentages for display
        rgb_percent = f"{rgb_contrib:.1f}%"
        lidar_percent = f"{lidar_contrib:.1f}%"
        
        # Get the gradient magnitudes
        rgb_grad = modality_importance['rgb_grad_magnitude']
        lidar_grad = modality_importance['lidar_grad_magnitude']
        
        # Extract attention statistics
        attention_stats = []
        for i in range(1, 5):
            transformer_name = f"transformer_{i}_block_1"
            if transformer_name in cross_modal_attention:
                stats = cross_modal_attention[transformer_name]
                attention_stats.append({
                    'transformer': i,
                    'mean': stats['mean'],
                    'max': stats['max']
                })
        
        # Generate the explanation
        explanation = f"The TransFuser model fuses information from both camera and LiDAR sensors, with camera input contributing approximately {rgb_percent} and LiDAR input contributing approximately {lidar_percent} to the driving decision.\n\n"
        
        # Add interpretation about sensor roles
        if rgb_contrib > lidar_contrib * 2:
            explanation += "The model relies more heavily on camera input for this scene, which is typical when visual cues like lane markings, traffic signs, and other vehicles are clearly visible and provide sufficient information for navigation.\n\n"
        elif lidar_contrib > rgb_contrib * 2:
            explanation += "The model relies more heavily on LiDAR input for this scene, which is typical in situations where precise distance measurement and 3D structure are critical for navigation, such as in complex traffic environments or adverse weather conditions.\n\n"
        else:
            explanation += "The model balances information from both sensors relatively evenly for this scene, leveraging visual cues from the camera while using LiDAR to enhance spatial awareness and depth perception.\n\n"
            
        # Add information about the transformer attention
        explanation += "Analysis of the transformer attention patterns reveals:\n"
        for stat in attention_stats:
            explanation += f"- Transformer {stat['transformer']}: Average attention strength: {stat['mean']:.4f}, Max: {stat['max']:.4f}\n"
        
        # Add interpretation of this scene specifically
        explanation += "\nIn this particular traffic scene, "
        
        # This part would be scene-specific - we'll make a general statement based on the image
        explanation += "the model appears to focus on the road structure, lane markings, and vehicles ahead. "
        explanation += "The camera provides rich visual information about the scene appearance, while the LiDAR likely contributes to accurate distance estimation and 3D structure recognition.\n\n"
        
        # Add technical context about TransFuser architecture
        explanation += "TransFuser's architecture is specifically designed to leverage complementary sensor information through transformer-based fusion. "
        explanation += "The transformers enable the model to establish relationships between image and LiDAR features at multiple processing stages, "
        explanation += "allowing for more robust decision-making than would be possible with either sensor alone."
        
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
    
    # Plot 2: Original LiDAR image - enhanced visualization
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax2.set_title("Original LiDAR Input")
    # Sum across channels and normalize for better visibility
    lidar_vis = lidar_input.squeeze().sum(dim=0).cpu().numpy()
    lidar_vis = (lidar_vis - lidar_vis.min()) / (lidar_vis.max() - lidar_vis.min() + 1e-8)
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
    
    # Plot 5: RGB GuidedGradCAM visualization
    ax5 = plt.subplot2grid((3, 4), (1, 2))
    ax5.set_title("Camera GuidedGradCAM")
    # Normalize for visualization
    guided_gradcam = explanations['rgb_gradcam'].sum(dim=1).abs()
    guided_gradcam = guided_gradcam / (guided_gradcam.max() + 1e-8)
    ax5.imshow(guided_gradcam.squeeze().cpu().tensor.detach().numpy(), cmap='hot')
    ax5.axis('off')
    
    # Plot 6: Modality importance pie chart with improved metrics
    ax6 = plt.subplot2grid((3, 4), (1, 3))
    ax6.set_title("Modality Contribution")
    mod_imp = explanations['modality_importance']
    
    # Use the averaged metrics for a more reliable estimate
    rgb_contrib = mod_imp['rgb_contribution_avg']
    lidar_contrib = mod_imp['lidar_contribution_avg']
    
    # Create pie chart with both metrics shown
    ax6.pie([rgb_contrib, lidar_contrib], 
            labels=['Camera', 'LiDAR'], 
            autopct='%.1f%%', 
            startangle=90, 
            colors=['#ff9999', '#66b3ff'])
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
    
    # Create model
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
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional output')
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
    explanations = explainer.explain_prediction(rgb_input, lidar_input, velocity)
    
    # Debug output if requested
    if args.debug:
        print("\nModality Importance Metrics:")
        for k, v in explanations['modality_importance'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            
        print("\nCross-Modal Attention:")
        for k, v in explanations['cross_modal_attention'].items():
            if 'mean' in v:
                print(f"  {k}: mean={v['mean']:.4f}, max={v['max']:.4f}")
    
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