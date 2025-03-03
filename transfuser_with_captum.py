import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import argparse

# Add the TransFuser path to sys.path
sys.path.append("./team_code_transfuser")

# Import TransFuser modules
from team_code_transfuser.transfuser import TransfuserBackbone
from team_code_transfuser.config import GlobalConfig

# Import Captum
from captum.attr import IntegratedGradients, GradientShap, LayerGradCam

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
    model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'), strict=False)
    model.eval()
    
    return model, config

def prepare_sample_input(rgb_path, lidar_path=None):
    """
    Prepare sample inputs for the model
    """
    # Load RGB image and preprocess
    rgb = Image.open(rgb_path).convert('RGB')  # Ensure RGB format
    rgb = rgb.resize((400, 300))  # Adjust based on TransFuser's expected input size
    rgb = np.array(rgb).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # Debug: check shape
    print(f"RGB input shape: {rgb.shape}")
    
    # If no LiDAR input provided, create a dummy one
    if lidar_path is None:
        # Create dummy LiDAR input with correct dimensions
        lidar_bev = torch.zeros((1, 2, 256, 256))  # Using 2 channels as expected by the model
    else:
        # Load actual LiDAR data if available
        lidar_bev = torch.load(lidar_path)
    
    # Debug: check shape
    print(f"LiDAR input shape: {lidar_bev.shape}")
        
    # Also create dummy velocity input since the model expects it
    velocity = torch.zeros((1, 1))
        
    return rgb, lidar_bev, velocity

def explain_transfuser_decision(model, rgb_input, lidar_input, velocity):
    """
    Generate explanations for TransFuser decisions using a simpler approach
    """
    # Create baseline inputs (all zeros)
    rgb_baseline = torch.zeros_like(rgb_input)
    lidar_baseline = torch.zeros_like(lidar_input)
    
    # Get the gradients manually
    rgb_input.requires_grad = True
    lidar_input.requires_grad = True
    
    # Forward pass
    output = model(rgb_input, lidar_input, velocity)
    
    # Backward pass for rgb
    model.zero_grad()
    output[:, 0].sum().backward(retain_graph=True)
    rgb_gradients = rgb_input.grad.clone()
    
    # Reset gradients
    model.zero_grad()
    rgb_input.grad = None
    lidar_input.grad = None
    
    # Backward pass for lidar
    output = model(rgb_input, lidar_input, velocity)
    output[:, 0].sum().backward()
    lidar_gradients = lidar_input.grad.clone()
    
    # Clean up
    rgb_input.requires_grad = False
    lidar_input.requires_grad = False
    
    return {
        'rgb_gradients': rgb_gradients,
        'lidar_gradients': lidar_gradients
    }

def visualize_attributions(attributions, original_input, title):
    """
    Visualize attributions overlaid on the original input
    """
    # For RGB attributions
    # Sum across color channels and take absolute value
    attribution = attributions.sum(dim=1).abs()
    
    # Normalize attribution
    attribution = attribution / (attribution.max() + 1e-8)
    
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.title("Original Input")
    plt.imshow(original_input.squeeze().permute(1, 2, 0).cpu().numpy())
    
    # Plot attribution
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(attribution.squeeze().cpu().numpy(), cmap='hot')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def main():
    # Paths to model and sample input
    model_path = "model_ckpt/models_2022/transfuser/model_seed1_39.pth"
    args_path = "model_ckpt/models_2022/transfuser/args.txt"
    
    # Sample image path - replace with an actual traffic image path
    sample_image_path = "carla.png"
    
    # Load model
    print("Loading TransFuser model...")
    model, config = load_transfuser_model(model_path, args_path)
    print("Model loaded successfully!")
    
    # Prepare inputs
    print("Preparing sample inputs...")
    rgb_input, lidar_input, velocity = prepare_sample_input(sample_image_path)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    rgb_input = rgb_input.to(device)
    lidar_input = lidar_input.to(device)
    velocity = velocity.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        features, image_features_grid, fused_features = model(rgb_input, lidar_input, velocity)
        # For simplicity, let's use fused_features as our prediction
        predicted_features = fused_features
    
    print("Prediction shape:", predicted_features.shape)
    
    # Define a simple wrapper to make Captum work with our model
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, rgb, lidar, velocity):
            _, _, fused_features = self.model(rgb, lidar, velocity)
            return fused_features
    
    wrapped_model = ModelWrapper(model)
    
   # Generate explanations
    print("Generating explanations...")
    explanations = explain_transfuser_decision(wrapped_model, rgb_input, lidar_input, velocity)

    # Visualize attributions
    print("Visualizing attributions...")
    visualize_attributions(
        explanations['rgb_gradients'],
        rgb_input,
        "RGB Gradients"
    )
    
    # You can similarly visualize the other attributions
    
    print("Done!")

if __name__ == "__main__":
    main()