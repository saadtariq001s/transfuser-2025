import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, GuidedGradCam

# 1. Define the Model
class TrafficDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Using a pre-trained ResNet50 and modifying it for traffic detection
        self.model = models.resnet50(pretrained=True)
        
        # Modify the final layer for traffic-specific classes
        num_traffic_classes = 10  # Number of traffic-related classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_traffic_classes)
        
        # Add custom layers for better traffic feature extraction
        self.model.layer4[0].conv1 = nn.Conv2d(
            1024, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        
    def forward(self, x):
        return self.model(x)

# 2. Load and Preprocess the Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # Save original image for visualization
    original_image = np.array(image)
    
    # Transform image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return input_tensor, original_image, original_size

# 3. Enhanced Visualization Function
def visualize_attribution(attribution, original_image, method="heat_map", class_name=""):
    attribution = attribution.sum(dim=1).squeeze().cpu().detach().numpy()
    
    plt.figure(figsize=(12, 6))
    
    if method == "heat_map":
        plt.subplot(1, 2, 1)
        plt.title(f'Original Image - Detecting {class_name}')
        plt.imshow(original_image)
        
        plt.subplot(1, 2, 2)
        plt.title(f'Attribution Heat Map - {class_name}')
        plt.imshow(attribution, cmap='hot')
        plt.colorbar()
        
    elif method == "overlay":
        plt.subplot(1, 2, 1)
        plt.title(f'Original Image - Detecting {class_name}')
        plt.imshow(original_image)
        
        plt.subplot(1, 2, 2)
        plt.title(f'Attribution Overlay - {class_name}')
        # Normalize attribution
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
        plt.imshow(original_image)
        plt.imshow(attribution, cmap='hot', alpha=0.5)

# 4. Enhanced Explainability Wrapper
class TrafficModelExplainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.guided_gradcam = GuidedGradCam(self.model, self.model.model.layer4[-1])
        
        # Traffic-specific class names
        self.traffic_classes = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic light', 'traffic sign', 
            'stop sign', 'parking meter'
        ]
    
    def explain_prediction(self, input_tensor, target_class=None):
        """
        Generate explanations for traffic object detection
        """
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Generate attributions
        ig_attr = self.integrated_gradients.attribute(
            input_tensor,
            target=target_class,
            n_steps=50
        )
        
        gc_attr = self.guided_gradcam.attribute(
            input_tensor,
            target=target_class
        )
        
        class_name = self.traffic_classes[target_class] if target_class < len(self.traffic_classes) else f"Class {target_class}"
        
        return {
            'integrated_gradients': ig_attr,
            'guided_gradcam': gc_attr,
            'target_class': target_class,
            'class_name': class_name
        }

# 5. Main Function
def main():
    # Initialize model and explainer
    model = TrafficDetectionModel()
    explainer = TrafficModelExplainer(model)
    
    # Load and preprocess an image
    image_path = "carla.png"  # Replace with your traffic scene image
    input_tensor, original_image, original_size = preprocess_image(image_path)
    
    # Generate explanations
    explanations = explainer.explain_prediction(input_tensor)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Integrated Gradients visualization
    plt.subplot(2, 1, 1)
    visualize_attribution(
        explanations['integrated_gradients'], 
        original_image,
        method="heat_map",
        class_name=explanations['class_name']
    )
    
    # Guided GradCAM visualization
    plt.subplot(2, 1, 2)
    visualize_attribution(
        explanations['guided_gradcam'], 
        original_image,
        method="overlay",
        class_name=explanations['class_name']
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()