# import torch
# import torch.nn as nn
# import torchvision.models as models
# import numpy as np
# from captum.attr import (
#     IntegratedGradients,
#     GuidedGradCam,
#     LayerActivation,
#     LayerAttribution
# )
# import matplotlib.pyplot as plt

# # 1. First, let's create a simple model
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Using a pretrained ResNet18 as example
#         self.model = models.resnet18(pretrained=True)
        
#     def forward(self, x):
#         return self.model(x)

# # 2. Create helper functions for visualization
# def visualize_attribution(attribution, original_image, method="heat_map"):
#     """
#     Visualize attribution maps
#     """
#     attribution = attribution.sum(dim=2)  # Sum across color channels
    
#     if method == "heat_map":
#         plt.imshow(attribution.squeeze().cpu().detach().numpy(), cmap='hot')
#         plt.colorbar()
#     elif method == "overlay":
#         attribution = attribution.squeeze().cpu().detach().numpy()
#         # Normalize attribution
#         attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
#         plt.imshow(original_image.squeeze().cpu().detach().numpy().transpose(1,2,0))
#         plt.imshow(attribution, cmap='hot', alpha=0.5)

# # 3. Create XAI wrapper
# class ModelExplainer:
#     def __init__(self, model):
#         self.model = model
#         self.model.eval()
        
#         # Initialize attribution methods
#         self.integrated_gradients = IntegratedGradients(self.model)
#         self.guided_gradcam = GuidedGradCam(self.model, self.model.model.layer4)
        
#     def explain_prediction(self, input_tensor, target_class=None):
#         """
#         Generate multiple types of explanations
#         """
#         # If no target class specified, use model's prediction
#         if target_class is None:
#             with torch.no_grad():
#                 output = self.model(input_tensor)
#                 target_class = output.argmax(dim=1).item()
        
#         # 1. Integrated Gradients
#         ig_attr = self.integrated_gradients.attribute(
#             input_tensor,
#             target=target_class,
#             n_steps=50
#         )
        
#         # 2. Guided GradCAM
#         gc_attr = self.guided_gradcam.attribute(
#             input_tensor,
#             target=target_class
#         )
        
#         return {
#             'integrated_gradients': ig_attr,
#             'guided_gradcam': gc_attr,
#             'target_class': target_class
#         }

# # 4. Example usage
# def main():
#     # Create model and explainer
#     model = SimpleCNN()
#     explainer = ModelExplainer(model)
    
#     # Create dummy input (normally would be your actual image)
#     dummy_input = torch.randn(1, 3, 224, 224)
    
#     # Get explanations
#     explanations = explainer.explain_prediction(dummy_input)
    
#     # Visualize different attribution methods
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.title('Integrated Gradients')
#     visualize_attribution(explanations['integrated_gradients'], dummy_input)
    
#     plt.subplot(1, 2, 2)
#     plt.title('Guided GradCAM')
#     visualize_attribution(explanations['guided_gradcam'], dummy_input)
    
#     plt.show()

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, GuidedGradCam

# 1. Define the Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        
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
    return transform(image).unsqueeze(0), image  # Add batch dimension

# 3. Visualization Function
def visualize_attribution(attribution, original_image, method="heat_map"):
    attribution = attribution.sum(dim=1).squeeze().cpu().detach().numpy()  # Sum across channels
    
    if method == "heat_map":
        plt.imshow(attribution, cmap='hot')
        plt.colorbar()
    elif method == "overlay":
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())  # Normalize
        plt.imshow(original_image)
        plt.imshow(attribution, cmap='hot', alpha=0.5)

# 4. Explainability Wrapper
class ModelExplainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        self.integrated_gradients = IntegratedGradients(self.model)
        self.guided_gradcam = GuidedGradCam(self.model, self.model.model.layer4)
        
    def explain_prediction(self, input_tensor, target_class=None):
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        ig_attr = self.integrated_gradients.attribute(input_tensor, target=target_class, n_steps=50)
        gc_attr = self.guided_gradcam.attribute(input_tensor, target=target_class)
        
        return {'integrated_gradients': ig_attr, 'guided_gradcam': gc_attr, 'target_class': target_class}

# 5. Run the Model on a Real Image
def main():
    model = SimpleCNN()
    explainer = ModelExplainer(model)
    
    # Load and preprocess an image
    image_path = "download.jpeg"  # <-- Replace with your image file path
    input_tensor, original_image = preprocess_image(image_path)
    
    # Generate explanations
    explanations = explainer.explain_prediction(input_tensor)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Integrated Gradients')
    visualize_attribution(explanations['integrated_gradients'], original_image)
    
    plt.subplot(1, 2, 2)
    plt.title('Guided GradCAM')
    visualize_attribution(explanations['guided_gradcam'], original_image)
    
    plt.show()

if __name__ == "__main__":
    main()
