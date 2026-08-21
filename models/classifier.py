import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_cifar10_resnet18(num_classes=10):
    """
    Modifies the standard torchvision ResNet18 to work with 32x32 CIFAR-10 images.
    """
    # Initialize a standard ResNet18 without pretrained weights
    model = resnet18(weights=None)
    
    # Replace the initial 7x7 conv with a 3x3 conv suitable for 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the maxpool layer (replace with Identity) to prevent downsampling too early
    model.maxpool = nn.Identity()
    
    # Adjust the final fully connected layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
