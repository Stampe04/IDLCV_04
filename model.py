import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(pretrained=True, freeze_backbone=True):
    # Load ResNet-18 with pretrained ImageNet weights
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Freeze pretrained backbone initially
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final classification layer
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2))

    return model

def unfreeze_backbone(model):
    # Unfreeze the backbone layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    return model