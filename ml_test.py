import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class CustomMobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Load the pre-trained ImageNet MobileNetV3-Small model
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Replace the classifier head for your specific classes
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def train_step(self, optimizer, loss_fn, inputs, targets):
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()