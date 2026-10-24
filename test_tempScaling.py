import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# Lock randomness for debugging
np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading MobileNetV3-Small on {device}...")
# Student Model (Frozen)
student_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
student_model.to(device)
student_model.eval()

print("Loading CIFAR-10 Dataset (Hackathon Subset)...")
# Using the local data folder you just set up
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=torchvision.transforms.ToTensor())
subset_indices = np.random.choice(len(dataset), 1000, replace=False)

# 1. Extract Raw Logits
print("Extracting Student Logits...")
all_logits = []
all_labels = []

with torch.no_grad():
    for idx in subset_indices:
        img, label = dataset[idx]
        img = img.unsqueeze(0).to(device)
        logits = student_model(img)
        all_logits.append(logits)
        all_labels.append(label)

logits_tensor = torch.cat(all_logits, dim=0)
labels_tensor = torch.tensor(all_labels).to(device)

# Note: MobileNet outputs 1000 classes, CIFAR has 10 labels. 
# For hackathon plumbing, PyTorch CrossEntropy handles this perfectly (labels 0-9).

# 2. Define the ECE Function
def calculate_ece(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

ece_before = calculate_ece(logits_tensor, labels_tensor)
print(f"\nECE Before Calibration: {ece_before:.4f}")

# 3. L-BFGS Temperature Scaling
print("Running L-BFGS Optimization...")
temperature = nn.Parameter(torch.ones(1, device=device) * 1.5) # Initial guess
nll_criterion = nn.CrossEntropyLoss()
optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

def eval_loss():
    optimizer.zero_grad()
    loss = nll_criterion(logits_tensor / temperature, labels_tensor)
    loss.backward()
    return loss

optimizer.step(eval_loss)
optimal_T = temperature.item()

ece_after = calculate_ece(logits_tensor / optimal_T, labels_tensor)
print(f"Optimal Temperature (T): {optimal_T:.4f}")
print(f"ECE After Calibration:  {ece_after:.4f}")

# 4. Generate the 2D Pitch Deck Graph
print("\nGenerating 2D Calibration Curve...")
t_values = np.linspace(0.5, 3.0, 50)
loss_values = []

with torch.no_grad():
    for t in t_values:
        t_tensor = torch.tensor([t], device=device)
        loss = nll_criterion(logits_tensor / t_tensor, labels_tensor).item()
        loss_values.append(loss)

plt.figure(figsize=(8, 5))
plt.plot(t_values, loss_values, linewidth=3, color='darkorange')
plt.axvline(optimal_T, color='green', linestyle='dashed', label=f'Optimal T = {optimal_T:.2f}')
plt.title("L-BFGS Temperature Optimization")
plt.xlabel("Temperature (T)")
plt.ylabel("Negative Log-Likelihood Loss (NLL)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("calibration_curve.png")
print("Saved 2D Graph to 'calibration_curve.png'")