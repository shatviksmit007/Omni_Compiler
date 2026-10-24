import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch 
import clip 
import numpy as np
import torchvision
import matplotlib.pyplot as plt 
from collections import defaultdict

np.random.seed(42)
torch.manual_seed(42)

os.makedirs("failed_bin", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP ViT-B/32 model on {device}...")
model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading CIFAR-10 dataset...")
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False)

num_samples = 500
subset_indices = np.random.choice(len(dataset), num_samples, replace=False)

features_by_class = defaultdict(list)
images_by_class = defaultdict(list)

print("Extracting CLIP Embeddings...")
with torch.no_grad():
    for idx in subset_indices:
        img, label = dataset[idx]
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        feature = model.encode_image(img_tensor).cpu().numpy().flatten()

        features_by_class[label].append(feature)
        images_by_class[label].append((idx, img))


print("Calculating Z square energy...")
z_scores_all = []
image_tracking_data = []


for label, features in features_by_class.items():
    features_np = np.array(features)

    mu = np.mean(features_np, axis=0)
    sigma = np.std(features_np, axis=0)

    for i, feature in enumerate(features_np):
        z_sq = np.sum(((feature - mu) / sigma) ** 2)
        z_scores_all.append(z_sq)

        idx, original_img = images_by_class[label][i]
        image_tracking_data.append((z_sq, idx, label, original_img))


threshold_95 = np.percentile(z_scores_all, 95)
print(f"95th percentile threshold for Z square energy: {threshold_95}")

plt.figure(figsize=(10, 6))
plt.hist(z_scores_all, bins=30, color='skyblue', edgecolor='black')
plt.axvline(threshold_95, color='red', linestyle='dashed', linewidth=2, label='95th Percentile Threshold')
plt.title("Z square energy distribution")
plt.xlabel("Z square energy")
plt.ylabel("Number of Images")
plt.legend()
plt.savefig("z_square_energy_distribution.png")
print("Histogram saved as 'z_square_energy_distribution.png'.")

print("Routing anomalies to failed_bin/")
outlier_count = 0
for z_sq, idx, label, original_img in image_tracking_data:
    if z_sq > threshold_95:
        outlier_count += 1
        img.save(f"failed_bin/class{label}_z{z_sq:.1f}_idx{idx}.png")


print(f"Stage 1 completed.")