import torch
import torch.nn.functional as F

# 1. Set seed for absolute reproducibility during your demo
torch.manual_seed(42)

def compute_jsd(p, q):
    m = 0.5 * (p + q)
    kl_p = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    kl_q = torch.sum(q * (torch.log(q + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    return 0.5 * kl_p + 0.5 * kl_q

# 2. Setup 500 mock images across 3 shoe classes
num_classes = 3
num_images = 500

teacher_logits = torch.randn(num_images, num_classes)
student_logits = torch.randn(num_images, num_classes)

# Inject one absolute failure at index 42 to guarantee a max JSD spike
teacher_logits[42] = torch.tensor([10.0, -5.0, -5.0])
student_logits[42] = torch.tensor([-5.0, 10.0, -5.0])

# 3. Convert to probability distributions
teacher_probs = F.softmax(teacher_logits, dim=-1)
student_probs = F.softmax(student_logits, dim=-1)

# 4. Calculate JSD for all 500 images
jsd_scores = compute_jsd(teacher_probs, student_probs)

# 5. Dynamically calculate the 90th percentile threshold
# torch.quantile requires float tensors, which jsd_scores naturally is
percentile_90_threshold = torch.quantile(jsd_scores, 0.90)

print(f"Dataset Size: {num_images} Images")
print(f"90th Percentile JSD Threshold: {percentile_90_threshold:.4f}\n")
print(f"{'Image Index':<15} | {'JSD Score':<15} | {'Routing Action'}")
print("-" * 58)

# 6. Filter and route only the top 10% worst discrepancies
flagged_indices = torch.where(jsd_scores > percentile_90_threshold)[0]

for idx in flagged_indices:
    score = jsd_scores[idx].item()
    print(f"Image {idx.item():<9} | {score:<15.4f} | --> FAILED_BIN")

print("-" * 58)
print(f"Total anomalies caught: {len(flagged_indices)} out of {num_images}")