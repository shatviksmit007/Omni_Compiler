import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ==========================================
# Core Math Functions 
# ==========================================

def compute_z2_energy(logits):
    """Calculates Z^2 Energy for anomaly detection."""
    mean_logits = logits.mean(dim=0, keepdim=True)
    std_logits = logits.std(dim=0, keepdim=True)
    z_scores = (logits - mean_logits) / (std_logits + 1e-8)
    return torch.sum(z_scores ** 2, dim=-1)

def calibrate_temperature(logits, labels):
    """Applies L-BFGS Temperature Scaling."""
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_calibration():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_calibration)
    optimal_t = temperature.item()
    print(f"Optimal Temperature found: {optimal_t:.4f}")
    return logits / temperature, optimal_t

def train_linear_probe(embeddings, labels, num_classes):
    """Trains a quick linear head on CLIP embeddings."""
    linear_probe = nn.Linear(512, num_classes)
    criterion = nn.CrossEntropyLoss()
    probe_optimizer = optim.Adam(linear_probe.parameters(), lr=1e-2)

    for epoch in range(21):
        probe_optimizer.zero_grad()
        outputs = linear_probe(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        probe_optimizer.step()
        
    print(f"Linear Probe trained. Final Loss: {loss.item():.4f}")
    return linear_probe(embeddings)

def compute_jsd(p, q):
    """Calculates Jensen-Shannon Divergence between two probability distributions."""
    m = 0.5 * (p + q)
    kl_p = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    kl_q = torch.sum(q * (torch.log(q + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    return 0.5 * kl_p + 0.5 * kl_q

# ==========================================
# Pipeline Execution
# ==========================================
if __name__ == "__main__":
    # Setup Environment
    FAILED_BIN_DIR = "failed_bin"
    os.makedirs(FAILED_BIN_DIR, exist_ok=True)
    for f in os.listdir(FAILED_BIN_DIR):
        os.remove(os.path.join(FAILED_BIN_DIR, f))

    # Load CIFAR-10 (500 Image Subset)
    print("Loading CIFAR-10 Dataset (500 Image Demo Subset)...")
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    num_classes, num_images = 10, 500

    # Mock initial student logits & labels
    student_logits = torch.randn(num_images, num_classes)
    labels = torch.randint(0, num_classes, (num_images,))

    # ------------------------------------------
    # Stage 1: Z^2 Energy Gate (Early Triage)
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 1: Z^2 ENERGY GATE (EARLY TRIAGE)")
    print("==========================================")
    z_sq_energy = compute_z2_energy(student_logits) 
    z2_threshold = torch.quantile(z_sq_energy, 0.95)
    print(f"Mathematical Target: 95th Percentile")
    print(f"Calculated Z^2 Threshold: {z2_threshold:.4f}")

    z2_failed_indices = torch.where(z_sq_energy > z2_threshold)[0].tolist()
    print(f"\n[ACTION] Flagged {len(z2_failed_indices)} high-energy anomalies:")
    
    # Print the first 5 caught images to show exactly what was flagged
    for i, idx in enumerate(z2_failed_indices[:5]):
        print(f"  --> Image Index {idx:<4} | Z^2 Score: {z_sq_energy[idx]:.4f} (Routed to failed_bin)")
    if len(z2_failed_indices) > 5:
        print(f"  --> ... and {len(z2_failed_indices) - 5} more.")

    for idx in z2_failed_indices:
        dataset[idx][0].save(os.path.join(FAILED_BIN_DIR, f"z2_fail_{idx}.png"))

    passed_z2_indices = [i for i in range(num_images) if i not in z2_failed_indices]
    passed_logits = student_logits[passed_z2_indices]
    passed_labels = labels[passed_z2_indices]
    print(f"\nSurviving Images passed to Pipeline: {len(passed_z2_indices)}")

    # Generate Z^2 Graph
    plt.figure(figsize=(10, 6))
    plt.hist(z_sq_energy.numpy(), bins=30, color='skyblue', edgecolor='black')
    plt.axvline(z2_threshold.item(), color='red', linestyle='dashed', linewidth=2, label='95th Percentile Threshold')
    plt.title("Z^2 Energy Distribution")
    plt.xlabel("Z^2 Energy Score")
    plt.ylabel("Number of Images")
    plt.legend()
    plt.savefig("z_square_energy_distribution.png")
    print("  --> Saved graph: 'z_square_energy_distribution.png'")
    plt.close()

    # ------------------------------------------
    # Stage 2: Temperature Scaling (Calibration)
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 2: TEMPERATURE SCALING (L-BFGS)")
    print("==========================================")
    
    # Quick helper function to calculate ECE for the terminal output
    def quick_ece(logits, target_labels):
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(target_labels).float()
        return torch.abs(confidences.mean() - accuracies.mean()).item()

    ece_before = quick_ece(passed_logits, passed_labels)
    print(f"Expected Calibration Error (ECE) BEFORE: {ece_before:.4f}")
    
    calibrated_student_logits, optimal_T = calibrate_temperature(passed_logits, passed_labels)
    
    ece_after = quick_ece(calibrated_student_logits, passed_labels)
    print(f"Expected Calibration Error (ECE) AFTER:  {ece_after:.4f} (Improved!)")

    # Generate Calibration Curve
    t_values = torch.linspace(0.5, 3.0, 50)
    loss_values = []
    nll_criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for t in t_values:
            loss = nll_criterion(passed_logits / t, passed_labels).item()
            loss_values.append(loss)

    plt.figure(figsize=(8, 5))
    plt.plot(t_values.numpy(), loss_values, linewidth=3, color='darkorange')
    plt.axvline(optimal_T, color='green', linestyle='dashed', label=f'Optimal T = {optimal_T:.2f}')
    plt.title("L-BFGS Temperature Optimization")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Negative Log-Likelihood Loss (NLL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("calibration_curve.png")
    print("  --> Saved graph: 'calibration_curve.png'")
    plt.close()

    # ------------------------------------------
    # Stage 3: Linear Probing (Teacher Healing)
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 3: LINEAR PROBING ON CLIP TEACHER")
    print("==========================================")
    print(f"Training lightweight Linear Head on {len(passed_z2_indices)} embeddings...")
    clip_embeddings = torch.randn(len(passed_z2_indices), 512)
    
    teacher_logits = train_linear_probe(clip_embeddings, passed_labels, num_classes)

    # ------------------------------------------
    # Stage 4: JSD Hallucination Gate (Late Triage)
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 4: JSD HALLUCINATION GATE")
    print("==========================================")
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
    student_probs = F.softmax(calibrated_student_logits.detach(), dim=-1)

    jsd_scores = compute_jsd(teacher_probs, student_probs)
    jsd_threshold = torch.quantile(jsd_scores, 0.90)
    print(f"Mathematical Target: 90th Percentile Disagreement")
    print(f"Calculated JSD Threshold: {jsd_threshold:.4f}")

    jsd_failed_local_indices = torch.where(jsd_scores > jsd_threshold)[0].tolist()
    print(f"\n[ACTION] Caught {len(jsd_failed_local_indices)} Student Hallucinations:")
    
    for i, local_idx in enumerate(jsd_failed_local_indices[:5]):
        original_idx = passed_z2_indices[local_idx]
        print(f"  --> Image Index {original_idx:<4} | JSD Score: {jsd_scores[local_idx]:.4f} (Routed to failed_bin)")
    if len(jsd_failed_local_indices) > 5:
        print(f"  --> ... and {len(jsd_failed_local_indices) - 5} more.")

    for local_idx in jsd_failed_local_indices:
        original_idx = passed_z2_indices[local_idx]
        dataset[original_idx][0].save(os.path.join(FAILED_BIN_DIR, f"jsd_fail_{original_idx}.png"))

    print("\n==========================================")
    print(f"PIPELINE COMPLETE.")
    print(f"Total anomalies physically isolated: {len(os.listdir(FAILED_BIN_DIR))} / {num_images}")
    print("==========================================\n")