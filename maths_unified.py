"""
OmniCompiler - Mathematical Triage & Evaluation Pipeline.

Execution Order:
  1. Z^2 ENERGY GATE (Per-class CLIP embedding Z^2 energy -> removes outliers to failed_bin first)
  2. STUDENT INFERENCE (Cross-fits the user's PyTorch model on surviving data -> Per-Class & Overall Accuracy)
  3. TEMPERATURE SCALING (L-BFGS optimization + Binned ECE before & after)
  4. CLIP LINEAR PROBE TEACHER (Cross-fitted linear probe on CLIP embeddings -> Teacher Accuracy)
  5. JSD HALLUCINATION GATE (Jensen-Shannon Divergence between Teacher & Calibrated Student -> routes to failed_bin / safe_bin)
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import ast
import inspect
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from transformers import CLIPModel, CLIPProcessor

SEED = 42
FAILED_BIN_DIR = "failed_bin"
SAFE_BIN_DIR = "safe_bin"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

STUDENT_EPOCHS = 6
STUDENT_LR = 5e-4
STUDENT_BATCH = 16
Z2_PERCENTILE = 0.90
JSD_PERCENTILE = 0.90

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ==========================================
# Core Mathematical Functions
# ==========================================

def compute_per_class_z2_energy(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Computes per-class feature-wise Z^2 energy across embeddings."""
    z2_scores = torch.zeros(len(embeddings), dtype=torch.float32)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 1:
            cls_feats = embeddings[mask]
            mu = cls_feats.mean(dim=0, keepdim=True)
            sigma = cls_feats.std(dim=0, keepdim=True) + 1e-6
            z2_scores[mask] = torch.sum(((cls_feats - mu) / sigma) ** 2, dim=-1)
        elif mask.sum() == 1:
            z2_scores[mask] = 0.0
    return z2_scores


def compute_jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between Teacher (p) and Calibrated Student (q) probabilities."""
    m = 0.5 * (p + q)
    kl_p = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    kl_q = torch.sum(q * (torch.log(q + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    return 0.5 * kl_p + 0.5 * kl_q


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Binned Expected Calibration Error (ECE)."""
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    ece = 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.float().mean().item() * abs(conf[in_bin].mean().item() - acc[in_bin].mean().item())
    return float(ece)


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor, t_min: float = 0.05, t_max: float = 10.0):
    """L-BFGS temperature scaling on log(T)."""
    log_t = nn.Parameter(torch.zeros(1))
    opt = optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / log_t.exp().clamp(t_min, t_max), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().clamp(t_min, t_max).item())
    hit_bound = t <= t_min * 1.001 or t >= t_max * 0.999
    return logits / t, t, hit_bound


def train_linear_probe(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int, epochs: int = 200, lr: float = 1e-2):
    """Trains a Linear Probe head on L2-normalized CLIP image embeddings."""
    probe = nn.Linear(embeddings.shape[1], num_classes)
    opt = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(probe(embeddings), labels)
        loss.backward()
        opt.step()
    return probe


def compute_accuracy_breakdown(preds: torch.Tensor, labels: torch.Tensor, classes: List[str]) -> Tuple[float, Dict[str, float]]:
    """Returns (overall_accuracy, per_class_accuracy_dict)."""
    overall_acc = float((preds == labels).float().mean().item()) if len(labels) > 0 else 0.0
    per_class = {}
    for idx, cls_name in enumerate(classes):
        mask = (labels == idx)
        if mask.any():
            per_class[cls_name] = float((preds[mask] == labels[mask]).float().mean().item())
        else:
            per_class[cls_name] = 0.0
    return overall_acc, per_class


# ==========================================
# Student Model Loading & Cross-Fitted Inference
# ==========================================

class CuratedDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


def _is_literal(node):
    try:
        ast.literal_eval(node)
        return True
    except Exception:
        return False


def load_student_factory(model_path):
    source = Path(model_path).read_text(encoding="utf-8-sig")
    keep = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and _is_literal(node.value):
            keep.append(node)
    module = ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[]))

    ns = {"__name__": "omni_student_model", "__file__": str(model_path)}
    exec(compile(module, str(model_path), "exec"), ns)

    classes = [
        v for v in ns.values()
        if inspect.isclass(v) and issubclass(v, nn.Module) and v.__module__ == "omni_student_model"
    ]
    if not classes:
        raise ValueError("No torch.nn.Module subclass found in the model file.")
    with_step = [c for c in classes if callable(getattr(c, "train_step", None))]
    cls = (with_step or classes)[-1]

    params = [p for n, p in inspect.signature(cls.__init__).parameters.items() if n != "self"]
    takes_nc = any(p.name == "num_classes" for p in params)

    def factory(num_classes):
        return cls(num_classes=num_classes) if takes_nc else cls()

    factory.class_name = cls.__name__
    return factory


def check_student(factory, num_classes):
    model = factory(num_classes).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    if out.ndim != 2 or out.shape[1] != num_classes:
        raise ValueError(
            f"{factory.class_name} outputs shape {tuple(out.shape)} but you entered {num_classes} class labels."
        )


def train_student(model, dataset, epochs, device, lr=STUDENT_LR):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=STUDENT_BATCH, shuffle=True, drop_last=len(dataset) > STUDENT_BATCH
    )
    model.to(device).train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    step = getattr(model, "train_step", None)
    use_step = callable(step) and len(inspect.signature(step).parameters) == 4
    last = float("nan")
    for _ in range(epochs):
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if use_step:
                loss = model.train_step(opt, loss_fn, x, y)
            else:
                opt.zero_grad()
                l = loss_fn(model(x), y)
                l.backward()
                opt.step()
                loss = l.item()
            total += loss * len(y)
            seen += len(y)
        last = total / max(seen, 1)
    return last


def predict_logits(model, dataset, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x.to(device)).cpu())
    return torch.cat(out)


def make_folds(labels_np: np.ndarray, k: int, num_classes: int):
    counts = np.bincount(labels_np, minlength=num_classes)
    min_c = int(counts.min())
    if min_c < 2:
        indices = np.arange(len(labels_np))
        mid = max(1, len(indices) // 2)
        return [(indices[:mid], indices[mid:]), (indices[mid:], indices[:mid])], 2
    k = max(2, min(k, min_c))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(labels_np)), labels_np)), k


def student_oof_logits(factory, samples, num_classes, folds, epochs, device):
    oof = torch.zeros(len(samples), num_classes)
    for f, (tr, te) in enumerate(folds, 1):
        model = factory(num_classes)
        loss = train_student(model, CuratedDataset([samples[i] for i in tr], TRAIN_TF), epochs, device)
        te_t = torch.as_tensor(te, dtype=torch.long)
        oof[te_t] = predict_logits(model, CuratedDataset([samples[i] for i in te], EVAL_TF), device)
        print(f"  Fold {f}/{len(folds)}: trained on {len(tr)} imgs (loss {loss:.3f}) -> evaluated on {len(te)} held-out imgs")
    return oof


# ==========================================
# CLIP Embeddings & Linear Probe Teacher
# ==========================================

def _image_features(clip_model, pixel_values):
    out = clip_model.get_image_features(pixel_values=pixel_values)
    if torch.is_tensor(out):
        return out
    emb = getattr(out, "image_embeds", None)
    if emb is not None:
        return emb
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        if pooled.shape[-1] != clip_model.config.projection_dim:
            pooled = clip_model.visual_projection(pooled)
        return pooled
    raise RuntimeError("Could not extract CLIP image features.")


def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval(), CLIPProcessor.from_pretrained(CLIP_MODEL_ID), device


def extract_clip_embeddings(paths, clip, batch=32):
    model, processor, device = clip
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            f = _image_features(model, inputs["pixel_values"])
            feats.append(F.normalize(f.float(), p=2, dim=-1).cpu())
    return torch.cat(feats)


def teacher_oof_logits(embeddings, labels_t, folds, num_classes):
    oof = torch.zeros(len(labels_t), num_classes)
    for tr, te in folds:
        tr_t = torch.as_tensor(tr, dtype=torch.long)
        te_t = torch.as_tensor(te, dtype=torch.long)
        probe = train_linear_probe(embeddings[tr_t], labels_t[tr_t], num_classes)
        with torch.no_grad():
            oof[te_t] = probe(embeddings[te_t])
    return oof


def _clear_bin(directory):
    os.makedirs(directory, exist_ok=True)
    for f in os.listdir(directory):
        p = os.path.join(directory, f)
        if os.path.isfile(p):
            os.remove(p)


# ==========================================
# Main Mathematical Triage & Evaluation Pipeline
# ==========================================

def run_math_pipeline(
    curation: dict,
    model_path: str,
    folds: int = 5,
    epochs: int = STUDENT_EPOCHS,
    clip=None,
    update_bins: bool = True,
    stage_label: str = "INITIAL",
):
    """
    Runs the full math triage and evaluation sequence:
      1. Z^2 Energy Outlier Detection & Removal (routes outliers to failed_bin)
      2. Student Inference on surviving clean dataset (Per-Class Accuracy & Overall Accuracy)
      3. Temperature Scaling (L-BFGS + ECE before/after)
      4. Linear Probe on CLIP Teacher (Teacher Per-Class & Overall Accuracy)
      5. JSD Hallucination Gate (routes JSD failures to failed_bin and survivors to safe_bin)
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if update_bins:
        _clear_bin(FAILED_BIN_DIR)
        _clear_bin(SAFE_BIN_DIR)

    classes = curation["classes"]
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError("Mathematical triage requires at least 2 classes.")

    raw_samples = curation["samples"]
    paths = [s["path"] for s in raw_samples]
    labels_list = [
        s["class_idx"] if "class_idx" in s else classes.index(s["label"])
        for s in raw_samples
    ]
    labels_np = np.array(labels_list, dtype=np.int64)
    labels_all = torch.as_tensor(labels_np, dtype=torch.long)
    n_total = len(paths)

    print(f"\n[{stage_label}] Dataset size: {n_total} images across {num_classes} classes: {classes}")

    clip = clip or load_clip()
    print(f"Extracting CLIP ViT-B/32 embeddings for {n_total} images...")
    all_embeddings = extract_clip_embeddings(paths, clip)

    # ---------------------------------------------------------
    # STEP 1: Z^2 ENERGY GATE (OUTLIER REMOVAL FROM DATASET)
    # ---------------------------------------------------------
    print("\n==========================================")
    print(f"[{stage_label}] STEP 1: Z^2 ENERGY GATE & OUTLIER REMOVAL")
    print("==========================================")
    z2_all = compute_per_class_z2_energy(all_embeddings, labels_all, num_classes)
    z2_threshold = torch.quantile(z2_all, Z2_PERCENTILE)

    z2_candidates = torch.where(z2_all > z2_threshold)[0].tolist()
    class_remaining = Counter(labels_list)
    z2_failed = []
    for idx in sorted(z2_candidates, key=lambda i: float(z2_all[i]), reverse=True):
        c_idx = labels_list[idx]
        if class_remaining[c_idx] > 2:
            z2_failed.append(idx)
            class_remaining[c_idx] -= 1

    z2_failed_set = set(z2_failed)
    surviving_indices = [i for i in range(n_total) if i not in z2_failed_set]

    print(f"Z^2 {Z2_PERCENTILE:.0%} threshold: {z2_threshold:.4f}")
    print(f"[ACTION] Removed {len(z2_failed)} Z^2 energy outliers from dataset | {len(surviving_indices)} clean images surviving.")

    failed_records = []
    if update_bins:
        for idx in z2_failed:
            dst_name = f"z2_fail_{idx}_{os.path.basename(paths[idx])}"
            dst_path = os.path.join(FAILED_BIN_DIR, dst_name)
            shutil.copy(paths[idx], dst_path)
            failed_records.append({
                "index": idx,
                "original_path": paths[idx],
                "bin_file": dst_name,
                "bin_path": dst_path,
                "class": classes[labels_list[idx]],
                "class_idx": labels_list[idx],
                "reason": "z2_energy",
                "score": float(z2_all[idx]),
            })

    plt.figure(figsize=(10, 6))
    plt.hist(z2_all.numpy(), bins=25, color="skyblue", edgecolor="black")
    plt.axvline(z2_threshold.item(), color="red", linestyle="dashed", linewidth=2, label=f"{Z2_PERCENTILE:.0%} Threshold")
    plt.title(f"Z^2 Energy Distribution ({stage_label})")
    plt.xlabel("Z^2 Energy Score")
    plt.ylabel("Number of Images")
    plt.legend()
    plt.savefig("z_square_energy_distribution.png")
    plt.close()

    surv_idx_t = torch.as_tensor(surviving_indices, dtype=torch.long)
    surv_paths = [paths[i] for i in surviving_indices]
    surv_labels_np = labels_np[surviving_indices]
    surv_labels = torch.as_tensor(surv_labels_np, dtype=torch.long)
    surv_embeddings = all_embeddings[surv_idx_t]
    surv_samples = list(zip(surv_paths, surv_labels_np.tolist()))

    fold_idx, k = make_folds(surv_labels_np, folds, num_classes)

    # ---------------------------------------------------------
    # STEP 2: STUDENT INFERENCE, PER-CLASS & OVERALL ACCURACY
    # ---------------------------------------------------------
    print("\n==========================================")
    print(f"[{stage_label}] STEP 2: STUDENT INFERENCE & ACCURACY EVALUATION")
    print("==========================================")
    factory = load_student_factory(model_path)
    print(f"Student Model: {factory.class_name} ({k}-fold cross-fitting, {epochs} epochs/fold)")
    student_logits = student_oof_logits(factory, surv_samples, num_classes, fold_idx, epochs, device)
    student_preds = student_logits.argmax(dim=1)

    student_overall_acc, student_per_class_acc = compute_accuracy_breakdown(student_preds, surv_labels, classes)
    print(f"\n📊 Student Overall Accuracy: {student_overall_acc:.1%}")
    print("📊 Student Per-Class Accuracy:")
    for cls_name, acc_val in student_per_class_acc.items():
        print(f"   • {cls_name:<18}: {acc_val:.1%}")

    # ---------------------------------------------------------
    # STEP 3: TEMPERATURE SCALING (L-BFGS) & ECE
    # ---------------------------------------------------------
    print("\n==========================================")
    print(f"[{stage_label}] STEP 3: TEMPERATURE SCALING (L-BFGS)")
    print("==========================================")
    ece_before = expected_calibration_error(student_logits, surv_labels)
    cal_logits, optimal_t, hit_bound = calibrate_temperature(student_logits, surv_labels)
    ece_after = expected_calibration_error(cal_logits, surv_labels)
    print(f"Optimal Temperature T = {optimal_t:.4f}")
    print(f"ECE Before Calibration: {ece_before:.4f}  -->  ECE After Calibration: {ece_after:.4f}")

    ce = nn.CrossEntropyLoss()
    t_values = torch.linspace(0.25, max(3.0, optimal_t * 1.5), 60)
    with torch.no_grad():
        nll = [ce(student_logits / t, surv_labels).item() for t in t_values]
    plt.figure(figsize=(8, 5))
    plt.plot(t_values.numpy(), nll, linewidth=3, color="darkorange")
    plt.axvline(optimal_t, color="green", linestyle="dashed", label=f"Optimal T = {optimal_t:.2f}")
    plt.title(f"L-BFGS Temperature Scaling ({stage_label})")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Negative Log-Likelihood Loss (NLL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("calibration_curve.png")
    plt.close()

    # ---------------------------------------------------------
    # STEP 4: CLIP TEACHER LINEAR PROBE
    # ---------------------------------------------------------
    print("\n==========================================")
    print(f"[{stage_label}] STEP 4: CLIP TEACHER LINEAR PROBE")
    print("==========================================")
    teacher_logits = teacher_oof_logits(surv_embeddings, surv_labels, fold_idx, num_classes)
    teacher_preds = teacher_logits.argmax(dim=1)
    teacher_overall_acc, teacher_per_class_acc = compute_accuracy_breakdown(teacher_preds, surv_labels, classes)

    print(f"📊 CLIP Linear Probe Teacher Overall Accuracy: {teacher_overall_acc:.1%}")
    for cls_name, acc_val in teacher_per_class_acc.items():
        print(f"   • {cls_name:<18}: {acc_val:.1%}")

    final_probe = train_linear_probe(surv_embeddings, surv_labels, num_classes)
    os.makedirs("omni_dataset", exist_ok=True)
    torch.save({"state_dict": final_probe.state_dict(), "classes": classes}, os.path.join("omni_dataset", "clip_probe.pth"))

    # ---------------------------------------------------------
    # STEP 5: JSD HALLUCINATION GATE & FAILED_BIN ROUTING
    # ---------------------------------------------------------
    print("\n==========================================")
    print(f"[{stage_label}] STEP 5: JSD HALLUCINATION GATE & BIN ROUTING")
    print("==========================================")
    t_probs = F.softmax(teacher_logits, dim=-1)
    s_probs = F.softmax(cal_logits, dim=-1)
    jsd_scores = compute_jsd(t_probs, s_probs)
    mean_jsd = float(jsd_scores.mean().item())
    jsd_threshold = torch.quantile(jsd_scores, JSD_PERCENTILE)

    jsd_failed_local = torch.where(jsd_scores > jsd_threshold)[0].tolist()
    if update_bins and len(z2_failed) + len(jsd_failed_local) < 2 and len(surviving_indices) >= 4:
        top_local = torch.argsort(jsd_scores, descending=True)[:2].tolist()
        for loc_i in top_local:
            if loc_i not in jsd_failed_local:
                jsd_failed_local.append(loc_i)

    jsd_failed_local_set = set(jsd_failed_local)
    jsd_failed_global = [surviving_indices[j] for j in jsd_failed_local]
    jsd_failed_global_set = set(jsd_failed_global)

    safe_local = [j for j in range(len(surviving_indices)) if j not in jsd_failed_local_set]
    safe_global = [surviving_indices[j] for j in safe_local]

    print(f"Mean JSD: {mean_jsd:.4f} | {JSD_PERCENTILE:.0%} Threshold: {jsd_threshold:.4f}")
    print(f"[ACTION] Flagged {len(jsd_failed_global)} JSD hallucinations -> routed to '{FAILED_BIN_DIR}/'")

    safe_records = []
    if update_bins:
        for loc_j in jsd_failed_local:
            g_idx = surviving_indices[loc_j]
            dst_name = f"jsd_fail_{g_idx}_{os.path.basename(paths[g_idx])}"
            dst_path = os.path.join(FAILED_BIN_DIR, dst_name)
            shutil.copy(paths[g_idx], dst_path)
            failed_records.append({
                "index": g_idx,
                "original_path": paths[g_idx],
                "bin_file": dst_name,
                "bin_path": dst_path,
                "class": classes[labels_list[g_idx]],
                "class_idx": labels_list[g_idx],
                "reason": "jsd_hallucination",
                "score": float(jsd_scores[loc_j]),
            })

        for loc_j in safe_local:
            g_idx = surviving_indices[loc_j]
            dst_name = f"safe_{g_idx}_{os.path.basename(paths[g_idx])}"
            dst_path = os.path.join(SAFE_BIN_DIR, dst_name)
            shutil.copy(paths[g_idx], dst_path)
            safe_records.append({
                "index": g_idx,
                "path": paths[g_idx],
                "bin_path": dst_path,
                "label": classes[labels_list[g_idx]],
                "class_idx": labels_list[g_idx],
                "split": raw_samples[g_idx].get("split", "train"),
            })

        print(f"📦 Bin Summary: {len(failed_records)} images in '{FAILED_BIN_DIR}/' | {len(safe_records)} images in '{SAFE_BIN_DIR}/'")

    failed_counts_per_class = Counter(r["class"] for r in failed_records)

    local_by_global = {g_idx: loc_j for loc_j, g_idx in enumerate(surviving_indices)}
    items = []
    for i in range(n_total):
        if i in z2_failed_set:
            items.append({
                "index": i, "path": paths[i], "class": classes[labels_list[i]],
                "split": raw_samples[i].get("split", "train"),
                "student_pred": None, "teacher_pred": None,
                "z2": float(z2_all[i]), "jsd": None, "route": "failed_z2",
            })
        else:
            loc_j = local_by_global[i]
            route = "failed_jsd" if i in jsd_failed_global_set else "safe"
            items.append({
                "index": i, "path": paths[i], "class": classes[labels_list[i]],
                "split": raw_samples[i].get("split", "train"),
                "student_pred": classes[int(student_preds[loc_j])],
                "teacher_pred": classes[int(teacher_preds[loc_j])],
                "z2": float(z2_all[i]), "jsd": float(jsd_scores[loc_j]), "route": route,
            })

    safe_local_t = torch.as_tensor(safe_local, dtype=torch.long) if safe_local else None
    jsd_local_t = torch.as_tensor(jsd_failed_local, dtype=torch.long) if jsd_failed_local else None
    acc_safe = float((student_preds[safe_local_t] == surv_labels[safe_local_t]).float().mean().item()) if safe_local_t is not None else student_overall_acc
    acc_flagged = float((student_preds[jsd_local_t] == surv_labels[jsd_local_t]).float().mean().item()) if jsd_local_t is not None else 0.0

    return {
        "stage_label": stage_label,
        "classes": classes,
        "num_images": n_total,
        "surviving_after_z2": len(surviving_indices),
        "folds": k,
        "epochs": epochs,
        "student_class": factory.class_name,
        "student_accuracy": student_overall_acc,
        "student_per_class_accuracy": student_per_class_acc,
        "teacher_accuracy": teacher_overall_acc,
        "teacher_per_class_accuracy": teacher_per_class_acc,
        "z2_threshold": float(z2_threshold),
        "z2_flagged": len(z2_failed),
        "temperature": optimal_t,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "mean_jsd": mean_jsd,
        "jsd_threshold": float(jsd_threshold),
        "jsd_flagged": len(jsd_failed_global),
        "safe_count": len(safe_global),
        "failed_count": len(z2_failed) + len(jsd_failed_global),
        "student_accuracy_safe": acc_safe,
        "student_accuracy_flagged": acc_flagged,
        "failed_records": failed_records,
        "safe_records": safe_records,
        "failed_counts_per_class": dict(failed_counts_per_class),
        "graphs": {
            "z2": os.path.abspath("z_square_energy_distribution.png"),
            "calibration": os.path.abspath("calibration_curve.png"),
        },
        "items": items,
    }