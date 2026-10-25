"""
OmniCompiler - mathematical triage (Stages 3-4 of the pipeline). Everything runs on the real curated images.

  0. STUDENT   : your model file (the class from the CLI's "Model Python file") is trained with K-fold
                 cross-fitting, so every image is scored by a model that never saw it.
  1. Z^2 GATE  : feature-wise z-score energy on the student's held-out logits -> top 5% to failed_bin.
  2. CALIBRATE : L-BFGS temperature scaling + binned ECE on the surviving images.
  3. TEACHER   : real CLIP ViT-B/32 embeddings + a linear probe trained on them (also cross-fitted).
  4. JSD GATE  : Jensen-Shannon divergence between teacher and calibrated student -> top 10% to failed_bin.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")  # Windows: torch + sklearn both ship an OpenMP runtime

import ast
import inspect
import shutil
from pathlib import Path

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
Z2_PERCENTILE = 0.95
JSD_PERCENTILE = 0.90

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),   # the old code skipped this; ImageNet-pretrained nets need it
])
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ==========================================
# Core math
# ==========================================

def compute_z2_energy(logits):
    """Z^2 energy: per-logit z-score across the dataset (dim=0), squared and summed per image."""
    z = (logits - logits.mean(dim=0, keepdim=True)) / (logits.std(dim=0, keepdim=True) + 1e-8)
    return torch.sum(z ** 2, dim=-1)


def compute_jsd(p, q):
    """Jensen-Shannon divergence between two batches of probability vectors."""
    m = 0.5 * (p + q)
    kl_p = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    kl_q = torch.sum(q * (torch.log(q + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    return 0.5 * kl_p + 0.5 * kl_q


def expected_calibration_error(logits, labels, n_bins=15):
    """Real binned ECE (the old 'quick_ece' was |mean confidence - mean accuracy|, which can hide miscalibration)."""
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


def calibrate_temperature(logits, labels, t_min=0.05, t_max=10.0):
    """L-BFGS temperature scaling on log(T) (keeps T > 0). The old lr=0.01 barely moved T off its 1.5 init."""
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


def train_linear_probe(embeddings, labels, num_classes, epochs=200, lr=1e-2):
    """Full-batch linear head on (L2-normalised) CLIP embeddings. Returns the trained probe."""
    probe = nn.Linear(embeddings.shape[1], num_classes)
    opt = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(probe(embeddings), labels)
        loss.backward()
        opt.step()
    return probe


# ==========================================
# Student model: load the user's file, train it, score held-out images
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
    """Builds `factory(num_classes) -> nn.Module` from the user's model file.

    Only imports, classes, functions and literal constants are executed - top-level script code
    (training loops, dataset downloads, plotting) in the file is skipped, so importing it is side-effect free.
    """
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

    classes = [v for v in ns.values()
               if inspect.isclass(v) and issubclass(v, nn.Module) and v.__module__ == "omni_student_model"]
    if not classes:
        raise ValueError(
            "No torch.nn.Module subclass found in the model file. The pipeline trains your model on the curated "
            "images, so the file must define a PyTorch model class (scikit-learn models are not supported here).")
    with_step = [c for c in classes if callable(getattr(c, "train_step", None))]
    cls = (with_step or classes)[-1]

    params = [p for n, p in inspect.signature(cls.__init__).parameters.items() if n != "self"]
    takes_nc = any(p.name == "num_classes" for p in params)
    required = [p.name for p in params
                if p.name != "num_classes" and p.default is p.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if required:
        raise ValueError(f"{cls.__name__}.__init__ needs arguments {required} that OmniCompiler cannot supply. "
                         f"Give them defaults (only `num_classes` is passed in).")

    def factory(num_classes):
        return cls(num_classes=num_classes) if takes_nc else cls()

    factory.class_name = cls.__name__
    return factory


def check_student(factory, num_classes):
    """Preflight: the model must accept 224x224 RGB and output one logit per class."""
    model = factory(num_classes).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))  # shape check only; no results come from this tensor
    if out.ndim != 2 or out.shape[1] != num_classes:
        raise ValueError(f"{factory.class_name} outputs shape {tuple(out.shape)} but you entered {num_classes} "
                         f"class labels. Make its final layer output num_classes values.")


def train_student(model, dataset, epochs, device, lr=STUDENT_LR):
    loader = torch.utils.data.DataLoader(dataset, batch_size=STUDENT_BATCH, shuffle=True,
                                         drop_last=len(dataset) > STUDENT_BATCH)
    model.to(device).train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    step = getattr(model, "train_step", None)
    use_step = callable(step) and len(inspect.signature(step).parameters) == 4  # (optimizer, loss_fn, x, y)
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


def make_folds(labels, k):
    counts = np.bincount(labels)
    if counts.min() < 2:
        raise ValueError(f"Every class needs at least 2 curated images (smallest class has {counts.min()}).")
    k = max(2, min(k, int(counts.min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(labels)), labels)), k


def student_oof_logits(factory, samples, num_classes, folds, epochs, device):
    oof = torch.zeros(len(samples), num_classes)
    for f, (tr, te) in enumerate(folds, 1):
        model = factory(num_classes)
        loss = train_student(model, CuratedDataset([samples[i] for i in tr], TRAIN_TF), epochs, device)
        te_t = torch.as_tensor(te)
        oof[te_t] = predict_logits(model, CuratedDataset([samples[i] for i in te], EVAL_TF), device)
        print(f"  fold {f}/{len(folds)}: trained on {len(tr)} imgs (final loss {loss:.3f}), scored {len(te)} held-out")
    return oof


# ==========================================
# Teacher: real CLIP embeddings + linear probe
# ==========================================

def _image_features(clip_model, pixel_values):
    out = clip_model.get_image_features(pixel_values=pixel_values)
    if torch.is_tensor(out):
        return out
    emb = getattr(out, "image_embeds", None)          # newer transformers may return an output object
    if emb is not None:
        return emb
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        if pooled.shape[-1] != clip_model.config.projection_dim:
            pooled = clip_model.visual_projection(pooled)
        return pooled
    raise RuntimeError("Could not read image features from this transformers version's CLIP output.")


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
        tr_t, te_t = torch.as_tensor(tr), torch.as_tensor(te)
        probe = train_linear_probe(embeddings[tr_t], labels_t[tr_t], num_classes)
        with torch.no_grad():
            oof[te_t] = probe(embeddings[te_t])
    return oof


# ==========================================
# Pipeline
# ==========================================

def _clear_bin(directory):
    os.makedirs(directory, exist_ok=True)
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))


def run_math_pipeline(curation, model_path, folds=5, epochs=STUDENT_EPOCHS, clip=None):
    """curation = result of tinyFish.run_curation(); model_path = the user's model .py file. Returns a report dict."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _clear_bin(FAILED_BIN_DIR)
    _clear_bin(SAFE_BIN_DIR)

    classes = curation["classes"]
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError("Triage needs at least 2 classes.")
    paths = [s["path"] for s in curation["samples"]]
    labels_np = np.array([s["class_idx"] for s in curation["samples"]])
    labels = torch.as_tensor(labels_np)
    samples = list(zip(paths, labels_np.tolist()))
    n = len(samples)
    print(f"\nCurated dataset: {n} real images across {num_classes} classes: {classes}")

    fold_idx, k = make_folds(labels_np, folds)

    # ------------------------------------------
    # Stage 0: student (the user's model), trained for real
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 0: STUDENT MODEL (CROSS-FITTED ON REAL DATA)")
    print("==========================================")
    factory = load_student_factory(model_path)
    print(f"Student: {factory.class_name} from {Path(model_path).name} | device={device} | "
          f"{k}-fold, {epochs} epochs/fold")
    student_logits = student_oof_logits(factory, samples, num_classes, fold_idx, epochs, device)
    student_pred = student_logits.argmax(1)
    student_acc = (student_pred == labels).float().mean().item()
    print(f"Student held-out accuracy: {student_acc:.1%}")

    # ------------------------------------------
    # Stage 1: Z^2 energy gate
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 1: Z^2 ENERGY GATE (EARLY TRIAGE)")
    print("==========================================")
    z2 = compute_z2_energy(student_logits)
    z2_threshold = torch.quantile(z2, Z2_PERCENTILE)
    print(f"Target: {Z2_PERCENTILE:.0%} percentile | Z^2 threshold: {z2_threshold:.4f}")
    z2_failed = torch.where(z2 > z2_threshold)[0].tolist()
    z2_failed_set = set(z2_failed)
    print(f"[ACTION] Flagged {len(z2_failed)} high-energy anomalies:")
    for idx in z2_failed[:5]:
        print(f"  --> Image {idx:<4} | class {classes[labels_np[idx]]:<14} | Z^2 {z2[idx]:.4f} (-> failed_bin)")
    if len(z2_failed) > 5:
        print(f"  --> ... and {len(z2_failed) - 5} more.")
    for idx in z2_failed:
        shutil.copy(paths[idx], os.path.join(FAILED_BIN_DIR, f"z2_fail_{idx}_{os.path.basename(paths[idx])}"))

    passed = [i for i in range(n) if i not in z2_failed_set]
    passed_t = torch.as_tensor(passed)
    print(f"Surviving images: {len(passed)}")

    plt.figure(figsize=(10, 6))
    plt.hist(z2.numpy(), bins=30, color="skyblue", edgecolor="black")
    plt.axvline(z2_threshold.item(), color="red", linestyle="dashed", linewidth=2,
                label=f"{Z2_PERCENTILE:.0%} Percentile Threshold")
    plt.title("Z^2 Energy Distribution")
    plt.xlabel("Z^2 Energy Score")
    plt.ylabel("Number of Images")
    plt.legend()
    plt.savefig("z_square_energy_distribution.png")
    plt.close()
    print("  --> Saved graph: 'z_square_energy_distribution.png'")

    # ------------------------------------------
    # Stage 2: temperature scaling
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 2: TEMPERATURE SCALING (L-BFGS)")
    print("==========================================")
    p_logits, p_labels = student_logits[passed_t], labels[passed_t]
    ece_before = expected_calibration_error(p_logits, p_labels)
    cal_logits, optimal_t, hit_bound = calibrate_temperature(p_logits, p_labels)
    ece_after = expected_calibration_error(cal_logits, p_labels)
    print(f"Optimal temperature T = {optimal_t:.4f}" + ("  (hit search bound: logits carry little signal)" if hit_bound else ""))
    print(f"ECE before: {ece_before:.4f} | after: {ece_after:.4f} "
          f"({'improved' if ece_after < ece_before else 'not improved'})")

    ce = nn.CrossEntropyLoss()
    t_values = torch.linspace(0.25, max(3.0, optimal_t * 1.5), 60)
    with torch.no_grad():
        nll = [ce(p_logits / t, p_labels).item() for t in t_values]
    plt.figure(figsize=(8, 5))
    plt.plot(t_values.numpy(), nll, linewidth=3, color="darkorange")
    plt.axvline(optimal_t, color="green", linestyle="dashed", label=f"Optimal T = {optimal_t:.2f}")
    plt.title("L-BFGS Temperature Optimization")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Negative Log-Likelihood Loss (NLL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("calibration_curve.png")
    plt.close()
    print("  --> Saved graph: 'calibration_curve.png'")

    # ------------------------------------------
    # Stage 3: CLIP teacher (real embeddings + linear probe)
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 3: LINEAR PROBING ON CLIP TEACHER")
    print("==========================================")
    clip = clip or load_clip()
    print(f"Extracting real CLIP embeddings for {n} images...")
    embeddings = extract_clip_embeddings(paths, clip)
    teacher_logits = teacher_oof_logits(embeddings, labels, fold_idx, num_classes)
    teacher_pred = teacher_logits.argmax(1)
    teacher_acc = (teacher_pred == labels).float().mean().item()
    print(f"Teacher (CLIP + linear probe) held-out accuracy: {teacher_acc:.1%}")
    final_probe = train_linear_probe(embeddings, labels, num_classes)   # all data; reusable on new images
    os.makedirs("omni_dataset", exist_ok=True)
    torch.save({"state_dict": final_probe.state_dict(), "classes": classes}, os.path.join("omni_dataset", "clip_probe.pth"))

    # ------------------------------------------
    # Stage 4: JSD hallucination gate
    # ------------------------------------------
    print("\n==========================================")
    print("STAGE 4: JSD HALLUCINATION GATE")
    print("==========================================")
    t_probs = F.softmax(teacher_logits[passed_t], dim=-1)
    s_probs = F.softmax(cal_logits, dim=-1)
    jsd = compute_jsd(t_probs, s_probs)
    jsd_threshold = torch.quantile(jsd, JSD_PERCENTILE)
    print(f"Target: {JSD_PERCENTILE:.0%} percentile disagreement | JSD threshold: {jsd_threshold:.4f}")
    jsd_failed_local = torch.where(jsd > jsd_threshold)[0].tolist()
    jsd_failed = [passed[j] for j in jsd_failed_local]
    print(f"[ACTION] Caught {len(jsd_failed)} student hallucinations:")
    for j in jsd_failed_local[:5]:
        idx = passed[j]
        print(f"  --> Image {idx:<4} | class {classes[labels_np[idx]]:<14} | JSD {jsd[j]:.4f} (-> failed_bin)")
    if len(jsd_failed_local) > 5:
        print(f"  --> ... and {len(jsd_failed_local) - 5} more.")
    jsd_failed_set = set(jsd_failed)
    for idx in jsd_failed:
        shutil.copy(paths[idx], os.path.join(FAILED_BIN_DIR, f"jsd_fail_{idx}_{os.path.basename(paths[idx])}"))

    safe = [i for i in passed if i not in jsd_failed_set]
    for idx in safe:
        shutil.copy(paths[idx], os.path.join(SAFE_BIN_DIR, f"safe_{idx}_{os.path.basename(paths[idx])}"))

    # ------------------------------------------
    # Report (real per-image results for the dashboard)
    # ------------------------------------------
    jsd_by_idx = {passed[j]: float(jsd[j]) for j in range(len(passed))}
    items = []
    for i in range(n):
        route = "failed_z2" if i in z2_failed_set else ("failed_jsd" if i in jsd_failed_set else "safe")
        items.append({
            "index": i, "path": paths[i], "class": classes[labels_np[i]], "split": curation["samples"][i]["split"],
            "student_pred": classes[int(student_pred[i])], "teacher_pred": classes[int(teacher_pred[i])],
            "z2": float(z2[i]), "jsd": jsd_by_idx.get(i), "route": route,
        })

    def group_acc(idxs):
        return float((student_pred[idxs] == labels[idxs]).float().mean()) if idxs else None

    failed_all = sorted(z2_failed_set | jsd_failed_set)
    acc_safe, acc_failed = group_acc(safe), group_acc(failed_all)

    print("\n==========================================")
    print("PIPELINE COMPLETE (triage)")
    print(f"failed_bin: {len(os.listdir(FAILED_BIN_DIR))} images | safe_bin: {len(os.listdir(SAFE_BIN_DIR))} images")
    if acc_safe is not None and acc_failed is not None:
        print(f"Student held-out accuracy: safe {acc_safe:.1%} vs flagged {acc_failed:.1%}")
    print("==========================================\n")

    return {
        "classes": classes, "num_images": n, "folds": k, "epochs": epochs, "student_class": factory.class_name,
        "student_accuracy": student_acc, "teacher_accuracy": teacher_acc,
        "z2_threshold": float(z2_threshold), "z2_flagged": len(z2_failed),
        "temperature": optimal_t, "ece_before": ece_before, "ece_after": ece_after,
        "jsd_threshold": float(jsd_threshold), "jsd_flagged": len(jsd_failed),
        "safe_count": len(safe), "failed_count": len(failed_all),
        "student_accuracy_safe": acc_safe, "student_accuracy_flagged": acc_failed,
        "graphs": {"z2": os.path.abspath("z_square_energy_distribution.png"),
                   "calibration": os.path.abspath("calibration_curve.png")},
        "items": items,
    }