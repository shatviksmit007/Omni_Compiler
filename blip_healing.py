"""
OmniCompiler - BLIP multimodal healing (Stages 5-6).
Captions the real images in safe_bin / failed_bin with BLIP, clusters the failure captions
(TF-IDF + K-Means) and finds the words that define each failure mode (Laplace-smoothed log odds).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import math
import random
from collections import Counter

import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BlipForConditionalGeneration, BlipProcessor

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
STOP_WORDS = {"a", "an", "the", "in", "on", "of", "and", "is", "to", "with", "at", "for", "it",
              "this", "that", "up", "down", "image", "picture", "shows"}


def calculate_laplace_log_odds(safe_array, target_failed_array):
    """Laplace-smoothed (alpha=1) log odds. Positive = word is associated with the failure cluster."""
    def tokens(caption):
        return [w for w in caption.lower().split() if w not in STOP_WORDS]

    safe_counts = Counter(w for _, cap in safe_array for w in tokens(cap))
    failed_counts = Counter(w for _, cap in target_failed_array for w in tokens(cap))
    vocab = set(safe_counts) | set(failed_counts)
    V = len(vocab)
    n_safe, n_failed = sum(safe_counts.values()), sum(failed_counts.values())
    return {w: math.log((failed_counts.get(w, 0) + 1) / (n_failed + V))
               - math.log((safe_counts.get(w, 0) + 1) / (n_safe + V)) for w in vocab}


def _top_words(safe_array, items, k=3):
    scores = calculate_laplace_log_odds(safe_array, items)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"word": w, "log_odds": float(s)} for w, s in top]


def cluster_and_analyze_failures(safe_array, failed_array):
    """TF-IDF + K-Means over failure captions, then log odds per cluster. Returns a list of cluster dicts."""
    n = len(failed_array)
    if n < 2:
        print(f"Only {n} anomalies found. Not enough to cluster.")
        return []

    num_clusters = min(3, max(1, n // 2))
    groups = {0: list(failed_array)}
    if num_clusters > 1:
        try:
            X = TfidfVectorizer(stop_words="english").fit_transform([c for _, c in failed_array])
            labels = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(X)
            groups = {i: [] for i in range(num_clusters)}
            for item, label in zip(failed_array, labels):
                groups[int(label)].append(item)
        except ValueError:                      # e.g. captions reduce to an empty vocabulary
            groups = {0: list(failed_array)}

    clusters = []
    for cid, items in groups.items():
        if not items:
            continue
        top = _top_words(safe_array, items)
        print(f"\n--- Failure Cluster {len(clusters) + 1} ({len(items)} images) ---")
        print(f"  Semantic profile example: '{items[0][1]}'")
        print("  Top 3 words defining this failure mode:")
        for i, t in enumerate(top, 1):
            print(f"    {i}. '{t['word']}' (log odds: {t['log_odds']:+.4f})")
        clusters.append({"cluster": len(clusters) + 1, "size": len(items), "example_caption": items[0][1],
                         "top_words": top, "files": [f for f, _ in items]})
    return clusters


def load_blip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading BLIP ({BLIP_MODEL_ID}) on {device}...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID).to(device).eval()
    return processor, model, device


def caption_images(paths, blip, batch_size=8):
    """Real BLIP captions for a list of image paths (batched)."""
    processor, model, device = blip
    captions = []
    for i in range(0, len(paths), batch_size):
        images = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        captions.extend(c.strip() for c in processor.batch_decode(out, skip_special_tokens=True))
    return captions


def _list_images(directory):
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS))


def run_blip_analytics(failed_dir="failed_bin", safe_dir="safe_bin", max_safe=200, blip=None):
    """Captions both bins and analyses the failure modes. Returns a dict for the pipeline report."""
    if not os.path.isdir(safe_dir) or not os.path.isdir(failed_dir):
        raise FileNotFoundError("failed_bin / safe_bin not found - run the math triage stage first.")

    safe_paths, failed_paths = _list_images(safe_dir), _list_images(failed_dir)
    if len(safe_paths) > max_safe:                       # the safe bin is only the log-odds background
        safe_paths = sorted(random.Random(42).sample(safe_paths, max_safe))
    blip = blip or load_blip()

    print(f"\nCaptioning {len(safe_paths)} clean images from {safe_dir}...")
    safe_array = list(zip((os.path.basename(p) for p in safe_paths), caption_images(safe_paths, blip)))
    print(f"Captioning {len(failed_paths)} anomalous images from {failed_dir}...")
    failed_array = list(zip((os.path.basename(p) for p in failed_paths), caption_images(failed_paths, blip)))

    print("\n==========================================")
    print("STAGE 5: BLIP MULTIMODAL CAPTIONING RESULTS")
    print("==========================================")
    print("\n--- Sample captions from safe bin ---")
    for f, c in safe_array[:3]:
        print(f"  {f:<40} | '{c}'")
    print("\n--- Sample captions from failed bin ---")
    for f, c in failed_array[:3]:
        print(f"  {f:<40} | '{c}'")

    print("\n==========================================")
    print("STAGE 6: FAILURE CLUSTERING & LAPLACE LOG ODDS")
    print("==========================================")
    clusters = cluster_and_analyze_failures(safe_array, failed_array)

    return {
        "safe_captions": [{"file": f, "caption": c} for f, c in safe_array],
        "failed_captions": [{"file": f, "caption": c} for f, c in failed_array],
        "clusters": clusters,
    }


if __name__ == "__main__":
    run_blip_analytics()