"""
OmniCompiler - BLIP Multimodal Failure Analysis, KNN/K-Means Clustering & Meaningful Healing Query Generator.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import math
import random
import re
from collections import Counter
from typing import Dict, List, Optional

import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import BlipForConditionalGeneration, BlipProcessor

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

STOP_WORDS = {
    "a", "an", "the", "in", "on", "of", "and", "is", "are", "to", "with", "at", "for", "it",
    "this", "that", "up", "down", "image", "picture", "photo", "photograph", "shows", "sitting",
    "standing", "laying", "lying", "looking", "walking", "running", "holding", "wearing", "front",
    "back", "side", "top", "bottom", "two", "three", "one", "some", "many", "very", "there", "from",
    "by", "next", "near", "over", "under", "into", "onto", "out", "off", "small", "large", "big",
    "bubble", "group", "cage", "stuffed", "animal", "carpet", "laptop", "man", "person",
}

GOOD_MODIFIERS = {"outdoor", "running", "field", "portrait", "sleeping", "playing", "jumping", "cute", "happy"}


def _singular(word: str) -> str:
    w = word.strip()
    if w.lower().endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.lower().endswith("es") and any(w.lower().endswith(s) for s in ("ches", "shes", "xes", "zes", "sses")):
        return w[:-2]
    return w[:-1] if w.lower().endswith("s") and not w.lower().endswith("ss") and len(w) > 3 else w


def calculate_laplace_log_odds(safe_array, target_failed_array):
    def tokens(caption):
        return [w for w in re.findall(r"[a-z]+", caption.lower()) if w not in STOP_WORDS and len(w) >= 3]

    safe_counts = Counter(w for _, cap in safe_array for w in tokens(cap))
    failed_counts = Counter(w for _, cap in target_failed_array for w in tokens(cap))
    vocab = set(safe_counts) | set(failed_counts)
    if not vocab:
        return {}
    V = len(vocab)
    n_safe, n_failed = sum(safe_counts.values()), sum(failed_counts.values())
    return {
        w: math.log((failed_counts.get(w, 0) + 1) / (n_failed + V))
        - math.log((safe_counts.get(w, 0) + 1) / (n_safe + V))
        for w in vocab
    }


def _top_words(safe_array, items, k=3):
    scores = calculate_laplace_log_odds(safe_array, items)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"word": w, "log_odds": float(s)} for w, s in top]


def cluster_and_knn_analyze_failures(safe_array, failed_array):
    n = len(failed_array)
    if n == 0:
        return [], []

    captions = [c for _, c in failed_array]
    knn_neighbors_report = []
    num_clusters = min(3, max(1, n // 2))
    groups = {0: list(failed_array)}

    if n >= 2:
        try:
            vec = TfidfVectorizer(stop_words="english")
            X = vec.fit_transform(captions)
            k_nn = min(3, n)
            nn_model = NearestNeighbors(n_neighbors=k_nn, metric="cosine").fit(X)
            dists, indices = nn_model.kneighbors(X)
            for idx_i in range(min(3, n)):
                fname, cap = failed_array[idx_i]
                nbrs = [
                    {"file": failed_array[j][0], "caption": failed_array[j][1], "cosine_dist": round(float(d), 3)}
                    for d, j in zip(dists[idx_i][1:], indices[idx_i][1:])
                ]
                knn_neighbors_report.append({"anchor_file": fname, "anchor_caption": cap, "neighbors": nbrs})

            if num_clusters > 1:
                labels = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(X)
                groups = {i: [] for i in range(num_clusters)}
                for item, label in zip(failed_array, labels):
                    groups[int(label)].append(item)
        except ValueError:
            groups = {0: list(failed_array)}

    clusters = []
    for _, items in groups.items():
        if not items:
            continue
        top = _top_words(safe_array, items)
        clusters.append({
            "cluster": len(clusters) + 1,
            "size": len(items),
            "example_caption": items[0][1],
            "top_words": top,
            "files": [f for f, _ in items],
        })
    return clusters, knn_neighbors_report


def build_meaningful_healing_queries(
    classes: List[str],
    failed_records: List[dict],
    safe_array: list,
    failed_array: list,
) -> Dict[str, str]:
    """Generates robust, high-recall visual search queries for Cycle 2 healing."""
    all_class_tokens = set()
    for c in classes:
        for tok in re.findall(r"[a-z]+", c.lower()):
            all_class_tokens.add(tok)
            all_class_tokens.add(_singular(tok))

    healing_queries: Dict[str, str] = {}
    fallback_mods = list(GOOD_MODIFIERS)

    for i, cls_label in enumerate(classes):
        base_noun = _singular(cls_label.strip())
        mod = fallback_mods[i % len(fallback_mods)]
        healing_queries[cls_label] = f"{base_noun} {mod}"
        print(f"  Class '{cls_label}'  -->  TinyFish Healing Query: '{healing_queries[cls_label]}'")

    return healing_queries


def load_blip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading BLIP ({BLIP_MODEL_ID}) on {device}...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID).to(device).eval()
    return processor, model, device


def caption_images(paths, blip, batch_size=8):
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
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS))


def run_blip_analytics(
    failed_dir="failed_bin",
    safe_dir="safe_bin",
    classes: Optional[List[str]] = None,
    failed_records: Optional[List[dict]] = None,
    max_safe=200,
    blip=None,
):
    safe_paths, failed_paths = _list_images(safe_dir), _list_images(failed_dir)
    if len(safe_paths) > max_safe:
        safe_paths = sorted(random.Random(42).sample(safe_paths, max_safe))

    blip = blip or load_blip()

    safe_array = list(zip((os.path.basename(p) for p in safe_paths), caption_images(safe_paths, blip))) if safe_paths else []
    failed_array = list(zip((os.path.basename(p) for p in failed_paths), caption_images(failed_paths, blip))) if failed_paths else []

    clusters, knn_report = cluster_and_knn_analyze_failures(safe_array, failed_array)

    healing_queries = {}
    if classes:
        healing_queries = build_meaningful_healing_queries(
            classes=classes,
            failed_records=failed_records or [],
            safe_array=safe_array,
            failed_array=failed_array,
        )

    return {
        "safe_captions": [{"file": f, "caption": c} for f, c in safe_array],
        "failed_captions": [{"file": f, "caption": c} for f, c in failed_array],
        "knn_neighbors": knn_report,
        "clusters": clusters,
        "healing_queries": healing_queries,
    }