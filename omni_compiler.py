"""
OmniCompiler - Closed-Loop Autonomous Dataset Curation, Mathematical Triage & Self-Healing Pipeline.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import argparse
import inspect
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

import blip_healing
import cli5
import maths_unified
import tinyFish

console = Console()
BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "omni_report.json"
ACCENT = cli5.ACCENT


def parse_args():
    p = argparse.ArgumentParser(description="OmniCompiler Closed-Loop Self-Healing Pipeline")
    p.add_argument("--train-per-class", type=int, default=8, help="Training images per class for BOTH cycles (default: 8)")
    p.add_argument("--val-per-class", type=int, default=2, help="Validation images per class for BOTH cycles (default: 2)")
    p.add_argument("--sources", default=",".join(tinyFish.SOURCES), help=f"Comma list of: {', '.join(tinyFish.SOURCES)}")
    p.add_argument("--folds", type=int, default=5, help="Cross-fitting folds for student + teacher")
    p.add_argument("--epochs", type=int, default=maths_unified.STUDENT_EPOCHS, help="Student epochs per fold")
    return p.parse_args()


def stage(title):
    console.print()
    console.print(Rule(f"[bold {ACCENT}]{title}[/]", style="grey30"))


def fail(title, message):
    console.print(Panel(f"[bold red]{message}[/]", title=f"[bold red]{title}[/]", border_style="red"))


def write_report(report):
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]Report saved: {REPORT_PATH}[/dim]")


def ensure_class_indices(curation: dict, classes: list) -> dict:
    idx_map = {c.lower(): i for i, c in enumerate(classes)}
    for s in curation.get("samples", []):
        if "class_idx" not in s:
            lbl = s.get("label", "")
            s["class_idx"] = idx_map.get(lbl.lower(), 0)
    return curation


def call_math_pipeline(curation, model_path, folds, epochs, clip, update_bins=True, stage_label="INITIAL"):
    sig = inspect.signature(maths_unified.run_math_pipeline)
    kwargs = {"curation": curation, "model_path": model_path, "folds": folds, "epochs": epochs, "clip": clip}
    if "update_bins" in sig.parameters:
        kwargs["update_bins"] = update_bins
    if "stage_label" in sig.parameters:
        kwargs["stage_label"] = stage_label

    res = maths_unified.run_math_pipeline(**kwargs)
    classes = curation["classes"]
    items = res.get("items", [])
    if "surviving_after_z2" not in res:
        res["surviving_after_z2"] = res["num_images"] - res.get("z2_flagged", 0)
    if "student_per_class_accuracy" not in res:
        per_cls = {}
        for c in classes:
            c_items = [it for it in items if it.get("class") == c and it.get("student_pred") is not None]
            per_cls[c] = (
                sum(1 for it in c_items if it["student_pred"] == c) / len(c_items)
                if c_items else res.get("student_accuracy", 0.0)
            )
        res["student_per_class_accuracy"] = per_cls
    if "teacher_per_class_accuracy" not in res:
        t_per_cls = {}
        for c in classes:
            c_items = [it for it in items if it.get("class") == c and it.get("teacher_pred") is not None]
            t_per_cls[c] = (
                sum(1 for it in c_items if it["teacher_pred"] == c) / len(c_items)
                if c_items else res.get("teacher_accuracy", 0.0)
            )
        res["teacher_per_class_accuracy"] = t_per_cls
    if "mean_jsd" not in res:
        jsds = [it["jsd"] for it in items if it.get("jsd") is not None]
        res["mean_jsd"] = sum(jsds) / len(jsds) if jsds else 0.0
    if "failed_records" not in res:
        f_recs = []
        for it in items:
            if it.get("route") in ("failed_z2", "failed_jsd"):
                prefix = "z2_fail" if it["route"] == "failed_z2" else "jsd_fail"
                bname = f"{prefix}_{it['index']}_{os.path.basename(it['path'])}"
                f_recs.append({"index": it["index"], "original_path": it["path"], "bin_file": bname, "class": it["class"], "class_idx": classes.index(it["class"])})
        res["failed_records"] = f_recs
    if "safe_records" not in res:
        s_recs = []
        for it in items:
            if it.get("route") == "safe":
                s_recs.append({"index": it["index"], "path": it["path"], "label": it["class"], "class_idx": classes.index(it["class"]), "split": it.get("split", "train")})
        res["safe_records"] = s_recs
    if "failed_counts_per_class" not in res:
        res["failed_counts_per_class"] = dict(Counter(r["class"] for r in res["failed_records"]))
    return res


def call_blip_analytics(classes, failed_records):
    sig = inspect.signature(blip_healing.run_blip_analytics)
    kwargs = {"failed_dir": maths_unified.FAILED_BIN_DIR, "safe_dir": maths_unified.SAFE_BIN_DIR}
    if "classes" in sig.parameters:
        kwargs["classes"] = classes
    if "failed_records" in sig.parameters:
        kwargs["failed_records"] = failed_records
    return blip_healing.run_blip_analytics(**kwargs)


def run_healing_curation(classes, healing_queries, target_train: int, target_val: int, chosen_sources, clip, seen_urls, seen_hashes):
    sig = inspect.signature(tinyFish.run_curation)
    per_class_targets = {cls_name: (target_train, target_val) for cls_name in classes}

    if "custom_search_queries" in sig.parameters:
        curation_r2 = tinyFish.run_curation(
            labels=classes,
            dataset_query="",
            target_train=target_train,
            target_val=target_val,
            sources=chosen_sources,
            clip=clip,
            custom_search_queries=healing_queries,
            per_class_targets=per_class_targets,
            seen_urls=seen_urls,
            seen_hashes=seen_hashes,
        )
        return ensure_class_indices(curation_r2, classes)
    return {}


def print_comparison_table(pre_triage: dict, post_triage: dict, healing_queries: dict):
    table = Table(title="OMNICOMPILER SELF-HEALING METRICS COMPARISON (PRE vs. POST HEALING)", border_style="grey30")
    table.add_column("Metric", style="bold")
    table.add_column("Round 1 (Initial)", justify="right")
    table.add_column("Round 2 (After Healing)", justify="right")
    table.add_column("Delta / Status", justify="right")

    table.add_row("Evaluated Dataset Size (post-Z²)", f"{pre_triage['surviving_after_z2']} / {pre_triage['num_images']}", f"{post_triage['surviving_after_z2']} / {post_triage['num_images']}", f"{post_triage['num_images'] - pre_triage['num_images']:+d} total imgs")
    
    s_pre, s_post = pre_triage["student_accuracy"], post_triage["student_accuracy"]
    table.add_row("Student Overall Accuracy", f"{s_pre:.1%}", f"{s_post:.1%}", f"[bold {ACCENT}]{(s_post - s_pre) * 100:+.1f}%[/]")

    for cls_name in pre_triage["classes"]:
        c_pre = pre_triage["student_per_class_accuracy"].get(cls_name, 0.0)
        c_post = post_triage["student_per_class_accuracy"].get(cls_name, 0.0)
        hq = healing_queries.get(cls_name, cls_name)
        table.add_row(f"  • Class '{cls_name}' Accuracy (healed via '{hq}')", f"{c_pre:.1%}", f"{c_post:.1%}", f"{(c_post - c_pre) * 100:+.1f}%")

    t_pre, t_post = pre_triage["teacher_accuracy"], post_triage["teacher_accuracy"]
    table.add_row("CLIP Teacher (Linear Probe) Overall Accuracy", f"{t_pre:.1%}", f"{t_post:.1%}", f"{(t_post - t_pre) * 100:+.1f}%")

    ece_pre, ece_post = pre_triage["ece_after"], post_triage["ece_after"]
    table.add_row("Calibrated ECE (lower is better)", f"{ece_pre:.4f} (T={pre_triage['temperature']:.2f})", f"{ece_post:.4f} (T={post_triage['temperature']:.2f})", f"{ece_post - ece_pre:+.4f}")

    jsd_pre, jsd_post = pre_triage["mean_jsd"], post_triage["mean_jsd"]
    table.add_row("Mean Teacher-Student JSD (lower is better)", f"{jsd_pre:.4f}", f"{jsd_post:.4f}", f"{jsd_post - jsd_pre:+.4f}")

    console.print()
    console.print(table)


def main():
    args = parse_args()
    console.clear()
    console.print(f"[bold {ACCENT}]==================================================[/]")
    console.print("[bold #F5F5F0]OMNICOMPILER CLOSED-LOOP SELF-HEALING PIPELINE[/]")
    console.print(f"[bold {ACCENT}]==================================================[/]\n")

    config = cli5.collect_config()
    if config is None:
        return
    task, query, model_path, labels = config["task"], config["dataset_query"], config["model_path"], config["labels"]
    classes = tinyFish.normalize_labels(labels, query)
    config["labels"] = classes
    triage_enabled = len(classes) >= 2

    chosen_sources = [s.strip() for s in args.sources.split(",") if s.strip() in tinyFish.SOURCES]

    stage(f"STAGE 2: CYCLE 1 TINYFISH CURATION ({args.train_per_class} TRAIN + {args.val_per_class} VAL PER CLASS)")
    clip = tinyFish.load_clip_gate()
    curation_r1 = tinyFish.run_curation(labels=classes, dataset_query=query, target_train=args.train_per_class, target_val=args.val_per_class, sources=chosen_sources, clip=clip)
    curation_r1 = ensure_class_indices(curation_r1, classes)
    tinyFish.print_curation_summary(curation_r1)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "config": {k: config[k] for k in ("task", "dataset_query", "model_path", "labels")}}

    stage("STAGE 3: INITIAL MATH TRIAGE (Z² OUTLIERS -> STUDENT -> TEMP SCALING -> LINEAR PROBE -> JSD)")
    pre_triage = call_math_pipeline(curation_r1, model_path, folds=args.folds, epochs=args.epochs, clip=clip, update_bins=True, stage_label="ROUND 1 (PRE-HEALING)")
    report["triage"] = pre_triage

    stage("STAGE 4: BLIP MULTIMODAL CAPTIONING, KNN/K-MEANS & HEALING QUERY GENERATION")
    healing_info = call_blip_analytics(classes=classes, failed_records=pre_triage["failed_records"])
    report["healing"] = healing_info

    stage(f"STAGE 5: CYCLE 2 HEALING TINYFISH RE-CURATION ({args.train_per_class} TRAIN + {args.val_per_class} VAL PER CLASS)")
    healing_queries = healing_info.get("healing_queries", {})

    curation_r2 = run_healing_curation(classes=classes, healing_queries=healing_queries, target_train=args.train_per_class, target_val=args.val_per_class, chosen_sources=chosen_sources, clip=clip, seen_urls=curation_r1.get("seen_urls"), seen_hashes=curation_r1.get("seen_hashes"))
    tinyFish.print_curation_summary(curation_r2)

    healed_samples = [{"path": r["path"], "label": r["label"], "class_idx": r["class_idx"], "split": r.get("split", "train")} for r in pre_triage["safe_records"]] + curation_r2.get("samples", [])
    healed_counts = {c: {"train": 0, "val": 0} for c in classes}
    for s in healed_samples:
        split_key = "val" if s.get("split") == "val" else "train"
        healed_counts[s["label"]][split_key] += 1

    healed_curation = {"labels": classes, "classes": classes, "search_queries": healing_queries, "counts": healed_counts, "samples": healed_samples}

    stage("STAGE 6: POST-HEALING RE-EVALUATION & METRICS COMPARISON")
    post_triage = call_math_pipeline(healed_curation, model_path, folds=args.folds, epochs=args.epochs, clip=clip, update_bins=False, stage_label="ROUND 2 (POST-HEALING)")
    report["triage_post_healing"] = post_triage

    write_report(report)
    print_comparison_table(pre_triage, post_triage, healing_queries)

    console.print(f"\n[bold {ACCENT}]==================================================[/]")
    console.print("[bold #F5F5F0]CLOSED-LOOP SELF-HEALING PIPELINE COMPLETE. EXITING.[/]")
    console.print(f"[bold {ACCENT}]==================================================[/]\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Pipeline execution cancelled by user.[/dim]")