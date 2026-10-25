"""
OmniCompiler - end-to-end pipeline.

  CLI (cli5)  ->  curation (tinyFish)  ->  student + Z^2 + calibration + CLIP teacher + JSD (maths_unified)
              ->  BLIP failure-mode analysis (blip_healing)  ->  omni_report.json  ->  dashboard

Every stage runs on the real curated images and on the model file you give the CLI.
"""
import os

# Must be set before torch / sklearn are imported anywhere (Windows duplicate-OpenMP crash).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import argparse
import json
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
    p = argparse.ArgumentParser(description="OmniCompiler end-to-end pipeline")
    p.add_argument("--train-per-class", type=int, default=40)
    p.add_argument("--val-per-class", type=int, default=10)
    p.add_argument("--sources", default=",".join(tinyFish.SOURCES),
                   help=f"comma list of: {', '.join(tinyFish.SOURCES)}")
    p.add_argument("--folds", type=int, default=5, help="cross-fitting folds for student + teacher")
    p.add_argument("--epochs", type=int, default=maths_unified.STUDENT_EPOCHS, help="student epochs per fold")
    p.add_argument("--no-dashboard", action="store_true")
    return p.parse_args()


def stage(title):
    console.print()
    console.print(Rule(f"[bold {ACCENT}]{title}[/]", style="grey30"))


def fail(title, message):
    console.print(Panel(f"[bold red]{message}[/]", title=f"[bold red]{title}[/]", border_style="red"))


def write_report(report):
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim]Report saved: {REPORT_PATH}[/dim]")


def main():
    args = parse_args()
    console.clear()
    console.print(f"[bold {ACCENT}]==================================================[/]")
    console.print("[bold #F5F5F0]OMNICOMPILER END-TO-END PIPELINE[/]")
    console.print(f"[bold {ACCENT}]==================================================[/]\n")

    # ---------------------------------------------------------
    # STAGE 1: CLI (cli5.py) - task, dataset query, model file, labels, review
    # ---------------------------------------------------------
    config = cli5.collect_config()
    if config is None:
        return
    task, query, model_path, labels = config["task"], config["dataset_query"], config["model_path"], config["labels"]
    classes = labels if labels else [query]          # captioning has no labels: one pseudo-class
    triage = len(classes) >= 2

    if task == "Object Detection":
        console.print("[yellow]Note: triage scores whole images (image-level object presence per class), "
                      "not bounding boxes.[/]")

    # Fail fast, before the slow download stage: can your model file actually be trained here?
    if triage:
        try:
            factory = maths_unified.load_student_factory(model_path)
            maths_unified.check_student(factory, len(classes))
            console.print(f"[bold {ACCENT}]✔ Student model:[/] {factory.class_name} "
                          f"({len(classes)} outputs) from {Path(model_path).name}")
        except Exception as e:
            fail("MODEL FILE CANNOT BE USED", str(e))
            return

    # ---------------------------------------------------------
    # STAGE 2: curation (tinyFish.py) - real, licensed, gate-checked images per class
    # ---------------------------------------------------------
    stage("STAGE 2: TINYFISH CURATION & NETSAFE GATE")
    clip = tinyFish.load_clip_gate()                 # one CLIP instance shared with the math stage
    curation = tinyFish.run_curation(
        labels=classes, dataset_query=query,
        target_train=args.train_per_class, target_val=args.val_per_class,
        sources=[s.strip() for s in args.sources.split(",")], clip=clip)
    tinyFish.print_curation_summary(curation)

    empty = [c for c, v in curation["counts"].items() if v["train"] + v["val"] == 0]
    if empty or not curation["samples"]:
        fail("CURATION FAILED", f"No images passed the safety gates for: {', '.join(empty) or 'any class'}.\n"
             "Check the rejection counts above, try a broader label, or set FLICKR_API_KEY / HF_TOKEN.")
        return

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: config[k] for k in ("task", "dataset_query", "model_path", "labels")},
        "curation": {k: curation[k] for k in ("labels", "classes", "search_queries", "counts")},
    }

    # ---------------------------------------------------------
    # Captioning / single-class runs: no classifier to triage -> caption what was curated
    # ---------------------------------------------------------
    if not triage:
        stage("STAGE 3: BLIP CAPTIONING (no triage: needs 2+ classes)")
        paths = [s["path"] for s in curation["samples"]]
        captions = blip_healing.caption_images(paths, blip_healing.load_blip())
        report["captions"] = [{"path": p, "caption": c} for p, c in zip(paths, captions)]
        for p, c in list(zip(paths, captions))[:5]:
            console.print(f"  {Path(p).name:<28} '{c}'")
        write_report(report)
        return

    # ---------------------------------------------------------
    # STAGES 3-4: student, Z^2, calibration, CLIP teacher, JSD (maths_unified.py)
    # ---------------------------------------------------------
    stage("STAGE 3: MATH TRIAGE (STUDENT, Z², CALIBRATION, CLIP TEACHER, JSD)")
    try:
        report["triage"] = maths_unified.run_math_pipeline(
            curation, model_path, folds=args.folds, epochs=args.epochs, clip=clip)
    except Exception as e:
        fail("MATH TRIAGE FAILED", f"{type(e).__name__}: {e}")
        write_report(report)
        return

    # ---------------------------------------------------------
    # STAGES 5-6: BLIP failure-mode analysis (blip_healing.py)
    # ---------------------------------------------------------
    stage("STAGE 4: BLIP MULTIMODAL CAPTIONING & LOG ODDS")
    try:
        report["healing"] = blip_healing.run_blip_analytics(
            failed_dir=maths_unified.FAILED_BIN_DIR, safe_dir=maths_unified.SAFE_BIN_DIR)
    except Exception as e:
        fail("BLIP ANALYSIS FAILED", f"{type(e).__name__}: {e}")

    write_report(report)

    # ---------------------------------------------------------
    # Summary + dashboard
    # ---------------------------------------------------------
    t = report["triage"]
    table = Table(title="RUN SUMMARY", border_style="grey30")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Images curated", str(t["num_images"]))
    table.add_row("Student held-out accuracy", f"{t['student_accuracy']:.1%}")
    table.add_row("CLIP teacher held-out accuracy", f"{t['teacher_accuracy']:.1%}")
    table.add_row("Temperature / ECE (before → after)",
                  f"{t['temperature']:.2f} / {t['ece_before']:.3f} → {t['ece_after']:.3f}")
    table.add_row("Flagged by Z² / JSD", f"{t['z2_flagged']} / {t['jsd_flagged']}")
    table.add_row("safe_bin / failed_bin", f"{t['safe_count']} / {t['failed_count']}")
    console.print(table)

    console.print(f"\n[bold {ACCENT}]==================================================[/]")
    console.print("[bold #F5F5F0]PIPELINE EXECUTION COMPLETE.[/]")
    console.print(f"[bold {ACCENT}]==================================================[/]\n")

    if not args.no_dashboard:
        config["report_path"] = str(REPORT_PATH)
        cli5.launch_dashboard(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Pipeline execution cancelled by user.[/dim]")