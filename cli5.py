from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import questionary
from questionary import Choice
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich import box


# ============================================================
# OMNICOMPILER CONFIGURATION
# ============================================================

APP_NAME = "OMNICOMPILER"
APP_SUBTITLE = "AUTONOMOUS DATASET CURATION ENGINE"
APP_VERSION = "CLI v2.1.0"

BG = "#0A0A0A"
CARD = "#222220"
HEADING = "#F5F5F0"
SECONDARY = "#9B9B96"
ACCENT = "#BEFF00"

console = Console()

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = BASE_DIR / "dashboard.py"

TASKS = [
    {
        "number": "01",
        "name": "Image Classification",
        "description": "Assign classes to images",
        "value": "Image Classification",
    },
    {
        "number": "02",
        "name": "Object Detection",
        "description": "Locate and classify objects",
        "value": "Object Detection",
    },
    {
        "number": "03",
        "name": "Image Captioning",
        "description": "Generate image descriptions",
        "value": "Image Captioning",
    },
]

FRAMEWORK_MODULES = {
    "torch": "PyTorch",
    "sklearn": "scikit-learn",
}

TRAINING_CALLS = {"fit", "fit_predict", "partial_fit", "backward", "step", "train"}


# ============================================================
# TERMINAL DESIGN
# ============================================================

def clear_screen():
    console.clear()


def show_header(step: int, title: str):
    clear_screen()
    progress = Text()
    for i in range(1, 5):
        if i <= step:
            progress.append("━━━━", style=f"bold {ACCENT}")
        else:
            progress.append("━━━━", style="dim")
        if i < 4:
            progress.append("  ")

    console.print(progress)
    console.print()
    console.print(
        Panel(
            Text.assemble(
                (APP_NAME + "\n", f"bold {ACCENT}"),
                (APP_SUBTITLE + "\n", f"bold {HEADING}"),
                (APP_VERSION, f"dim {SECONDARY}"),
            ),
            border_style="grey30",
            box=box.SQUARE,
            padding=(1, 2),
        )
    )
    console.print(Text("Build better datasets. Train better models.", style=f"italic {SECONDARY}"))
    console.print()
    console.print(
        Rule(
            Text.assemble(
                (f"STEP {step}/4 ", f"bold {ACCENT}"),
                (title.upper(), f"bold {HEADING}"),
            ),
            style="grey30",
        )
    )
    console.print()


def show_error(title: str, problem: str, fix: str):
    console.print()
    console.print(
        Panel(
            f"[bold red]✖ {problem}[/bold red]\n\n[white]How to fix:[/white] {fix}",
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            box=box.HEAVY,
            padding=(1, 2),
        )
    )
    console.print()


def ask_text(message: str, default: str = ""):
    return questionary.text(
        message,
        default=default,
        qmark="?",
        style=questionary.Style([
            ("question", f"bold {HEADING}"),
            ("answer", f"bold {ACCENT}"),
            ("pointer", f"bold {ACCENT}"),
            ("highlighted", f"bold {ACCENT}"),
            ("instruction", SECONDARY),
        ]),
    ).ask()


# ============================================================
# STEP 1: TASK SELECTION
# ============================================================

def select_task():
    show_header(1, "Task Selection")
    console.print(Text("Choose the computer vision task for your project.", style=SECONDARY))
    console.print()

    choices = []
    for task in TASKS:
        title = f"{task['number']}    {task['name']}    —    {task['description']}"
        choices.append(Choice(title=title, value=task["value"]))

    selected = questionary.select(
        "What are you building?",
        choices=choices,
        instruction="(Use ↑/↓ arrows and Enter to select)",
        pointer="»",
        use_shortcuts=False,
        use_arrow_keys=True,
        use_indicator=True,
        qmark="?",
        style=questionary.Style([
            ("question", f"bold {HEADING}"),
            ("answer", f"bold {ACCENT}"),
            ("pointer", f"bold {ACCENT}"),
            ("highlighted", f"bold {ACCENT}"),
            ("selected", f"bold {ACCENT}"),
            ("instruction", SECONDARY),
        ]),
    ).ask()
    return selected


# ============================================================
# STEP 2: DATASET DISCOVERY
# ============================================================

def dataset_setup():
    show_header(2, "Dataset Discovery")
    console.print(Text("Describe the dataset you want OmniCompiler to find.", style=SECONDARY))
    console.print()

    while True:
        query = ask_text("Dataset search query")
        if query is None:
            return None
        query = query.strip()
        if not query:
            show_error(
                "MISSING INPUT",
                "Dataset query cannot be empty.",
                "Describe the images you need, e.g. 'cats and dogs'.",
            )
            continue
        return query


# ============================================================
# STEP 3: MODEL CONFIGURATION (WITH VALIDATION)
# ============================================================

def inspect_model_file(path: Path):
    try:
        source = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return [], "The file is not readable text (it is not UTF-8 encoded).", "file"
    except OSError as e:
        return [], f"The file could not be read: {e}", "file"

    if not source.strip():
        return [], "The file is empty.", "file"

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [], f"The file has a Python syntax error on line {e.lineno}: {e.msg}", "file"

    frameworks = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FRAMEWORK_MODULES:
                    frameworks.add(FRAMEWORK_MODULES[root])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".")[0]
            if root in FRAMEWORK_MODULES:
                frameworks.add(FRAMEWORK_MODULES[root])

    if not frameworks:
        return [], "No PyTorch or scikit-learn import was found. This does not look like an ML training program.", "ml"

    has_ml_code = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in TRAINING_CALLS:
                has_ml_code = True
                break
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                if name == "Module":
                    has_ml_code = True
                    break
            if has_ml_code:
                break

    if not has_ml_code:
        return sorted(frameworks), "The file imports an ML library but contains no model or training code.", "ml"

    return sorted(frameworks), None, ""


def model_setup():
    show_header(3, "Model Configuration")
    console.print(Text("Provide the Python source file of your ML training program.", style=SECONDARY))
    console.print(Text("Supported frameworks: PyTorch, scikit-learn (or both).", style=SECONDARY))
    console.print()

    while True:
        model_path = ask_text("Model Python file path")
        if model_path is None:
            return None

        model_path = model_path.strip().strip('"').strip("'")
        if not model_path:
            show_error("MISSING INPUT", "No file path entered.", "Enter the path to your PyTorch or scikit-learn .py file.")
            continue

        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        path = path.resolve()

        if not path.exists():
            show_error("FILE NOT FOUND", f"No file exists at: {path}", "Check the spelling and try again.")
            continue

        if not path.is_file():
            show_error("NOT A FILE", f"This path is a folder, not a file: {path}", "Enter the path to a single .py file.")
            continue

        if path.suffix.lower() != ".py":
            show_error("WRONG FILE TYPE", f"Expected a .py file.", "Select a Python source file ending in .py.")
            continue

        frameworks, problem, kind = inspect_model_file(path)
        if problem:
            fix = "Open the file and ensure it runs on its own." if kind == "file" else "Use a script that trains a model."
            show_error("INVALID MODEL FILE", problem, fix)
            continue

        console.print(f"\n[bold {ACCENT}]✔ Valid model file detected:[/bold {ACCENT}] [{HEADING}]{' + '.join(frameworks)}[/{HEADING}]")
        time.sleep(1)
        return str(path)


# ============================================================
# STEP 4: LABELS + REVIEW
# ============================================================

def get_labels(task: str, default_query: str = ""):
    if task == "Image Captioning":
        return []

    show_header(4, "Class Labels")
    console.print(Text("Enter the class names separated by commas (or 'and').", style=SECONDARY))

    if task == "Image Classification":
        console.print(Text("Example: cat, dog", style=ACCENT))
    else:
        console.print(Text("Example: person, car, bicycle", style=ACCENT))
    console.print()

    suggested_parts = [
        p.strip()
        for p in re.split(r"\s*(?:,|\band\b|\bvs\.?\b|\bversus\b|\bor\b|/)\s*", default_query, flags=re.IGNORECASE)
        if p.strip()
    ]
    default_labels_str = ", ".join(suggested_parts)

    while True:
        raw = ask_text("Class labels", default=default_labels_str)
        if raw is None:
            return None

        labels = list(
            dict.fromkeys(
                label.strip()
                for label in re.split(r"\s*(?:,|\band\b|\bvs\.?\b|\bversus\b|\bor\b|/)\s*", raw, flags=re.IGNORECASE)
                if label.strip()
            )
        )

        if labels:
            return labels

        show_error(
            "MISSING INPUT",
            "No class labels entered.",
            "Enter at least one label, separated by commas, e.g. cat, dog.",
        )


def show_review(config: dict):
    while True:
        show_header(4, "Review Configuration")
        console.print(
            Panel(
                f"[bold {ACCENT}]TASK[/bold {ACCENT}]\n[{HEADING}]{config['task']}[/{HEADING}]\n\n"
                f"[bold {ACCENT}]DATASET QUERY[/bold {ACCENT}]\n[{HEADING}]{config['dataset_query']}[/{HEADING}]\n\n"
                f"[bold {ACCENT}]MODEL SOURCE[/bold {ACCENT}]\n[{HEADING}]{config['model_path'] or 'Not provided'}[/{HEADING}]\n\n"
                f"[bold {ACCENT}]CLASS LABELS[/bold {ACCENT}]\n[{HEADING}]{', '.join(config['labels']) if config['labels'] else 'N/A'}[/{HEADING}]",
                title=f"[bold {ACCENT}]CONFIGURATION[/bold {ACCENT}]",
                border_style="grey30",
                padding=(1, 2),
            )
        )
        console.print()

        action = questionary.select(
            "Continue?",
            choices=[
                Choice("Confirm configuration", value="launch"),
                Choice("Edit configuration", value="edit"),
                Choice("Cancel setup", value="cancel"),
            ],
            pointer="»",
            style=questionary.Style([
                ("question", f"bold {HEADING}"),
                ("pointer", f"bold {ACCENT}"),
                ("highlighted", f"bold {ACCENT}"),
                ("selected", f"bold {ACCENT}"),
            ]),
        ).ask()

        if action is None or action == "cancel":
            return None
        if action == "launch":
            return config
        if action == "edit":
            return "edit"


# ============================================================
# DASHBOARD LAUNCH
# ============================================================

def launch_dashboard(config: dict):
    if not DASHBOARD_PATH.exists():
        console.print(Panel("[bold yellow]Dashboard not found.[/bold yellow]", border_style="yellow"))
        return

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="omnicompiler_", encoding="utf-8", delete=False
        ) as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            temp_file = f.name

        env = os.environ.copy()
        env["OMNICOMPILER_CONFIG"] = temp_file

        console.print(
            Panel(
                f"[bold {ACCENT}]Launching OmniCompiler Dashboard[/bold {ACCENT}]\n\n[white]Local address:[/white]\n[bold]http://localhost:8501[/bold]",
                border_style="green",
            )
        )

        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_PATH), "--server.address", "127.0.0.1"],
            cwd=str(BASE_DIR),
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")
    except Exception as e:
        console.print(f"\n[bold red]Dashboard launch failed:[/bold red] {e}")
    finally:
        if temp_file:
            try:
                os.unlink(temp_file)
            except OSError:
                pass


# ============================================================
# MAIN APPLICATION
# ============================================================

def collect_config():
    config = {"task": "", "dataset_query": "", "model_path": "", "labels": []}

    task = select_task()
    if task is None:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
        return
    config["task"] = task

    query = dataset_setup()
    if query is None:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
        return
    config["dataset_query"] = query

    model_path = model_setup()
    if model_path is None:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
        return
    config["model_path"] = model_path

    labels = get_labels(task, config["dataset_query"])
    if labels is None:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
        return
    config["labels"] = labels

    while True:
        result = show_review(config)
        if result is None:
            console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
            return

        if result == "edit":
            edit = questionary.select(
                "What would you like to edit?",
                choices=["Task", "Dataset query", "Model path", "Class labels", "Back"],
                pointer="»",
                style=questionary.Style([("question", f"bold {HEADING}"), ("pointer", f"bold {ACCENT}"), ("highlighted", f"bold {ACCENT}")]),
            ).ask()

            if edit is None or edit == "Back":
                continue

            if edit == "Task":
                new_task = select_task()
                if new_task is not None:
                    config["task"] = new_task
                    config["labels"] = []
                    if new_task != "Image Captioning":
                        new_labels = get_labels(new_task, config["dataset_query"])
                        if new_labels is not None:
                            config["labels"] = new_labels
            elif edit == "Dataset query":
                new_query = dataset_setup()
                if new_query is not None:
                    config["dataset_query"] = new_query
            elif edit == "Model path":
                new_path = model_setup()
                if new_path is not None:
                    config["model_path"] = new_path
            elif edit == "Class labels":
                new_labels = get_labels(config["task"], config["dataset_query"])
                if new_labels is not None:
                    config["labels"] = new_labels
            continue

        return result


def main():
    config = collect_config()
    if config is not None:
        launch_dashboard(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")