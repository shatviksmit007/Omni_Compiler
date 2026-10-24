# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    config = {
        "task": "",
        "dataset_query": "",
        "model_path": "",
        "labels": [],
    }

    # STEP 1: Select task
    task = select_task()

    if task is None:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
        return

    config["task"] = task

    # STEP 2: Dataset discovery query
    query = dataset_setup()

    if query is None:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
        return

    config["dataset_query"] = query

    # STEP 3: Model configuration
    model_path = model_setup()

    if model_path is None:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
        return

    config["model_path"] = model_path

    # STEP 4: Class labels
    labels = get_labels(task)

    if labels is None:
        console.print("\n[dim]OmniCompiler setup cancelled.[/dim]")
        return

    config["labels"] = labels

    # STEP 5: Review and edit configuration
    while True:
        result = show_review(config)

        if result is None:
            console.print(
                "\n[dim]OmniCompiler setup cancelled.[/dim]"
            )
            return

        if result == "edit":
            edit = questionary.select(
                "What would you like to edit?",
                choices=[
                    "Task",
                    "Dataset query",
                    "Model path",
                    "Class labels",
                    "Back",
                ],
                pointer="»",
                style=questionary.Style([
                    ("question", f"bold {HEADING}"),
                    ("pointer", f"bold {ACCENT}"),
                    ("highlighted", f"bold {ACCENT}"),
                ]),
            ).ask()

            if edit is None or edit == "Back":
                continue

            if edit == "Task":
                new_task = select_task()

                if new_task is None:
                    continue

                config["task"] = new_task
                config["labels"] = []

                if new_task != "Image Captioning":
                    new_labels = get_labels(new_task)

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
                new_labels = get_labels(config["task"])

                if new_labels is not None:
                    config["labels"] = new_labels

            continue

        # STEP 6: Launch dashboard
        launch_dashboard(result)
        break


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        console.print(
            "\n[dim]OmniCompiler setup cancelled.[/dim]"
        )

    except Exception as e:
        console.print(
            f"\n[bold red]Unexpected error:[/bold red] {e}"
        )