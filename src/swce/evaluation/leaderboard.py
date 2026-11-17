"""Leaderboard of published results.

The CSV file is the source. It holds one row per model, method setting, task and metric, with the
bootstrap interval of the metric and its paired difference from the unmodified model. The
Markdown summary is rewritten from it and never edited by hand.
"""

from pathlib import Path

import pandas as pd

from swce.registry import Registry, load_registry

KEYS = ["model", "method", "fingerprint", "task", "task_fingerprint", "metric"]
"""Columns that identify a row. A new row replaces an earlier one with the same values."""
COLUMNS = [
    "model",
    "method",
    "setting",
    "fingerprint",
    "fitted_on",
    "task",
    "task_fingerprint",
    "metric",
    "primary",
    "value",
    "low",
    "high",
    "delta",
    "delta_low",
    "delta_high",
    "p_mcnemar",
    "fixed",
    "broken",
    "items",
]
"""Columns of the CSV file, in order."""


def _cell(value: float, low: float, high: float) -> str:
    """Value with its interval, or a dash when it is undefined."""
    if pd.isna(value):
        return "n/a"
    if pd.isna(low) or pd.isna(high):
        return f"{value:.4g}"
    return f"{value:.4g} [{low:.4g}, {high:.4g}]"


class Leaderboard:
    """Results kept in a CSV file next to their Markdown summary.

    Attributes:
        path: CSV file holding the rows.
    """

    def __init__(self, path: Path, registry: Registry | None = None):
        self.path = path
        self._registry = registry or load_registry()

    def rows(self) -> pd.DataFrame:
        """Rows of the leaderboard.

        Returns:
            The rows of the CSV file, or an empty frame with the same columns when the file does
            not exist.
        """
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.path, dtype={"fingerprint": str, "task_fingerprint": str})

    def add(self, rows: pd.DataFrame) -> None:
        """Add rows, replacing earlier rows of the same model, method setting, task and metric.

        Args:
            rows: Rows with the columns of `COLUMNS`.
        """
        if rows.empty:
            return
        current = self.rows()
        merged = rows[COLUMNS] if current.empty else pd.concat([current, rows[COLUMNS]])
        merged = merged.drop_duplicates(KEYS, keep="last")
        merged = merged.sort_values(KEYS, kind="stable").reset_index(drop=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(self.path, index=False, lineterminator="\n")
        self.refresh()

    def current(self) -> pd.DataFrame:
        """Rows obtained under the task definitions of the registry in use.

        Returns:
            The rows whose task fingerprint equals the present one. Rows of tasks whose
            definition changed since stay in the file and are left out here.
        """
        from swce.tasks.task import Task

        rows = self.rows()
        if rows.empty:
            return rows
        known = set(self._registry.task_keys())
        present = {key: Task(key, registry=self._registry).fingerprint for key in known}
        keep = [
            present.get(task) == mark
            for task, mark in zip(rows.task, rows.task_fingerprint, strict=True)
        ]
        return rows[keep]

    def summary(self) -> str:
        """Markdown summary of the primary metric of every task.

        Returns:
            One table per model, with one row per method setting and one column per task, each
            cell holding the primary metric with its interval.
        """
        lines = [
            "# Leaderboard",
            "",
            "Generated from `leaderboard.csv`. Do not edit by hand. Cells hold the primary metric",
            "of the task with its 95% bootstrap interval over images.",
            "",
        ]
        rows = self.current()
        rows = rows[rows.primary.astype(bool)] if not rows.empty else rows
        if rows.empty:
            return "\n".join([*lines, "No published result yet.", ""])
        for model, of_model in rows.groupby("model", sort=True):
            tasks = [key for key in self._registry.task_keys() if key in set(of_model.task)]
            lines += [f"## {model}", ""]
            lines += ["| Method | Setting | Fitted on | " + " | ".join(tasks) + " |"]
            lines += ["|---|---|---|" + "---|" * len(tasks)]
            for (method, _mark), group in of_model.groupby(["method", "fingerprint"], sort=True):
                first = group.iloc[0]
                cells = []
                for task in tasks:
                    match = group[group.task == task]
                    cells.append(
                        "n/a"
                        if match.empty
                        else _cell(match.iloc[0].value, match.iloc[0].low, match.iloc[0].high)
                    )
                lines += [
                    f"| {method} | {first.setting} | {first.fitted_on} | "
                    + " | ".join(cells)
                    + " |"
                ]
            metrics = {task: of_model[of_model.task == task].metric.iloc[0] for task in tasks}
            lines += ["", "Metrics: " + ", ".join(f"`{t}` {m}" for t, m in metrics.items()), ""]
        return "\n".join(lines).rstrip("\n") + "\n"

    def refresh(self) -> None:
        """Rewrite the Markdown summary next to the CSV file."""
        target = self.path.with_suffix(".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.summary(), encoding="utf-8", newline="\n")
