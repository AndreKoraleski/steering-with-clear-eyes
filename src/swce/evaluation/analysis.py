"""Reading a gain against what it costs, from the leaderboard alone.

A method is never given one number. Its gain on a hallucination task and its change on the
capability benchmark are kept apart and read in four steps.

1. The gain is the paired difference from the unmodified model, signed so that positive is
   better, with its interval. Its McNemar test is corrected over every method and task of a model
   with the procedure of Holm (1979).
2. The capability cost is tested for non inferiority against a margin fixed in advance. Finding
   no significant loss is not evidence of no loss on a benchmark as small as MME.
3. The gain is compared with the gain of the control of the method, when one was evaluated.
4. Two settings are ordered only when one is at least as good on both axes and better on one.

The capability change is read on the MME subtasks that do not overlap with object hallucination,
since a gain on the others would otherwise count as preserved capability.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from swce.registry import Registry

MARGIN = 0.02
"""Default non inferiority margin, as a fraction of the capability score of the unmodified model."""
CAPABILITY_TASK = "mme"
"""Task measuring general capability."""
CAPABILITY_METRIC = "other"
"""Metric of that task read as capability, the total without the hallucination subtasks."""
CONTROL_SUFFIX = "-random"
"""Suffix of the name of the control of a method."""
COLUMNS = [
    "model",
    "method",
    "setting",
    "fingerprint",
    "task",
    "metric",
    "gain",
    "gain_low",
    "gain_high",
    "p_mcnemar",
    "p_holm",
    "capability",
    "capability_low",
    "capability_high",
    "capability_relative",
    "non_inferior",
    "control_gain",
    "beyond_control",
    "dominated_by",
]
"""Columns of the trade off table, in order."""


def holm(pvalues: Sequence[float]) -> list[float]:
    """P values adjusted for multiple comparisons with the step down procedure of Holm.

    Args:
        pvalues: Raw p values. NaN marks a comparison that was not tested.

    Returns:
        The adjusted p values, in the same order, NaN where the raw value is NaN.
    """
    raw = np.array(pvalues, dtype=np.float64)
    adjusted = np.full(raw.shape, np.nan)
    tested = np.flatnonzero(~np.isnan(raw))
    order = tested[np.argsort(raw[tested], kind="stable")]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (order.size - rank) * raw[index]))
        adjusted[index] = running
    return [float(value) for value in adjusted]


def trade_offs(rows: pd.DataFrame, registry: Registry, margin: float = MARGIN) -> pd.DataFrame:
    """Gain of every method setting on every hallucination task next to its capability change.

    Args:
        rows: Rows of the leaderboard under the task definitions in use.
        registry: Registry defining the tasks.
        margin: Non inferiority margin, as a fraction of the capability score of the unmodified
            model.

    Returns:
        One row per model, method setting and hallucination task with the `gain` and its
        interval, `p_mcnemar` and `p_holm`, the capability change `capability` with its interval
        and its value `capability_relative` relative to the unmodified model, `non_inferior`,
        whether the whole interval of that change lies above minus the margin, the gain of the
        control as `control_gain` and the difference `beyond_control`, and `dominated_by`, the
        settings that are at least as good on both axes and better on one. Empty when the
        leaderboard holds no method.
    """
    nan = float("nan")
    higher = {
        task.key: task.higher_is_better for card in registry.benchmarks for task in card.tasks
    }
    found = rows.to_dict("records") if not rows.empty else []
    records: list[dict[str, Any]] = [
        {str(key): value for key, value in row.items()} for row in found
    ]
    steered = [row for row in records if not math.isnan(float(row["delta"]))]
    costs = {
        (row["model"], row["method"], row["fingerprint"]): row
        for row in steered
        if row["task"] == CAPABILITY_TASK and row["metric"] == CAPABILITY_METRIC
    }
    table: list[dict[str, Any]] = []
    for row in steered:
        if not bool(row["primary"]) or row["task"] == CAPABILITY_TASK:
            continue
        sign = 1.0 if higher.get(row["task"], True) else -1.0
        low, high = sorted((sign * float(row["delta_low"]), sign * float(row["delta_high"])))
        entry: dict[str, Any] = {
            "model": row["model"],
            "method": row["method"],
            "setting": row["setting"],
            "fingerprint": row["fingerprint"],
            "task": row["task"],
            "metric": row["metric"],
            "gain": sign * float(row["delta"]),
            "gain_low": low,
            "gain_high": high,
            "p_mcnemar": float(row["p_mcnemar"]),
            "capability": nan,
            "capability_low": nan,
            "capability_high": nan,
            "capability_relative": nan,
            "non_inferior": None,
        }
        cost = costs.get((row["model"], row["method"], row["fingerprint"]))
        if cost is not None:
            change, bound = float(cost["delta"]), float(cost["delta_low"])
            base = float(cost["value"]) - change
            entry.update(
                capability=change,
                capability_low=bound,
                capability_high=float(cost["delta_high"]),
                capability_relative=change / base if base else nan,
                non_inferior=None if math.isnan(bound) or not base else bound > -margin * base,
            )
        table.append(entry)
    for model in {entry["model"] for entry in table}:
        group = [entry for entry in table if entry["model"] == model]
        for entry, adjusted in zip(group, holm([e["p_mcnemar"] for e in group]), strict=True):
            entry["p_holm"] = adjusted
    controls = {
        (e["model"], e["method"].removesuffix(CONTROL_SUFFIX), e["setting"], e["task"]): e["gain"]
        for e in table
        if e["method"].endswith(CONTROL_SUFFIX)
    }
    for entry in table:
        key = (entry["model"], entry["method"], entry["setting"], entry["task"])
        entry["control_gain"] = controls.get(key, nan)
        entry["beyond_control"] = entry["gain"] - entry["control_gain"]
        rivals = [
            other
            for other in table
            if other is not entry
            and (other["model"], other["task"]) == (entry["model"], entry["task"])
            and not math.isnan(other["capability"])
            and not math.isnan(entry["capability"])
            and other["gain"] >= entry["gain"]
            and other["capability"] >= entry["capability"]
            and (other["gain"] > entry["gain"] or other["capability"] > entry["capability"])
        ]
        entry["dominated_by"] = ", ".join(f"{o['method']} {o['setting']}" for o in rivals)
    return pd.DataFrame(table, columns=COLUMNS)
