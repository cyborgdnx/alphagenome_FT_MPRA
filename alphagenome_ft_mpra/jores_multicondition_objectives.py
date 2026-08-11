"""Parsing + scoped-metric utilities for the Jores 2026 evolution objectives.

Ported verbatim from alphagenome-ft-jores26
(``src/alphagenome_encoder_ft/evolution_objectives.py``).

Each evolution row's `experiment` column reads like
"evolution: round = 2|2, objective = light|cold-specific, mutations/round = 3|3" --
the sequence was carried through one or more sequential rounds of directed evolution,
each selecting for a specific condition (pipe-separated for multi-stage lineages).

Scoring every evolution row against all 5 Jores conditions (cold/dark/light/warm/maize)
dilutes the signal with conditions nobody actually selected the sequence on. The
functions here recover, per row, which condition(s) its evolution objective(s) actually
targeted, so evaluation can be scoped to just those.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

CONDITION_NAMES = ("cold", "dark", "light", "warm", "maize")

# Objective token (as it appears after "objective = ", split on "|") -> the condition
# column(s) it targets. "constitutive" was selected for uniformly high expression across
# every condition; "tobacco-specific" was selected for high expression across the tobacco
# conditions as a group (vs. maize), without targeting one of them in particular.
OBJECTIVE_TO_CONDITIONS: dict[str, tuple[str, ...]] = {
    "cold": ("cold",),
    "cold-specific": ("cold",),
    "dark": ("dark",),
    "dark-specific": ("dark",),
    "light": ("light",),
    "light-specific": ("light",),
    "warm": ("warm",),
    "warm-specific": ("warm",),
    "maize": ("maize",),
    "maize-specific": ("maize",),
    "constitutive": CONDITION_NAMES,
    "tobacco-specific": ("cold", "dark", "light", "warm"),
}

_OBJECTIVE_RE = re.compile(r"objective\s*=\s*([^,]+)")
_ROUND_RE = re.compile(r"round\s*=\s*([^,]+)")


def parse_objective_tokens(experiment: str) -> tuple[str, ...]:
    """Raw pipe-separated objective token(s) from an `experiment` string.

    E.g. "evolution: round = 2|2, objective = light|cold-specific, mutations/round = 3|3"
    -> ("light", "cold-specific"). Returns () for non-evolution experiment strings
    (no "objective = " substring).
    """
    match = _OBJECTIVE_RE.search(experiment)
    if match is None:
        return ()
    return tuple(token.strip() for token in match.group(1).split("|"))


def parse_round_tokens(experiment: str) -> tuple[int, ...]:
    """Raw pipe-separated round count(s) from an `experiment` string.

    E.g. "evolution: round = 2|2, objective = light|cold-specific, mutations/round = 3|3"
    -> (2, 2). Returns () for non-evolution experiment strings (no "round = " substring).
    """
    match = _ROUND_RE.search(experiment)
    if match is None:
        return ()
    return tuple(int(token.strip()) for token in match.group(1).split("|"))


def total_evolution_rounds(experiment: str) -> int:
    """Total number of evolution rounds a sequence went through, across every stage.

    Multi-stage lineages (pipe-separated round counts) went through their stages
    sequentially -- see module docstring -- so the total rounds experienced is the sum
    of every stage's round count, not the max or the last value. Returns 0 for
    non-evolution experiment strings.
    """
    return sum(parse_round_tokens(experiment))


def target_conditions_for_experiment(experiment: str) -> tuple[str, ...]:
    """Union of condition columns targeted by every objective token in `experiment`.

    Multi-stage evolution (pipe-separated objectives) is treated as targeting the union
    of all stages' conditions, since the final sequence was selected under all of them
    in sequence. Raises on an unrecognized objective token -- see OBJECTIVE_TO_CONDITIONS
    for the known vocabulary -- rather than silently dropping it from scoring.
    """
    conditions: set[str] = set()
    for token in parse_objective_tokens(experiment):
        try:
            conditions.update(OBJECTIVE_TO_CONDITIONS[token])
        except KeyError:
            raise ValueError(
                f"Unrecognized evolution objective token {token!r} in experiment: {experiment!r}"
            ) from None
    return tuple(sorted(conditions, key=CONDITION_NAMES.index))


def load_target_conditions(input_tsv: str | Path, split: str = "test") -> list[tuple[str, ...]]:
    """Per-row target condition(s) for `split` rows of `input_tsv`, in the same
    original-file order create_jores_splits/JoresMPRADataset produce (filtered by the
    'set' column, no reordering) -- so this lines up row-for-row with a (y_true, y_pred)
    pair collected from that split with shuffle=False.

    Requires an `experiment` column in `input_tsv` (e.g. metadata/evolution_only_eval.tsv,
    not modelling_data_tamsACR.tsv, which has no such column).
    """
    with open(input_tsv, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row["set"] == split]
    return [target_conditions_for_experiment(row["experiment"]) for row in rows]


def compute_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size < 2:
        return float("nan")

    true_centered = y_true - y_true.mean()
    pred_centered = y_pred - y_pred.mean()
    denom = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(true_centered, pred_centered) / denom)


def compute_targeted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_conditions: list[tuple[str, ...]],
    condition_order: tuple[str, ...] = CONDITION_NAMES,
) -> dict[str, Any]:
    """Per-condition regression metrics, each restricted to the rows whose evolution
    objective(s) actually targeted that condition (see target_conditions_for_experiment).

    y_true/y_pred: (N, len(condition_order)) arrays, column-aligned to condition_order.
    target_conditions: length-N list of each row's target condition(s), e.g. from
    load_target_conditions -- must be in the same row order as y_true/y_pred.
    """
    if len(target_conditions) != y_true.shape[0]:
        raise ValueError(
            f"target_conditions length {len(target_conditions)} != y_true rows {y_true.shape[0]}"
        )

    metrics: dict[str, Any] = {"n_samples": int(y_true.shape[0])}
    per_condition: dict[str, dict[str, float]] = {}

    for i, name in enumerate(condition_order):
        mask = np.array([name in targets for targets in target_conditions])
        true_i = y_true[mask, i]
        pred_i = y_pred[mask, i]
        residual = pred_i - true_i
        mse = float(np.mean(np.square(residual))) if true_i.size else float("nan")
        per_condition[name] = {
            "n_samples": int(mask.sum()),
            "mse": mse,
            "rmse": float(math.sqrt(mse)) if not math.isnan(mse) else float("nan"),
            "mae": float(np.mean(np.abs(residual))) if true_i.size else float("nan"),
            "pearsonr": compute_pearsonr(true_i, pred_i),
        }
    metrics["per_condition"] = per_condition

    for stat in ("mse", "rmse", "mae", "pearsonr"):
        values = [per_condition[name][stat] for name in condition_order]
        metrics[f"mean_{stat}"] = float(np.mean(values))

    return metrics


def save_targeted_predictions(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_conditions: list[tuple[str, ...]],
    condition_order: tuple[str, ...] = CONDITION_NAMES,
) -> None:
    """Same CSV shape as evaluate_jores.py's save_predictions ("{condition}_true"/
    "{condition}_pred" columns per row), but leaves those cells blank for rows whose
    evolution objective(s) didn't target that condition.

    plot_jores_scatter.py treats a blank cell as missing and drops it before computing
    n / Pearson r / the scatter itself, so a plot built from this CSV matches
    compute_targeted_metrics' per-condition numbers -- unlike evaluate_jores.py's plain
    save_predictions, which fills in every condition for every row regardless of
    whether that row was ever selected on it.
    """
    if len(target_conditions) != y_true.shape[0]:
        raise ValueError(
            f"target_conditions length {len(target_conditions)} != y_true rows {y_true.shape[0]}"
        )

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["index"]
        for name in condition_order:
            header += [f"{name}_true", f"{name}_pred"]
        writer.writerow(header)
        for idx in range(y_true.shape[0]):
            row: list[Any] = [idx]
            targets = target_conditions[idx]
            for col, name in enumerate(condition_order):
                if name in targets:
                    row += [float(y_true[idx, col]), float(y_pred[idx, col])]
                else:
                    row += ["", ""]
            writer.writerow(row)
