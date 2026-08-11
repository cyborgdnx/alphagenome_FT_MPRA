"""Evaluate a fine-tuned Jores 2026 multi-condition AlphaGenome-encoder checkpoint.

Three modes (the three evals behind the benchmark figure):

  --mode test          Held-out test split of the training TSV. Per-condition
                       mse/rmse/mae/pearson/spearman + means.   (panel a)
  --mode by_objective  On-target design eval: each condition scored only on the evolved
                       sequences that targeted it. Needs an `experiment` column.  (panel c)
  --mode categories    Design-validation library: evolution on-/off-target and the TF
                       perturbation groups (TFBS insertion / shuffling).  (panel b)

PyTorch path; runs in the dedicated jores env (requirements/jores_multicondition.txt).
Loads via the standard alphagenome_encoder_ft (Nagai) package; the checkpoint's head is
MPRAHead(num_outputs=5).

USAGE:
    python scripts/test_jores_multicondition.py --mode test \\
        --checkpoint <finetuned_encoder.pt> \\
        --input_tsv /grid/koo/home/shared/data/plant_acr/jores_2026/modelling_data_tamsACR.tsv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from alphagenome_encoder_ft import AlphaGenomeEncoderModel
from alphagenome_ft_mpra.jores_multicondition_data import (
    CONDITIONS, JoresMPRADataset, create_dataloader, read_jores_tsv,
)
from alphagenome_ft_mpra.jores_multicondition_objectives import (
    compute_pearsonr, compute_targeted_metrics, target_conditions_for_experiment,
    total_evolution_rounds,
)


# ---- prediction ------------------------------------------------------------
def collect_predictions(model, loader, device, use_amp=True):
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
            if use_amp and torch.device(device).type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(sequences, organism_idx)
            else:
                out = model(sequences, organism_idx)
            preds.append(out.float().cpu().numpy())
            trues.append(targets.numpy())
    return np.concatenate(trues).reshape(-1, len(CONDITIONS)), np.concatenate(preds).reshape(-1, len(CONDITIONS))


def _test_rows(input_tsv):
    return [r for r in read_jores_tsv(input_tsv) if r["set"] == "test"]


def _test_loader(model_seq_len, rows, batch_size):
    ds = JoresMPRADataset(rows, use_adapters=True, sequence_length=model_seq_len,
                          reverse_complement=False, random_shift=False)
    return create_dataloader(ds, batch_size, shuffle=False)


# ---- metrics ---------------------------------------------------------------
def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation) if a.size >= 2 else float("nan")


def per_condition_metrics(y_true, y_pred, masks=None):
    """Per-condition regression metrics (+ means), optionally restricted to a per-
    condition boolean mask."""
    metrics = {"n_samples": int(y_true.shape[0]), "per_condition": {}}
    for i, name in enumerate(CONDITIONS):
        m = masks[name] if masks else np.ones(y_true.shape[0], bool)
        t, p = y_true[m, i], y_pred[m, i]
        res = p - t
        mse = float(np.mean(res ** 2)) if t.size else float("nan")
        metrics["per_condition"][name] = {
            "n_samples": int(m.sum()), "mse": mse,
            "rmse": float(math.sqrt(mse)) if not math.isnan(mse) else float("nan"),
            "mae": float(np.mean(np.abs(res))) if t.size else float("nan"),
            "pearsonr": compute_pearsonr(t, p),
            "spearmanr": _spearman(t, p),
        }
    for stat in ("mse", "rmse", "mae", "pearsonr", "spearmanr"):
        metrics[f"mean_{stat}"] = float(np.nanmean(
            [metrics["per_condition"][n][stat] for n in CONDITIONS]))
    return metrics


# ---- categories (panel b / evolution on-off-target) ------------------------
import re
_TFBS_INSERTION_RE = re.compile(r"^TFBS insertion \(([^)]*)\)")


def classify_row(experiment):
    e = experiment.strip()
    if e.startswith("evolution:"):
        return {"category": "evolution",
                "target_conditions": target_conditions_for_experiment(e),
                "perturbation": ""}
    if e.startswith("TFBS shuffling"):
        return {"category": "tfbs_shuffling", "target_conditions": (), "perturbation": "shuffling"}
    m = _TFBS_INSERTION_RE.match(e)
    if m is not None:
        n = sum(1 for s in m.group(1).split(",") if s.strip() != "none")
        return {"category": "tfbs_insertion", "target_conditions": (), "perturbation": f"insertion_{n}"}
    if e in ("unmodified control", "validation of ACR sequence library"):
        return {"category": "other", "target_conditions": (), "perturbation": ""}
    raise ValueError(f"Unrecognized experiment: {experiment!r}")


def categories_metrics(rows, y_true, y_pred):
    meta = [classify_row(r["experiment"]) for r in rows]
    tconds = [m["target_conditions"] if m["category"] == "evolution" else None for m in meta]
    perturb = np.array([m["perturbation"] for m in meta])

    def uniform(mask):
        return per_condition_metrics(y_true, y_pred, {n: mask for n in CONDITIONS})

    def evo(mode):
        masks = {n: np.array([tc is not None and ((n in tc) == (mode == "on"))
                              for tc in tconds]) for n in CONDITIONS}
        return per_condition_metrics(y_true, y_pred, masks)

    out = {"n_samples": int(y_true.shape[0]),
           "evolution_on_target": evo("on"), "evolution_off_target": evo("off"),
           "perturbation": {}}
    groups = {"insertion_1": perturb == "insertion_1", "insertion_2": perturb == "insertion_2",
              "insertion_3": perturb == "insertion_3", "shuffling": perturb == "shuffling"}
    groups["insertion"] = np.isin(perturb, ["insertion_1", "insertion_2", "insertion_3"])
    groups["combined"] = groups["insertion"] | groups["shuffling"]
    for name, mask in groups.items():
        if mask.any():
            out["perturbation"][name] = uniform(mask)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--mode", choices=["test", "by_objective", "categories"], default="test")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_tsv", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = AlphaGenomeEncoderModel.from_checkpoint(args.checkpoint, device=args.device)
    seq_len = getattr(model.head, "sequence_length", None) or 170
    rows = _test_rows(args.input_tsv)
    loader = _test_loader(seq_len, rows, args.batch_size)
    y_true, y_pred = collect_predictions(model, loader, args.device)
    print(f"scored {len(rows)} test rows")

    if args.mode == "test":
        metrics = per_condition_metrics(y_true, y_pred)
    elif args.mode == "by_objective":
        tconds = [target_conditions_for_experiment(r["experiment"]) for r in rows]
        metrics = compute_targeted_metrics(y_true, y_pred, tconds, condition_order=tuple(CONDITIONS))
    else:
        metrics = categories_metrics(rows, y_true, y_pred)

    metrics["checkpoint"] = str(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / f"{args.mode}_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    key = "mean_pearsonr" if "mean_pearsonr" in metrics else "n_samples"
    print(f"{args.mode}: {key} = {metrics.get(key)}")
    print(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    sys.exit(main())
