"""Zero-shot perturbation: in-silico saturation mutagenesis for a Jores multi-condition
AlphaGenome checkpoint.

Scores every single-base substitution's effect on each condition's predicted activity
(via ``tangermeme.saturation_mutagenesis``) over the held-out test sequences, and caches
the raw result (``X``, ``y0`` reference preds, ``y_hat`` per-substitution preds). The
per-position, per-condition attribution ``y_hat - y0`` is the zero-shot variant-effect /
importance map; downstream motif discovery and sequence logos (in the alphagenome-ft-
jores26 repo) build on this cache.

PyTorch path; dedicated jores env (requirements/jores_multicondition.txt).

USAGE:
    python scripts/perturbation_jores_multicondition.py \\
        --checkpoint <finetuned_encoder.pt> \\
        --input_tsv /grid/koo/home/shared/data/plant_acr/jores_2026/modelling_data_tamsACR.tsv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from alphagenome_encoder_ft import AlphaGenomeEncoderModel
from alphagenome_ft_mpra.jores_multicondition_data import (
    CONDITIONS, JoresMPRADataset, read_jores_tsv,
)

CONDITION = {name: i for i, name in enumerate(CONDITIONS)}   # cold,dark,light,warm,maize


class TangermemeWrapper(torch.nn.Module):
    """Adapts the model's (N, L, 4) convention to the (N, 4, L) layout tangermeme's
    saturation_mutagenesis uses, so the model can be passed straight to tangermeme."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, organism_idx):
        return self.model(X.transpose(1, 2), organism_idx)


def process(y0, y_hat, X, hypothetical=True):
    """Per-position attribution: substitution prediction minus the reference, centered
    across the 4 bases. hypothetical keeps all 4 bases; else projects onto the observed."""
    attr = y_hat - y0[:, None, None]
    attr = attr - attr.mean(dim=1, keepdim=True)
    return attr if hypothetical else X * attr


def run_ism(wrapped, dataset, indices=None, batch_size=64):
    """Run saturation mutagenesis once over the dataset. Returns (X, y0, y_hat)."""
    from tangermeme.saturation_mutagenesis import saturation_mutagenesis

    indices = range(len(dataset)) if indices is None else indices
    X = torch.stack([dataset[i][0] for i in indices]).transpose(1, 2)   # (N, 4, L)
    organism_idx = torch.zeros(X.shape[0], dtype=torch.long)
    y0, y_hat = saturation_mutagenesis(wrapped, X, args=(organism_idx,),
                                       raw_outputs=True, batch_size=batch_size)
    return X, y0, y_hat


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_tsv", required=True)
    ap.add_argument("--cache_path", default=None,
                    help="Where to save the raw ISM result (default: <checkpoint>_ism_cache.pt).")
    ap.add_argument("--use_adapters", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_seqs", type=int, default=None, help="Cap sequences (debug).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = AlphaGenomeEncoderModel.from_checkpoint(args.checkpoint, device=args.device)
    wrapped = TangermemeWrapper(model).to(args.device).eval()

    rows = [r for r in read_jores_tsv(args.input_tsv) if r["set"] == "test"]
    dataset = JoresMPRADataset(rows, use_adapters=args.use_adapters, sequence_length=170,
                               reverse_complement=False, random_shift=False)
    idx = range(min(args.max_seqs, len(dataset))) if args.max_seqs else None
    print(f"running ISM over {len(idx) if idx else len(dataset)} test sequences...")

    X, y0, y_hat = run_ism(wrapped, dataset, indices=idx, batch_size=args.batch_size)
    cache = Path(args.cache_path) if args.cache_path else \
        Path(str(args.checkpoint).replace(".pt", "") + "_ism_cache.pt")
    torch.save({"X": X.cpu(), "y0": y0.cpu(), "y_hat": y_hat.cpu(),
                "conditions": CONDITIONS}, cache)

    # a compact summary: mean absolute per-condition effect magnitude
    attr = process(y0, y_hat, X)                        # (N, 4, W, 5)
    mag = attr.abs().mean(dim=(0, 1, 2)).cpu().numpy()  # per condition
    print("mean |ISM effect| per condition: " +
          ", ".join(f"{c}={mag[i]:.4f}" for i, c in enumerate(CONDITIONS)))
    print(f"Wrote ISM cache -> {cache}")


if __name__ == "__main__":
    sys.exit(main())
