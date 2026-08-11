#!/usr/bin/env python3
"""Build the panel-(d) evolution-trajectory result file from MEASURED library data.

Panel (d) shows one experimentally-measured evolved lineage from the panel-(c) library
across its available steps. The Jores 2026 evolution library only measured each lineage
at two mutation depths (6 and 12 single-base edits); together with the original ACR
(0 edits) that gives a 3-point trajectory. For the chosen lineage this records, at each
depth, the measured activity plus the AlphaGenome and plantGREP predictions.

Only measured library sequences are used — none of our own in-silico AG designs.

Predictions are computed here (AlphaGenome via the repo's JoresMPRADataset pipeline, so
adapters/one-hot match training; plantGREP via the released package) and verified against
the committed round-6/12 predictions before use. Writes
results/jores_multicondition/reference/evolution_trajectory_measured.csv.

Run with the alphagenome-ft-jores26 environment (torch + pytorch_lightning), on a GPU:
    <jores26>/.venv/bin/python scripts/make_jores_evolution_trajectory.py \\
        --plantgrep_package <plantGREP_package.pt> \\
        --checkpoint /grid/koo/home/shared/models/alphagenome_encoder/torch/jores_multicondition/finetuned_encoder.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REF = REPO_ROOT / "results" / "jores_multicondition" / "reference"
JORES = Path("/grid/koo/home/amurphy/projects/alphagenome-ft-jores26")
MAIN_TSV = "/grid/koo/home/shared/data/plant_acr/jores_2026/modelling_data_tamsACR.tsv"
CONDITIONS = ["cold", "dark", "light", "warm", "maize"]   # AG target order
PG_ORDER = ["light", "dark", "warm", "cold", "maize"]     # plantGREP output order
WARM = "warm"
BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def gather_lineage(seq_id, objective):
    """The 3 measured points (0, 6, 12 mutations) for one lineage, from the libraries."""
    import re
    main = {r["id"]: r for r in csv.DictReader(open(MAIN_TSV), delimiter="\t")}
    evo = list(csv.DictReader(open(JORES / "metadata/evolution_only_eval.tsv"), delimiter="\t"))

    def parse(e):
        o = re.search(r"objective = ([^,]+)", e); rd = re.search(r"round = (\d+)", e)
        m = re.search(r"mutations/round = (\d+)", e)
        return (o.group(1).strip() if o else None,
                int(rd.group(1)) if rd else None, int(m.group(1)) if m else None)

    pts = {0: (main[seq_id], None)}
    for i, r in enumerate(evo):
        o, rd, m = parse(r["experiment"])
        if r["id"] == seq_id and o == objective and m == 1 and rd in (6, 12):
            pts[rd] = (r, i)  # keep the committed-prediction row index for the check
    assert set(pts) == {0, 6, 12}, f"missing depths for {seq_id}/{objective}: {sorted(pts)}"
    return pts


def ag_predict(checkpoint, rows):
    """Warm prediction per row, via the repo's JoresMPRADataset (adapters + one-hot)."""
    sys.path.insert(0, str(JORES / "src"))
    import torch
    from alphagenome_encoder_ft import AlphaGenomeEncoderModel
    from alphagenome_encoder_ft.mydata import JoresMPRADataset

    model = AlphaGenomeEncoderModel.from_checkpoint(checkpoint, device="cuda")
    ds = JoresMPRADataset(rows, use_adapters=True, reverse_complement=False,
                          random_shift=False)
    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            oh, _ = ds[i]
            pred = model(oh.unsqueeze(0).to("cuda")).squeeze(0).cpu().numpy()  # (5,) AG order
            out.append(float(pred[CONDITIONS.index(WARM)]))
    return out


def pg_predict(package, seqs):
    """Warm prediction per bare 170bp insert, via the released plantGREP package."""
    import torch
    model = torch.package.PackageImporter(package).load_pickle("plantGREP", "model.pkl")
    model.eval()
    oh = np.zeros((len(seqs), 4, len(seqs[0])), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s.upper()):
            if c in BASE_IDX:
                oh[i, BASE_IDX[c], j] = 1.0
    with torch.no_grad():
        p = model(torch.tensor(oh)).cpu().numpy()   # (N,5) plantGREP order
    return [float(p[i, PG_ORDER.index(WARM)]) for i in range(len(seqs))]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    # comma-separated id:objective lineages, drawn as stacked panels in the order given
    p.add_argument("--lineages", default="At-52107_rev:warm,Sb-27124_rev:warm")
    p.add_argument("--plantgrep_package", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=str(REF / "evolution_trajectory_measured.csv"))
    args = p.parse_args()

    depths = [0, 6, 12]
    ag_ref = list(csv.DictReader(open(JORES / "models/jores_multicondition/analysis/stage2/"
                                       "evaluation/evolution_only_eval/test_predictions.csv")))
    pg_ref = list(csv.DictReader(open(JORES / "models/plantGREP/plantGREP_evolution_only/"
                                       "test_predictions.csv")))

    out_rows = []
    for spec in args.lineages.split(","):
        seq_id, objective = spec.split(":")
        pts = gather_lineage(seq_id, objective)
        rows = [pts[d][0] for d in depths]   # raw rows carry sequence + enrichment_*
        seqs = [pts[d][0]["sequence"] for d in depths]
        ag = ag_predict(args.checkpoint, rows)
        pg = pg_predict(args.plantgrep_package, seqs)
        for k, d in enumerate(depths):       # verify 6/12 against committed predictions
            idx = pts[d][1]
            if idx is None:
                continue
            da = abs(ag[k] - float(ag_ref[idx]["warm_pred"]))
            dp = abs(pg[k] - float(pg_ref[idx]["warm_pred"]))
            assert da < 5e-2 and dp < 1e-2, f"{seq_id} depth {d} does not match committed"
        for k, d in enumerate(depths):
            out_rows.append({"lineage": seq_id, "objective": objective, "mutations": d,
                             "warm_measured": float(pts[d][0]["enrichment_warm"]),
                             "warm_ag": ag[k], "warm_pg": pg[k]})
        print(f"{seq_id}: measured {[round(r['warm_measured'],1) for r in out_rows[-3:]]} "
              f"AG {[round(r['warm_ag'],1) for r in out_rows[-3:]]} "
              f"pG {[round(r['warm_pg'],1) for r in out_rows[-3:]]}")

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lineage", "objective", "mutations",
                                          "warm_measured", "warm_ag", "warm_pg"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
