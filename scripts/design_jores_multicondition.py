"""Zero-shot design: in-silico directed evolution of a plant promoter toward a target
warm-condition activity, using a Jores multi-condition AlphaGenome checkpoint.

Greedy single-base evolution — each round runs saturation mutagenesis on the current
insert, scores every substitution under the loss below, and accepts the single mutation
that most reduces it, until no mutation helps (converged) or a round cap is hit:

    loss = max(target_warm - pred_warm, 0)^2
           + other_condition_weight * sum_{cold,dark,light,maize} (pred_c - baseline_c)^2

i.e. push warm up to the target while holding the other four conditions near their
round-0 levels. Writes ``<id>_history.tsv`` (per-round predicted levels + the mutation
applied) and ``<id>_summary.json`` (initial/final sequence, stop reason) — the same
trajectory format the benchmark figure's panel (d) reads.

Optional Taskiran et al. 2024 branching (``--random_topk_steps``): the first N rounds
sample uniformly among the top-k lowest-loss mutations and apply them unconditionally,
to build diverse independent paths.

PyTorch path; dedicated jores env. Adapters are fixed library sequence and never mutated.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from alphagenome_pytorch.utils.sequence import sequence_to_onehot, onehot_tensor_to_sequence

from alphagenome_encoder_ft import AlphaGenomeEncoderModel
from alphagenome_ft_mpra.jores_multicondition_data import (
    CONDITIONS, JORES_ADAPTER_UP, JORES_ADAPTER_DOWN, read_jores_tsv,
)
from perturbation_jores_multicondition import CONDITION, TangermemeWrapper

BASES = "ACGT"


def evolve_sequence(wrapped, seq_id, insert_seq, left_adapter, right_adapter, *,
                    target_warm, max_iterations=100, other_condition_weight=0.5,
                    ism_batch_size=64, random_topk_steps=0, random_topk=20, rng=None):
    """One greedy evolution path for one insert. Returns a JSON-able result dict with
    the full per-round history and the final evolved sequence."""
    from tangermeme.saturation_mutagenesis import saturation_mutagenesis

    construct = f"{left_adapter}{insert_seq}{right_adapter}"
    X = torch.from_numpy(sequence_to_onehot(construct).astype("float32")).transpose(0, 1).unsqueeze(0)
    organism_idx = torch.zeros(1, dtype=torch.long)
    start, end = len(left_adapter), len(left_adapter) + len(insert_seq)
    warm_idx = CONDITION["warm"]
    other_idxs = [CONDITION[c] for c in CONDITIONS if c != "warm"]
    baseline_other = None

    def loss_of(pred):
        warm_term = torch.clamp(target_warm - pred[..., warm_idx], min=0.0) ** 2
        other_term = ((pred[..., other_idxs] - baseline_other) ** 2).sum(dim=-1)
        return warm_term + other_condition_weight * other_term

    history, round_idx, stop_reason, target_reached = [], 0, "max_iterations", None
    t0 = time.time()
    while True:
        y0, y_hat = saturation_mutagenesis(wrapped, X, args=(organism_idx,), start=start,
                                           end=end, raw_outputs=True, batch_size=ism_batch_size)
        y0, y_hat = y0[0], y_hat[0]                    # (5,), (4, W, 5)
        if baseline_other is None:
            baseline_other = y0[other_idxs].clone()
        current_loss = loss_of(y0).item()
        levels = {c: float(y0[CONDITION[c]].item()) for c in CONDITIONS}
        history.append({"round": round_idx, **levels, "loss": current_loss,
                        "mutation": None, "phase": None})
        if target_reached is None and levels["warm"] >= target_warm:
            target_reached = round_idx
        if round_idx >= max_iterations:
            break

        losses = loss_of(y_hat)                        # (4, W)
        if round_idx < random_topk_steps:
            cur = X[0, :, start:end].argmax(dim=0)
            is_id = torch.arange(4).unsqueeze(1) == cur.unsqueeze(0)
            masked = losses.masked_fill(is_id, float("inf")).reshape(-1)
            k = min(random_topk, masked.numel() - int(is_id.sum()))
            _, topk = torch.topk(masked, k=k, largest=False)
            flat_idx = int(topk[int(torch.randint(len(topk), (1,), generator=rng))])
            phase = "random_topk"
        else:
            flat_idx = int(torch.argmin(losses))
            phase = "greedy"

        best_char, best_pos_rel = divmod(flat_idx, losses.shape[1])
        if phase == "greedy" and losses.reshape(-1)[flat_idx].item() >= current_loss - 1e-9:
            stop_reason = "converged"
            break
        best_pos = best_pos_rel + start
        ref_char = int(X[0, :, best_pos].argmax())
        X[0, :, best_pos] = 0.0
        X[0, best_char, best_pos] = 1.0
        history[-1]["mutation"] = f"{BASES[ref_char]}{best_pos_rel}{BASES[best_char]}"
        history[-1]["phase"] = phase
        round_idx += 1

    final_construct = onehot_tensor_to_sequence(X[0].transpose(0, 1))
    print(f"[{seq_id}] {round_idx} rounds: {stop_reason} in {time.time()-t0:.1f}s",
          file=sys.stderr)
    return {
        "id": seq_id, "target_warm": target_warm,
        "other_condition_weight": other_condition_weight, "max_iterations": max_iterations,
        "n_rounds": round_idx, "stop_reason": stop_reason,
        "target_reached_round": target_reached, "history": history,
        "initial_sequence": insert_seq, "final_sequence": final_construct[start:end],
        "initial_levels": history[0], "final_levels": history[-1],
    }


def save_result(out_dir, result):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result["id"].replace("/", "_")
    with open(out_dir / f"{stem}_history.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", *CONDITIONS, "loss", "mutation", "phase"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(result["history"])
    summary = {k: v for k, v in result.items() if k != "history"}
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_tsv", required=True)
    ap.add_argument("--sequence_ids", nargs="+", required=True,
                    help="id(s) from the input TSV to evolve.")
    ap.add_argument("--target_warm", type=float, required=True)
    ap.add_argument("--other_condition_weight", type=float, default=0.5)
    ap.add_argument("--max_iterations", type=int, default=100)
    ap.add_argument("--no_adapters", action="store_true",
                    help="Evolve the bare insert (adapters are used by default).")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--ism_batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    by_id = {r["id"]: r for r in read_jores_tsv(args.input_tsv)}
    model = AlphaGenomeEncoderModel.from_checkpoint(args.checkpoint, device=args.device)
    wrapped = TangermemeWrapper(model).to(args.device).eval()
    left = "" if args.no_adapters else JORES_ADAPTER_UP
    right = "" if args.no_adapters else JORES_ADAPTER_DOWN
    out_dir = Path(args.output_dir) if args.output_dir else \
        Path(str(args.checkpoint).replace(".pt", "") + "_design")

    for seq_id in args.sequence_ids:
        if seq_id not in by_id:
            ap.error(f"sequence id {seq_id!r} not found in {args.input_tsv}")
        result = evolve_sequence(
            wrapped, seq_id, by_id[seq_id]["sequence"], left, right,
            target_warm=args.target_warm, max_iterations=args.max_iterations,
            other_condition_weight=args.other_condition_weight,
            ism_batch_size=args.ism_batch_size)
        save_result(out_dir, result)
        print(f"{seq_id}: warm {result['initial_levels']['warm']:.2f} -> "
              f"{result['final_levels']['warm']:.2f} in {result['n_rounds']} rounds "
              f"({result['stop_reason']})")
    print(f"Wrote design trajectories -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
