"""Fine-tune the AlphaGenome encoder on the Jores 2026 plant multi-condition MPRA.

PyTorch path (backbone from alphagenome-pytorch). Two-stage training — stage 1 trains the
head on a frozen encoder, stage 2 unfreezes the encoder at a lower LR — via the standard
``alphagenome_encoder_ft`` package (Nagai's). The Jores head is just
``MPRAHead(num_outputs=5)`` (``head_type="mpra"``), predicting the 5 conditions
[cold, dark, light, warm, maize]; the jores-specific data pipeline is
``alphagenome_ft_mpra.jores_multicondition_data``.

Runs in the dedicated jores env (see requirements/jores_multicondition.txt,
docs/jores_multicondition.md), NOT the repo's default JAX env.

USAGE:
    python scripts/finetune_jores_multicondition.py \\
        --config configs/jores_multicondition.json \\
        --input_tsv /grid/koo/home/shared/data/plant_acr/jores_2026/modelling_data_tamsACR.tsv \\
        --pretrained_weights /grid/koo/home/shared/models/alphagenome/torch/model_all_folds.safetensors
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from alphagenome_encoder_ft import (
    AlphaGenomeEncoderModel,
    TrainConfig,
    create_optimizer,
    create_scheduler,
    evaluate,
    run_two_stage_training,
    run_training_stage,
    save_checkpoint,
)
from alphagenome_ft_mpra.jores_multicondition_data import (
    CONDITIONS,
    create_jores_splits,
    create_dataloader,
)


def _mean_condition_pearson(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean of the per-condition Pearson r (the metric the sweep selected on) — not the
    pooled/flattened correlation."""
    p = preds.detach().float().cpu().numpy().reshape(-1, len(CONDITIONS))
    t = targets.detach().float().cpu().numpy().reshape(-1, len(CONDITIONS))
    rs = []
    for i in range(len(CONDITIONS)):
        if p[:, i].std() > 0 and t[:, i].std() > 0:
            rs.append(float(np.corrcoef(p[:, i], t[:, i])[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")


def _load_config(path, overrides) -> TrainConfig:
    raw = json.loads(Path(path).read_text())
    # force the Jores head shape regardless of what the config file says
    raw.setdefault("head", {})
    raw["head"].update({"head_type": "mpra", "num_outputs": len(CONDITIONS)})
    for section, key, value in overrides:
        if value is not None:
            raw.setdefault(section, {})[key] = value
    return TrainConfig.from_dict(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_tsv", default=None)
    ap.add_argument("--pretrained_weights", required=True,
                    help="AlphaGenome all_folds torch weights (.safetensors).")
    ap.add_argument("--output_dir", default=None,
                    help="Overrides checkpoint.checkpoint_dir.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = _load_config(args.config, [
        ("data", "input_tsv", args.input_tsv),
        ("checkpoint", "checkpoint_dir", args.output_dir),
    ])
    if config.data.input_tsv is None:
        ap.error("data.input_tsv must be in the config or passed via --input_tsv")
    seq_len = config.data.sequence_length or 170

    torch.manual_seed(args.seed)
    print(f"Device: {args.device} | conditions: {CONDITIONS} | seq_len: {seq_len}")

    model = AlphaGenomeEncoderModel.from_pretrained(
        args.pretrained_weights, config.head, device=args.device)
    model.initialize_head(seq_len, args.device)

    train_ds, val_ds, test_ds = create_jores_splits(
        config.data.input_tsv, seed=args.seed, subset_frac=config.data.subset_frac,
        use_adapters=True, sequence_length=seq_len,
        reverse_complement=config.data.reverse_complement, rc_prob=config.data.rc_prob,
        random_shift=config.data.random_shift, shift_prob=config.data.shift_prob,
        max_shift=config.data.max_shift,
    )
    loaders = {
        split: create_dataloader(ds, config.data.batch_size, shuffle=(split == "train"),
                                 num_workers=config.data.num_workers,
                                 pin_memory=config.data.pin_memory)
        for split, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds))
    }
    print(f"rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    metric_fns = {"pearson": _mean_condition_pearson}
    checkpoint_dir = Path(config.checkpoint.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    stage1_optimizer = create_optimizer(
        config.optim, model.trainable_parameters(include_encoder=False))
    stage1_scheduler = create_scheduler(
        config.optim, stage1_optimizer, config.stage.num_epochs)

    if config.stage.second_stage_lr:
        def stage2_optimizer_factory(m):
            return create_optimizer(config.optim, m.trainable_parameters(include_encoder=True),
                                    learning_rate=config.stage.second_stage_lr)

        run_two_stage_training(
            model, loaders["train"], stage1_optimizer=stage1_optimizer,
            stage2_optimizer_factory=stage2_optimizer_factory, config=config,
            device=args.device, val_loader=loaders["val"],
            stage1_scheduler=stage1_scheduler, metric_fns=metric_fns, show_progress=True)
        stage = "stage2"
    else:
        run_training_stage(
            model, loaders["train"], optimizer=stage1_optimizer, config=config,
            device=args.device, num_epochs=config.stage.num_epochs, stage="stage1",
            train_encoder=False, val_loader=loaders["val"], scheduler=stage1_scheduler,
            metric_fns=metric_fns, checkpoint_dir=checkpoint_dir / "stage1",
            show_progress=True)
        stage = "stage1"

    test_metrics = evaluate(model, loaders["test"], device=args.device, metric_fns=metric_fns)
    (checkpoint_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"\nDone ({stage}). test pearson (mean over conditions) = "
          f"{test_metrics.get('pearson', float('nan')):.4f}")
    print(f"Checkpoints + metrics under {checkpoint_dir}")


if __name__ == "__main__":
    sys.exit(main())
