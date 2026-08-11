"""PyTorch data pipeline for the Jores 2026 plant multi-condition MPRA.

Ported verbatim from alphagenome-ft-jores26 (``src/alphagenome_encoder_ft/mydata.py``)
so this repo can reproduce the training/testing without that project's package. Torch
only — the model, head and two-stage trainer come from the standard
``alphagenome_encoder_ft`` package (Nagai's); the head is ``MPRAHead(num_outputs=5)``.

Each example is one 170 bp core-promoter insert flanked by the 15 bp Jores library
adapters, one-hot encoded (via ``alphagenome_pytorch``), with a 5-vector target in the
fixed order ``[cold, dark, light, warm, maize]``. Splits come from the source TSV's own
``set`` column; augmentation is train-only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from alphagenome_pytorch.utils.sequence import sequence_to_onehot

JORES_ADAPTER_UP = "GAGCGGTCTCCACTC"
JORES_ADAPTER_DOWN = "CTGTAGAGACCGGGC"
CONDITIONS = ["cold", "dark", "light", "warm", "maize"]

# id-prefix -> species (the modelling TSV has no dedicated species column).
JORES_SPECIES_PREFIXES = {
    "Arabidopsis": ("At-",),
    "Maize": ("Zm-",),
    "Sorghum": ("Sb-",),
    "Tomato": ("Sl-", "Solyc"),
}


def _reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    return onehot[::-1, :][:, [3, 2, 1, 0]]      # reverse sequence, then complement bases


def _to_float(value: str) -> float:
    # the TSV writes missing values as an empty cell, not the text 'NaN'
    return float(value) if value != "" else np.nan


class JoresMPRADataset(Dataset):
    """Returns ``(onehot (L, 4) float32, target (5,) float32)`` per row.

    ``rows`` are already-read dicts (e.g. from ``csv.DictReader``) carrying a
    ``sequence`` column and ``enrichment_{cold,dark,light,warm,maize}`` columns.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        use_adapters: bool = True,
        left_adapter: str = JORES_ADAPTER_UP,
        right_adapter: str = JORES_ADAPTER_DOWN,
        sequence_length: int | None = 170,
        reverse_complement: bool = False,
        rc_prob: float = 0.5,
        random_shift: bool = False,
        shift_prob: float = 0.5,
        max_shift: int = 15,
        seed: int = 42,
        reseed_per_epoch: bool = True,
    ) -> None:
        if sequence_length is not None and sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if not 0 <= rc_prob <= 1:
            raise ValueError("rc_prob must be in [0, 1]")
        if not 0 <= shift_prob <= 1:
            raise ValueError("shift_prob must be in [0, 1]")
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")

        self.use_adapters = bool(use_adapters)
        self.left_adapter = left_adapter if self.use_adapters else ""
        self.right_adapter = right_adapter if self.use_adapters else ""
        self.sequence_length = sequence_length
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self._base_seed = seed
        self._reseed_per_epoch = reseed_per_epoch
        self._rng = np.random.default_rng(seed)

        self._payloads = [str(row["sequence"]) for row in rows]
        self._targets = np.asarray(
            [[_to_float(row[f"enrichment_{c}"]) for c in CONDITIONS] for row in rows],
            dtype=np.float32,
        ).reshape(-1, len(CONDITIONS))     # (N, 5): [cold, dark, light, warm, maize]

        if np.isnan(self._targets).any():
            n = int(np.isnan(self._targets).any(axis=1).sum())
            raise ValueError(f"{n} rows have NaN activity")

    def __len__(self) -> int:
        return len(self._payloads)

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the augmentation RNG per epoch (deterministic across resumes)."""
        if self._reseed_per_epoch:
            self._rng = np.random.default_rng(self._base_seed + epoch)

    def _augment(self, onehot: np.ndarray) -> np.ndarray:
        out = onehot
        if self.reverse_complement and self._rng.random() < self.rc_prob:
            out = _reverse_complement_onehot(out)
        if self.random_shift and self.max_shift > 0 and self._rng.random() < self.shift_prob:
            shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
            out = np.roll(out, shift, axis=0)
        return out

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        construct = f"{self.left_adapter}{self._payloads[index]}{self.right_adapter}"
        onehot = sequence_to_onehot(construct).astype(np.float32, copy=False)
        onehot = self._augment(onehot)
        return torch.from_numpy(onehot), torch.from_numpy(self._targets[index])


def read_jores_tsv(path: str | Path) -> list[dict[str, str]]:
    """Read a Jores-style tab-separated file into a list of str-keyed row dicts."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def create_jores_splits(
    input_tsv: str | Path,
    seed: int = 42,
    subset_frac: float = 1.0,
    **dataset_kwargs,
) -> tuple[JoresMPRADataset, JoresMPRADataset, JoresMPRADataset]:
    """Train/val/test datasets, partitioned by the TSV's own ``set`` column.

    Augmentation (``dataset_kwargs``) is applied only to the train split. ``subset_frac``
    downsamples train+val without replacement; test is always kept full size.
    """
    if not 0 < subset_frac <= 1:
        raise ValueError("subset_frac must be in (0, 1]")

    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in read_jores_tsv(input_tsv):
        split_name = row["set"]
        if split_name not in rows_by_split:
            raise ValueError(f"Unexpected 'set' value {split_name!r} in {input_tsv}")
        rows_by_split[split_name].append(row)

    if subset_frac < 1.0:
        rng = np.random.default_rng(seed)
        for split_name in ("train", "val"):        # test excluded — kept full size
            rows = rows_by_split[split_name]
            if rows:
                n = max(1, int(round(len(rows) * subset_frac)))
                idx = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
                rows_by_split[split_name] = [rows[i] for i in idx]

    noaug = {**dataset_kwargs, "reverse_complement": False, "random_shift": False}
    return (
        JoresMPRADataset(rows_by_split["train"], seed=seed, **dataset_kwargs),
        JoresMPRADataset(rows_by_split["val"], seed=seed, **noaug),
        JoresMPRADataset(rows_by_split["test"], seed=seed, **noaug),
    )


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, *,
                      num_workers: int = 0, pin_memory: bool = True,
                      drop_last: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def build_species_masks(ids: list[str]) -> dict[str, np.ndarray]:
    """species name -> boolean mask aligned row-for-row to ``ids`` (with an "Other"
    bucket for ids matching no defined prefix)."""
    masks = {
        species: np.array([str(i).startswith(prefixes) for i in ids])
        for species, prefixes in JORES_SPECIES_PREFIXES.items()
    }
    masks["Other"] = ~np.any(list(masks.values()), axis=0)
    return masks
