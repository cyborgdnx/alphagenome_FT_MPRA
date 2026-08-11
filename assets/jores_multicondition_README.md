# Jores 2026 plant multi-condition MPRA — AlphaGenome encoder checkpoints

AlphaGenome encoder fine-tuned on the Jores et al. 2026 plant MPRA, jointly predicting
activity in **5 conditions** — cold, dark, light, warm, maize — from a raw 170 bp core
promoter. Produced by [`alphagenome-ft-jores26`](https://github.com/katelynsyc/alphagenome-ft-jores26)
([PubMed 38513612](https://pubmed.ncbi.nlm.nih.gov/38513612/)).

Same PyTorch checkpoint format as the sibling folders (`save_mode="minimal"`; each `.pt`
carries `head_type`, `encoder_state_dict`, `head_state_dict`, `head_config`,
`construct_config`, `metrics`, `config`), with a `*.summary.json` sidecar per stage.

## ⚠️ Loads with a different package

Unlike the other `torch/` checkpoints — which use
[`alphagenome-encoder-ft`](https://github.com/MasayukiNagai/alphagenome-encoder-ft)
(class `EncoderMPRAModel`, head registry `{mpra, deepstarr}`) — this one uses the
[`alphagenome-ft-jores26`](https://github.com/katelynsyc/alphagenome-ft-jores26) fork
(class `AlphaGenomeEncoderModel`, registry `{mpra, joresmpra}`). Both use the import name
`alphagenome_encoder_ft`, so **they cannot be installed together** — load this from its
own environment.

## Checkpoints

`frozen_encoder.pt` = encoder frozen, head trained; `finetuned_encoder.pt` = encoder
unfrozen. Held-out test Pearson r:

| Condition | frozen | fine-tuned |
|---|---|---|
| cold | 0.618 | **0.822** |
| dark | 0.678 | **0.887** |
| light | 0.669 | **0.873** |
| warm | 0.668 | **0.853** |
| maize | 0.447 | **0.770** |
| **mean** | 0.616 | **0.841** |

Head (`head_type="joresmpra"`): `LayerNorm → flatten → Linear(512) → ReLU → Linear(5)`,
`pooling_type=flatten`, `dropout=0.6`, `num_outputs=5`. Split is the source TSV's own
`set` column (`split_mode="jores"`). Input is the raw 170 bp promoter (no adapters,
promoter or barcode).

## Loading

```python
from alphagenome_encoder_ft import AlphaGenomeEncoderModel   # the jores26 fork
model = AlphaGenomeEncoderModel.from_checkpoint("finetuned_encoder.pt")
model.eval()
preds = model.predict_sequences(["ACGT..."])   # (N, 5)
# column order: [cold, dark, light, warm, maize]
```

(Equivalently, `alphagenome_ft_mpra.hub.load_pretrained("jores_multicondition")`.)

Verified: loading `finetuned_encoder.pt` and scoring the held-out `set=="test"` rows
reproduces the per-condition Pearson above to within ~0.01–0.02 (the small gap is the
barcode-aggregated eval protocol vs row-level scoring).
