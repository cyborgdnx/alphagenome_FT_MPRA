# Jores 2026 plant multi-condition MPRA pipeline

Fine-tune the AlphaGenome encoder on the Jores et al. 2026 plant MPRA
([PubMed 38513612](https://pubmed.ncbi.nlm.nih.gov/38513612/)), jointly predicting 5
conditions — **cold, dark, light, warm, maize** — from a 170 bp core promoter, then run
held-out testing plus the zero-shot perturbation and design evaluations.

This is the **PyTorch** AlphaGenome-encoder path (backbone from `alphagenome-pytorch`),
so it uses a dedicated environment, **not** the repo's default JAX env. Ported from
[`alphagenome-ft-jores26`](https://github.com/katelynsyc/alphagenome-ft-jores26); the
model/head/two-stage trainer come from the standard
[`alphagenome-encoder-ft`](https://github.com/MasayukiNagai/alphagenome-encoder-ft)
package. The Jores head is just `MPRAHead(num_outputs=5)` (`head_type="mpra"`).

## Environment

```bash
uv venv --python 3.12 .venv-jores && source .venv-jores/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/jores_multicondition.txt
pip install -e . --no-deps        # make the alphagenome_ft_mpra helpers importable
```

## Data

`modelling_data_tamsACR.tsv` — columns `id`, `enrichment_{cold,dark,light,warm,maize}`,
`sequence`, `set`. Splits come from the `set` column; the pipeline adds the 15 bp Jores
adapters and one-hot encodes. Shared copy:
`/grid/koo/home/shared/data/plant_acr/jores_2026/modelling_data_tamsACR.tsv`.
The design/objective evals need an `experiment` column (`evolution_only_eval.tsv`,
`jores_design_validation.tsv`).

## Modules

| Module | What |
|---|---|
| `alphagenome_ft_mpra/jores_multicondition_data.py` | `JoresMPRADataset`, `create_jores_splits`, adapters, species masks |
| `alphagenome_ft_mpra/jores_multicondition_objectives.py` | evolution-objective parsing + on-target metrics |

## Scripts

```bash
# 1. Fine-tune (two-stage: head-only, then encoder unfrozen)
python scripts/finetune_jores_multicondition.py --config configs/jores_multicondition.json \
    --input_tsv <modelling_data_tamsACR.tsv> \
    --pretrained_weights /grid/koo/home/shared/models/alphagenome/torch/model_all_folds.safetensors

# 2. Test — held-out, on-target design, or TF-perturbation categories
python scripts/test_jores_multicondition.py --mode test         --checkpoint <finetuned_encoder.pt> --input_tsv <modelling_data_tamsACR.tsv>
python scripts/test_jores_multicondition.py --mode by_objective --checkpoint <ckpt> --input_tsv <evolution_only_eval.tsv>
python scripts/test_jores_multicondition.py --mode categories   --checkpoint <ckpt> --input_tsv <jores_design_validation.tsv>

# 3. Zero-shot perturbation — saturation mutagenesis (per-base, per-condition effects)
python scripts/perturbation_jores_multicondition.py --checkpoint <ckpt> --input_tsv <modelling_data_tamsACR.tsv>

# 4. Zero-shot design — greedy in-silico evolution toward a target warm activity
python scripts/design_jores_multicondition.py --checkpoint <ckpt> --input_tsv <modelling_data_tamsACR.tsv> \
    --sequence_ids At-19545_rev --target_warm 3.0
```

Held-out test reproduces mean Pearson **0.841** (cold 0.822 / dark 0.887 / light 0.873 /
warm 0.853 / maize 0.770). The benchmark figure is
`scripts/plot_jores_multicondition_benchmark_results.py`.

## Released weights

`jores_multicondition` at
[`Al-Murphy/alphagenome-encoder-ft`](https://huggingface.co/Al-Murphy/alphagenome-encoder-ft)
(`torch/jores_multicondition/{frozen,finetuned}_encoder.pt`, `head_type="mpra"`):

```python
from alphagenome_encoder_ft import EncoderMPRAModel
model = EncoderMPRAModel.from_checkpoint("finetuned_encoder.pt")   # standard package
```

or `alphagenome_ft_mpra.hub.load_pretrained("jores_multicondition")`.

> The downstream TF-MoDISco / sequence-logo / evolution-path plotting variants are not
> ported — they remain in the alphagenome-ft-jores26 repo.
