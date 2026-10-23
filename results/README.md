# Results

Everything here was produced by `notebooks/ai4mars_dinov3_final.ipynb` (final run on Kaggle, 2 × T4, 3 seeds) and, for `figures/`, by
`src/make_figures.py`. Checkpoints are not versioned.

## `final_run/`

| File | What it is |
|---|---|
| `table_main.csv` | Main table: mIoU (mean ± sd over seeds, 95 % bootstrap CI over images) and `big_rock` IoU for every configuration on M1, M2, M3 |
| `table_published.csv` | Published baselines (Mohammad, Li & Gao, iSpaRo 2025, Table I), as published |
| `hypotheses_preregistered.json` | Hypotheses H1–H6, expected outcomes and decision rules, written before the final run (internal record with a timestamp) |
| `hypothesis_verdicts.csv` | Verdict for each hypothesis and the numbers behind it |
| `figures/` | The four figures written by the notebook's stage E |
| `results/` | One `.json` + `.npz` pair per unit of work (see below). This is the input of the analysis stage |

### `results/` file names

| Pattern | Stage | Content |
|---|---|---|
| `A_layers_s{seed}` | A | Linear probes on blocks 2, 4, 6, 8, 10, 12 and the final norm |
| `B_decoder512_s{seed}` | B | Decoder head at 512 px |
| `C_{LP,FT2,FT4}_n{N}_s{seed}` | C | Linear probe / last 2 blocks / last 4 blocks fine-tuned with N labelled images |
| `C_sweep` | C | Learning-rate sweep for the fine-tuned runs |
| `D_{dinov3_b16_lvd,dinov2_b14,dinov3_b16_rand}_s{seed}` | D | Backbone comparison (N = 2,000) |

Each `.json` stores the seed, settings, validation history and trainable parameter count. Each `.npz` stores **per-image confusion
matrices** (4 × 4) for validation and the three test sets, for each boundary-band radius (0, 4, 8, 16 px at native resolution), with
shape `(images, radii, 4, 4)`; rows are the true class in the order soil, bedrock, sand, big rock. Any mIoU in the repository can be
recomputed from these arrays.

## `figures/`

`fig1_layers` … `fig7_m3_bigrock` (PNG and PDF) and `numbers.json`, the values quoted in the README. Regenerate with
`python src/make_figures.py`.
