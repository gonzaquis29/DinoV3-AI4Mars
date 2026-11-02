# Frozen DINOv3 features for Martian terrain segmentation (AI4Mars)

How far does a **frozen DINOv3 ViT-B/16** go on **AI4Mars** (Curiosity NavCam, 4 classes: soil, bedrock, sand, big rock)?
We probe it layer by layer with a linear head, then with a larger decoder and with partial fine-tuning, and test it against
controls (random-init ViT, DINOv2, fewer labels, label borders) and published end-to-end baselines.

> **Resumen (ES).** Un backbone DINOv3 congelado con una sonda lineal de 4,6 k parámetros queda a 0,02 mIoU de los mejores modelos
> publicados entrenados de punta a punta en AI4Mars (M1/M2); un decoder o el ajuste fino de los últimos bloques añaden como mucho 0,012.
> El conjunto M3 no sirve para comparar modelos en `big_rock` (solo 5 imágenes). De las seis hipótesis registradas antes de la corrida
> final, una (H4: el decoder no supera a la sonda) se apoya y cinco quedan inconclusas con este tamaño de muestra. Todo es reproducible desde los resultados incluidos.
>
> **Estudio v2 (15,564 imágenes).** La sonda lineal iguala, dentro del margen estadístico, a un DeepLabV3 entrenado de punta a punta con ~8,600 veces más parámetros. El techo no depende de la apariencia de la imagen ni de qué bloques se adaptan, sino de la ambigüedad de las etiquetas y de la clase `big_rock`. Ver la sección «What we found».

## What we found (v2 study, all 15,564 training images)

**Question.** How do the features of a DINOv3 pretrained on web images extend to the Martian domain (AI4Mars)?
The v2 study (notebooks in [`notebooks/v2/`](notebooks/v2/), results in [`results/v2_full/`](results/v2_full/)) asks it from three angles:
**P1 data** (is the gap about image appearance?), **P2 model** (where is it worth adapting?), **P3 errors** (where does it fail, and does it know?).

**Answer in short.** The frozen features already work on Mars with almost no adaptation. A 4.6 k-parameter linear probe matches, within the
statistical margin, a DeepLabV3 trained end to end with ~8,600 times more parameters. What caps the score is not how the image looks nor which
blocks are adapted, but label ambiguity and the `big_rock` class.

| Model (test mIoU, mean of seeds) | Trainable params | M1 | M2 | vs. probe |
|---|---|---|---|---|
| Linear probe on frozen DINOv3 (3 seeds) | 4.6 k | 0.674 | 0.723 | n/a |
| DeepLabV3 ResNet-50, ImageNet-pretrained, end to end (1 seed) | 39.6 M | 0.667 | 0.713 | −0.007 [−0.025, +0.008], inconclusive |
| DeepLabV3 ResNet-50, no pretraining (1 seed) | 39.6 M | 0.628 | 0.681 | −0.046 [−0.081, −0.016], real |

Same 15,564 images, same split and test sets (M1/M2/M3, 322 images each). **The baseline is under-trained** (3 epochs, 2,916 steps, 70 min budget per
model; the ImageNet one had plateaued, the one without pretraining was still improving) and is a single seed, so read the comparison as
"the probe matches it", not "the probe beats it". The gain is in cost, not in accuracy.

1. **P1, appearance.** Histogram equalisation, CLAHE and per-image standardisation do not change the probe's mIoU (CLAHE and standardisation are
   equivalent to the original on M1, ±0.01; global equalisation is slightly lower, −0.006). Blurring away fine texture does not sink it either
   (−0.002 on M1). The gap is not photometric. M2 intervals are wider than ±0.01, so several comparisons stay inconclusive.
2. **P2, where to adapt.** LoRA (rank 8, 221 k parameters) on blocks 1–2, 6–7 or 11–12 gives +0.005 to +0.007 over the probe on M1 and no
   difference between zones (late − early: −0.000 on M1, [−0.006, +0.015]). Adaptation adds little, wherever it goes.
3. **P3, errors.** Where the experts agree, mIoU is 0.862; where only one labelled the pixel, 0.546 (difference +0.315 [+0.180, +0.413]), so part of
   the ceiling is label ambiguity. Calibration overall is good (ECE 0.029) but on pixels predicted `big_rock` confidence exceeds accuracy by
   +0.674 [+0.435, +0.762], and 45 % of `big_rock` pixels are missed.
4. **Teardown.** Parameters, FLOPs, latency, memory and feature statistics per block: [`results/v2_full/teardown_table.csv`](results/v2_full/teardown_table.csv).

Full tables: [`P1_table.csv`](results/v2_full/P1_table.csv), [`P2_table.csv`](results/v2_full/P2_table.csv), [`baseline_table.csv`](results/v2_full/baseline_table.csv),
figures in [`results/v2_full/figures/`](results/v2_full/figures/). The decision rules were written down before running
([`hypotheses_v2.json`](notebooks/v2/hypotheses_v2.json), 02/11/2026 01:54; re-registered for the final data in
[`hypotheses_v2_full.json`](notebooks/v2/hypotheses_v2_full.json) after the pilot was known and before any full result was seen).
Verdicts per hypothesis: [`verdicts_v2.csv`](results/v2_full/verdicts_v2.csv) (HP1, HP1b, HP2 inconclusive; HP3a, HP3b supported).
A **real** difference needs a paired block-bootstrap 95 % CI (blocks of 10 consecutive images) that excludes 0 and |Δ| ≥ 0.01; **equivalent** means the CI lies
inside ±0.01; otherwise inconclusive. Several comparisons use Holm's correction.

**Pilot replicate.** The same study first ran with 2,000 training images ([`results/v2_pilot/`](results/v2_pilot/)) and gives the same verdicts, which
supports the findings. It is kept as an independent replicate, not as the main result.

**Limitations of the v2 run** (details in [`DESIGN_LOG.md`](DESIGN_LOG.md)):
- P1 probes used early stopping (patience 2) that cut most of them at epoch 5, before the learning rate had decayed; they land at 0.674 against 0.677 in the
  final run, and one CLAHE seed stopped earlier than the other two. The three seeds share batch order, so seed variance is understated.
- P2 uses 2 seeds, 2 epochs per run (~1,900 steps), and a learning rate fixed from the pilot validation, because of the time available.
- The baseline is 1 seed and under-trained (see above). Confidence intervals are limited by the 322-image test sets, so they are no narrower than in the pilot.
- M3 contains `big_rock` in only 5 of 322 images, so its mIoU cannot rank models on that class (a model-ranking caveat for AI4Mars users, see the final-run section).

## Final run (stages A–E): layer-wise probes, decoder, fine-tuning and controls

## Main results

Test mIoU over 4 classes, mean of 3 seeds, 512 px, 15,564 training images (stage D: 2,000). Test sets are AI4Mars' expert-labelled
M1 / M2 / M3 (labels agreed by at least 1 / 2 / 3 experts), 322 images each, reported separately and scored at native label resolution.

| Method (frozen backbone unless stated) | Trainable params | M1 | M2 | M3 |
|---|---|---|---|---|
| Linear probe, final layer | 4.6 k | 0.677 | 0.725 | 0.827 |
| Decoder (blocks 3, 6, 9, 12) | 3.5 M | 0.684 | 0.735 | 0.931 |
| Last 4 blocks fine-tuned (N = 4,000) | 28 M | 0.687 | 0.737 | 0.927 |
| DINOv2 ViT-B/14, linear probe (N = 2,000) | 4.6 k | 0.664 | 0.714 | 0.798 |
| DINOv2 ViT-B/14 at 448 px (same 32×32 token grid as DINOv3), probe (N = 2,000) | 4.6 k | 0.664 | 0.715 | 0.806 |
| DINOv3 architecture, random init, probe (N = 2,000) | 4.6 k | 0.264 | 0.272 | 0.272 |
| *Published: DeepLabV3+ R101 @512 px* | 58.8 M | 0.67 | 0.72 | 0.82 |
| *Published: SegFormer-B3 @1,024 px* | 47 M | 0.69 | 0.74 | 0.93 |
| *Published: UPerNet Swin-S @1,024 px* | 81.2 M | 0.70 | 0.74 | 0.94 |

The published rows come from Mohammad, Li & Gao (iSpaRo 2025, Table I): single runs on another split and with their own mIoU
convention. **The comparison is indicative, not a like-for-like benchmark.** Full table with 95 % CIs per class:
[`results/final_run/table_main.csv`](results/final_run/table_main.csv).

What the results say:

- **Saturation on M1/M2.** Probe, decoder and fine-tuning end up within 0.012 mIoU of each other; 250 labelled images give 99 % of the
  probe's M1 mIoU (0.671 vs 0.675 at 4,000).
- **Pretrained features matter.** A frozen random-init ViT reaches 0.264 on M1; this is a lower bound, not a from-scratch end-to-end control (not run), so it does not separate pretraining from architecture. All stage-D backbones use N = 2,000 images, so the gaps are between equal-N probes (DINOv3 at N = 2,000 matches the full-data probe: 0.677). DINOv3 beats DINOv2 by +0.013 on M1 (95 % CI [+0.002, +0.025]) but
  not clearly on M2. The two differ in patch size, so we reran DINOv2 at 448 px to match the 32×32 token grid (extra run, H3b): the gap does
  not shrink (+0.014 on M1 [+0.003, +0.026]; +0.010 on M2 [−0.004, +0.025]), so the grid does not explain it. Input pixels still differ (448 vs 512).
- **No intermediate layer beats the last one.** The best intermediate by validation (block 10) is −0.010 on M1 (95 % CI [−0.021, +0.001]).
- **M3 cannot rank models on `big_rock`.** Only 5 of its 322 images contain that class (77,698 px). Every model finds 96–99 % of those
  pixels; they differ in false positives (probe ~109 k px, decoder ~16 k px), which alone drives the M3 mIoU gap (0.83 vs 0.93).
  Without `big_rock`, all models score 0.96–0.97 on M3.
- **Label borders.** Excluding a band of 8 px around label edges raises every head by about 0.025 mIoU on M1 (probe 0.677 → 0.702).
  This is consistent with, but not proof of, label noise at region borders.

### Pre-specified hypotheses

Hypotheses and decision rules were written down before the final run ([`hypotheses_preregistered.json`](results/final_run/hypotheses_preregistered.json),
with a timestamp; this is an internal record, not an external registry, and H4–H6 were formulated after a pilot). H3b was added after the
final run and registered before its own run ([`hypotheses_extra.json`](results/final_run/hypotheses_extra.json)).
A difference counts as real only if its paired-bootstrap 95 % CI over images excludes 0 **and** |Δ| ≥ 0.01 mIoU.

| | Hypothesis | Verdict |
|---|---|---|
| H1 | Intermediate layers beat the last layer | inconclusive |
| H2 | The fine-tuning gain over linear probing grows with the number of labels | inconclusive |
| H3 | DINOv3 beats DINOv2, and both beat random init by a lot | inconclusive |
| H3b | Extra run: DINOv3's edge over DINOv2 survives matching the token grid (DINOv2 at 448 px) | inconclusive (real on M1, not on M2) |
| H4 | A 3.5 M decoder does not beat the linear probe on M1/M2 | supported (the CI includes 0; it does not rule out gains up to ~0.02) |
| H5 | Part of the M1/M2 ceiling is label noise at region borders | inconclusive |
| H6 | `big_rock` failures are size-driven | inconclusive |

Evidence per verdict: [`results/final_run/hypothesis_verdicts.csv`](results/final_run/hypothesis_verdicts.csv).
Inconclusive means the data neither support nor refute the hypothesis at this sample size, not that the effect is absent.

## Poster and abstract

WAIMLAp 2026: [poster (PDF)](docs/Poster_FrozenDINOv3MartianTerrain.pdf) and [abstract (PDF)](docs/Abstract_FrozenDINOv3MartianTerrain.pdf). Editable versions: [poster (PPTX)](docs/Poster_FrozenDINOv3MartianTerrain.pptx), [abstract (DOCX)](docs/Abstract_FrozenDINOv3MartianTerrain.docx).

## Repository layout

```
notebooks/
  ai4mars_dinov3_final.ipynb   final run: stages A-E, resumable, 2 GPUs (source: .py)
  py_to_ipynb.py               regenerates any .ipynb from its .py: python notebooks/py_to_ipynb.py <file.py>
  v2/                          v2 study (each notebook has a .py source and its .ipynb)
    A_datos_y_errores          teardown, P1 (image appearance), data for P3
    B_modelo                   P2 (LoRA by depth)
    C_resultados               analysis on CPU: tables, figures, verdicts (A4M_RUN=full | pilot)
    D_baseline                 DeepLabV3 ResNet-50 baseline trained end to end
    a4m_lib.py                 shared plumbing (data, token caches, training loop, metrics, result store)
    hypotheses_v2*.json        decision rules, written before running
src/
  make_figures.py              figures for the final run
results/
  final_run/                   output of the final run (see results/README.md)
  v2_full/                     v2 study with 15,564 images: tables, figures, raw/{a,b,d}
  v2_pilot/                    same study with 2,000 images (independent replicate)
  figures/                     figures built by src/make_figures.py
docs/                          WAIMLAp poster and abstract
DESIGN_LOG.md                  design log: each decision, why, and where AI assistance was used
```

Stages in the notebook: **A** one linear probe per layer · **B** decoder head · **C** label efficiency and partial fine-tuning ·
**D** DINOv3 vs DINOv2 vs random init · **E** analysis (bootstrap CIs, paired tests, label-noise ceiling, `big_rock` by size, verdicts).
Every metric is stored as a per-image confusion matrix, so any number can be re-bootstrapped over images.

## Reproduce

**1. Analysis and figures from the released results (CPU only, no dataset or weights needed; the analysis stage took about 12 s on a laptop CPU).**
It regenerates `table_main.csv` and `hypothesis_verdicts.csv` byte for byte.

```bash
pip install -r requirements.txt
cp -r results/final_run /tmp/a4m_repro        # stage E writes into its output folder; keep the committed copy intact
A4M_ONLY_E=1 A4M_OUT=/tmp/a4m_repro python notebooks/ai4mars_dinov3_final.py
python src/make_figures.py                    # rewrites results/figures/
```

**2. Full re-run on Kaggle** (measured: ≈ 6.5 h wall-clock on 2 × T4, ≈ 12 GPU-hours).
Create a Kaggle notebook from `notebooks/ai4mars_dinov3_final.ipynb`, set the accelerator to 2 × T4 and attach:

- an AI4Mars copy with the `msl/ncam/{images/edr, labels/train, labels/test/masked-gold-min{1,2,3}-100agree}` layout;
- the DINOv3 ViT-B/16 weights (`facebook/dinov3-vitb16-pretrain-lvd1689m`), either as an attached model/dataset folder containing
  `config.json` and `model.safetensors` or through a Hugging Face token stored as the `HF_TOKEN` secret.

Results are cached per unit of work, so an interrupted run resumes where it stopped (attach the earlier output as an input).
`A4M_SMOKE=1` runs a tiny offline self-test. To run only one stage on selected backbones, set e.g. `A4M_STAGES=D A4M_BACKBONES=dinov2_b14_448`
(this is how the H3b extra run was launched: about 25 min on one T4). Seeds are fixed (0, 1, 2) and each stage records its configuration next to its results.

**3. v2 analysis (CPU).** Regenerates `results/v2_full/` (or `results/v2_pilot/`) from the raw per-run files in `raw/`:

```bash
pip install -r requirements.txt
A4M_RUN=full  python notebooks/v2/C_resultados.py      # or A4M_RUN=pilot
```

**4. v2 re-run on Kaggle.** Notebooks A, B and D import `a4m_lib.py` (attach `notebooks/v2/` as a dataset). Environment variables select the mode:
`A4M_FULL=1` (all 15,564 images; token caches go to disk), `A4M_ZONES=early,mid,late` (B), `A4M_SMOKE=1` (tiny offline self-test). Without `A4M_FULL`
they run the 2,000-image pilot. Measured on 2 × T4 (including data loading): A ≈ 2 h 25 min, B ≈ 1 h 45 min, D ≈ 80 min (training budget set by `BUDGET_MIN`).

## Protocol notes

- Split: training pool 15,564 images, validation = 500 images held out by acquisition time, test = the three expert-labelled sets;
  training images that appear in a test set are removed (0 found).
- Images are resized bilinearly, labels with nearest neighbour, unlabelled pixels (255) are never counted. mIoU averages only the
  classes with non-empty union. The epoch and the layer are chosen on validation, never on test.
- Training augmentation (random crop 50–100 % and horizontal flip) is applied to the images after resizing to 512 px; stages C and D
  train on cached tokens without augmentation.
- Layer-wise stage A trains 4 epochs, the decoder 12; the probe and the decoder therefore differ in more than capacity.

## Limitations

One backbone size, one rover and camera, no end-to-end from-scratch control, three seeds, 322 test images per variant, and a comparison with published numbers that was not
re-run under the same protocol. The extra H3b run was launched in a separate session, with the same code and recipe as stage D. The AI4Mars labels are crowd-sourced. These results say nothing about other planets, sensors or
safety-critical use.

## Data, weights and licences

AI4Mars (Swan et al., CVPR Workshops 2021) is not included; get it from its original source and check its licence there
(its mirrors do not all state the same one). DINOv3 weights are not included and are subject to Meta's DINOv3 licence. The code in this
repository was written with the help of an AI model (Claude).

## References

- O. Siméoni et al., "DINOv3," arXiv:2508.10104, 2025.
- R. M. Swan et al., "AI4MARS: A dataset for terrain-aware autonomous driving on Mars," CVPR Workshops, 2021.
- F. Mohammad, Y. Li, Y. Gao, "Transformers vs CNNs: Enhancing semantic segmentation for Martian terrain classification," iSpaRo, 2025.
- E. Goh, J. Chen, B. Wilson, "Mars terrain segmentation with less labels," arXiv:2202.00791, 2022. Closest prior work: contrastive pretraining of their own backbone without labels, then few labels; reports pixel accuracy, not mIoU.
- P. Durai, G. Doran, "MANTLE: a framework for adaptive in-situ planetary perception using a modular uplink principle," arXiv:2608.28724, 2026.
