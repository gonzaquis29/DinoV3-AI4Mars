# Poster text (draft, English, follows the structure of INFORME_CORTO)

> Corrections for the printed poster: see [`ERRATA_POSTER_V4.md`](ERRATA_POSTER_V4.md).

**Title:** How Do Web-Pretrained DINOv3 Features Extend to Martian Terrain? A Study on AI4Mars
**Authors:** Gonzalo Quispe, Gustavo Meza, David Rodríguez · PUCP, Lima, Peru

---

## 1. Context and problem
Rovers such as Curiosity must tell soil, bedrock, sand and big rock apart to plan safe paths. Collecting pixel-level labels for this is costly. Vision foundation models such as DINOv3 (ViT-B/16, self-supervised on 1.689 billion Earth images) could cut that cost, but it is not known whether their features extend to a domain as different as Mars: grayscale images, no human-made objects, very similar terrains.

**Problem.** How much of what DINOv3 learned on Earth images transfers to Martian terrain without retraining, which part is worth adapting, and how confidently can such improvements be told apart from noise?

## 2. Background and gap
Models trained specifically on AI4Mars (47–81 M parameters) reach mIoU 0.67–0.70 on the expert test set; big rock is the weakest class [Mohammad et al. 2025]. Frozen foundation models on Mars have been tried only with DINOv2 on other data and a simpler task [MANTLE 2026]; Mars-Bench left AI4Mars out because of its crowd labels.

**Gap.** There is no controlled evaluation of a frozen foundation model on AI4Mars with the expert test set that compares degrees of adaptation and separates real gains from noise (published results are a single number measured on 322 images).

## 3. Objectives
**General:** determine how far frozen DINOv3 serves for Martian terrain segmentation and which adaptations really improve it.
**Specific:** (1) measure frozen DINOv3 with simple heads; (2) test whether changing contrast or texture helps; (3) compare adapting early, middle and late layers at equal budget; (4) quantify the ceiling set by ambiguous labels and overconfidence on big rock.

## 4. Hypotheses (written before the runs)
- **H1.** Changing image contrast or texture does not improve results (the difficulty is not photometric).
- **H2.** Adapting deep layers helps more than early ones.
- **H3.** Much of the error comes from ambiguous labels, and the model is overconfident on big rock.

*A difference is "real" if the 95 % CI excludes 0 and |Δ| ≥ 0.01; "equivalent" if the CI lies inside ±0.01; otherwise "inconclusive".*

## 5. Methodology
- **Data:** AI4Mars, Curiosity NavCam, 512 px, 4 classes. Expert tests M1/M2/M3 (≥1/≥2/3 experts agree, 322 images) plus an internal set T (2,000 images, crowd labels) used only to compare our models with each other. Temporal split by spacecraft clock, 24 h gap, automatic leakage audit (0 duplicates).
- **Model:** frozen DINOv3 ViT-B/16 + linear probe (4,612 params), multilayer decoder (3.5 M) or LoRA rank 8 on two blocks (221 k; early / middle / late).
- **Training:** AdamW, OneCycle, cross-entropy, best epoch by validation, 2–3 seeds.
- **Reference:** DeepLabV3-R50 with and without pretraining, and published results.
- **Statistics:** paired block bootstrap (blocks of 10 images, 2,000 resamples), Holm correction.
- **Two rounds:** round 1 (322-image tests) was inconclusive for lack of power; round 2 added T.

## 6. Results
| Model | Trainable params | M1 | M2 |
|---|---|---|---|
| Linear probe | 4.6 k | 0.674 | 0.723 |
| Decoder | 3.5 M | 0.682 | 0.732 |
| DeepLabV3 (full training) | 39.6 M | 0.675 | 0.727 |
| *Best published* | *47–81 M* | *0.69–0.70* | *0.74* |

- **H1 (appearance):** no alteration helps; CLAHE lowers mIoU by 0.010 (real); standardisation and blur are equivalent to the original.
- **H2 (where to adapt):** on T, late − early = +0.033 [+0.023, +0.044]; late ≈ middle (+0.004, inconclusive). On the expert set it is not confirmed (+0.009 [0.000, +0.032]).
- **H3 (labels, confidence):** mIoU is 0.86 where three experts agree and 0.55 where only one labelled (+0.315 [+0.180, +0.413]). On big rock, confidence exceeds accuracy by 0.674; ECE overall 0.029.
- **Complementary tests:** a loss for rare classes raises big-rock IoU by +0.024 but increases false positives; LoRA on top of the decoder adds +0.006 (inconclusive).

## 7. Discussion and conclusions
- A frozen foundation model with only 4.6 k trainable parameters reaches the range of models thousands of times larger; extra capacity adds at most about 0.01.
- Changing contrast or texture does not help; adapting deeper layers helps somewhat, with a caveat: the early zone had a lower learning rate, so part of the gain may come from that setting, and the gain appears on T but not on the expert set.
- The ceiling seems to come from the labels rather than the model; big rock is the weak point and the model does not know when it is wrong.
- We do not beat published results; the contribution is efficiency (almost the same with far fewer parameters) and rigour (pre-registered rules, confidence intervals, leakage audit).
- **Limits:** one rover, one backbone, 512 px, few repetitions (one for DeepLabV3), M3 contains big rock in only 5 images.
- **Next:** control run for the early zone, more seeds, a larger expert set, 1024 px, other DINOv3 sizes and rovers.

*Public NASA/JPL data; laboratory results, not sufficient alone for rover hazard decisions. Code written with AI assistance (Claude); decisions in DESIGN_LOG.md.*

---

### Figures available (results/estudio_principal/)
`figures/comparison_main.png` (models vs published), `figures/P1_apariencia.png` (H1), `figures_ronda2/6c_iou_clase.png` (IoU per class), `figures_ronda2/6e_calibracion.png` (H3), `figures_ronda2/6d_cualitativo.png` (examples), `figures_ronda2/6b_curvas.png` (curves).
