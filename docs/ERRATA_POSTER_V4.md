# Poster v4: clarifications and errata

The poster (WAIMLAp, 19 Nov 2026) is printed and cannot change, so corrections live here. Numbers below were recomputed from the raw per-run files in
`results/estudio_principal/raw/` (confusion matrices per image) unless marked *not re-checked*. The repository is the reference where it and the poster differ.


## 0. Title: the PDF that was sent differs from the repository's current poster

- The PDF **sent to the organisers** (`Poster_FrozenDINOv3MartianTerrain_v4_enviado.pdf`, exported 18 Nov 2026) is titled *"How Do Web-Pretrained DINOv3 Features Extend to Martian Terrain? A Study on AI4Mars"*.
- On 19 Nov 2026 the title was changed to *"Frozen DINOv3 Features for Martian Terrain Segmentation: A Study on AI4Mars"* (it keeps the opening of the title registered in October and drops "Layer-wise Linear Probing", which the poster body does not show). `Poster_FrozenDINOv3MartianTerrain_v4.pdf` and `.pptx` in this folder carry it.
- The new PDF was exported from the updated `.pptx` (same A0 page size). Besides the title it has the text corrections of sections 0b and 0c. Both files are kept so the difference is visible.
- Three titles exist in the project. The October abstract (`version_octubre/`, the registration version) has the *"Frozen DINOv3 Features … Layer-wise Linear Probing"* title; the sent poster and the final abstract and article have the *"How Do Web-Pretrained DINOv3 Features Extend…"* family. The poster body corresponds to the second family. If the organisers' programme lists the registered title, the programme and the poster body describe different stages of the same project (layer-wise study, then main study and round 2). The abstract and the article carry no round 2 yet.

## 0b. Minor text corrections applied to the final poster files (19 Nov 2026)

`Poster_..._v4.pdf` / `.pptx` in `version_final/` include these text-only fixes relative to the sent PDF (`_enviado.pdf`); figures are unchanged:

- Citations added in the text for references [1], [2], [4] and [6] (they were listed but never cited).
- "1.689 billion Earth images" → "web images".
- Background: "have only been tried with DINOv2" → "On AI4Mars, frozen foundation models are untested: DINOv2 was tried only on other data and a simpler task [5]; Mars-Bench left AI4Mars out…".
- "where only one labelled" → "on pixels labelled in M1 but not in M3".
- Fig. 3 caption: "Row C1 is big-rock IoU" (the axis label in the image says mIoU). Fig. 5 caption: IoU cannot exceed 1; the axis above 1.0 is plot padding (the image itself is unchanged).
- Limits: M3 "without a verified explanation" → "so it is noisy … cause not isolated".

**Not changed** (see section A): the Fig. 3 H2 colour and wording, the 0.674 / ECE 0.029 attribution to Fig. 4, "thousands of times larger" and "pre-registered", and the Fig. 2 mix of rounds. These are corrected only in this errata.

## 0c. Content corrections applied to the final poster files (19 Nov 2026)

The final files also correct, in the text and captions (figures are unchanged), statements that section A flags in the sent PDF:

- H2 is marked "exploratory" in the text and "real on T only" in the Fig. 3 caption (the figure's purple dot is unchanged).
- The 0.674 gap and ECE 0.029 are attributed to the probe on the expert test; the text says Fig. 4 shows other heads on T.
- "Thousands of times larger" is replaced by the size of the published models (47–81 M parameters); "pre-registered rules" by "decision rules written before the runs"; "does not know when it is wrong" by "is overconfident when it predicts [big rock]".
- Fig. 2 caption states that the probe is round 1 and the decoder and DeepLabV3 are round 2. DeepLabV3 is "not clearly different from the probe".
- Methodology: the 24 h gap and leakage audit are labelled round 2.
- Results 2 and 3 headings and bullet labels reworded from "H1 and H2" / "H3 labels and confidence" to descriptive titles (image appearance and where to adapt; label ambiguity and overconfidence); the hypotheses keep their H1–H3 names in Objectives.
- Wording polish with no change of numbers or claims: Problem, Gap, Statistics, Reference and Discussion bullets rephrased in plainer language ("Transfer" in the conclusions became "Bottom line").

Items in section A that remain true of the **sent** PDF are kept below for the record.

## A. What the poster says vs. what the repository shows

| # | Poster statement | Clarification |
|---|---|---|
| 1 | Fig. 3 draws H2 (late − early) as a "real gain" | Real **only on T** (+0.033 [+0.023, +0.044]). The rule fixed before the runs decided on the expert sets, and there it is **inconclusive** (M1 +0.009 [0.000, +0.032]). T was added after round 1 had no verdict, so H2 is **exploratory**, not confirmed. |
| 2 | Results 3: "confidence exceeds accuracy by 0.674 (Fig. 4)"; "overall calibration ECE 0.029" | Both numbers are the **linear probe on the expert test M1 + M2** (round 1, `P3`), not Fig. 4. Fig. 4 shows the round-2 heads on **T**, where the big-rock gap is smaller. What holds in both: big rock is worse calibrated than the whole set. |
| 3 | Fig. 2 mixes models | The probe bars (0.674 / 0.723 / 0.862) are the **round-1 probe** (15,564 training images). The decoder is round 2 (12,997 images, T held out). DeepLabV3 (0.675 / 0.727 / 0.884) is the **round-2 run** (`raw/d_T`, 14 epochs with early stopping, patience 3, 231 min), not the budget-matched round-1 run in the README table (3 epochs, 0.667 / 0.713 / 0.815). The round-2 probe, trained on the same 12,997 images, scores 0.672 / 0.720 / 0.799 on M1 / M2 / M3 (3 seeds). "Decoder" in Fig. 2 is decoder + rare-class loss. Published bars use 1,024 px and another split. |
| 4 | "DeepLabV3 level with the probe" | One seed per DeepLabV3 variant. Round 1: budget-matched, not converged (ImageNet run had plateaued; the run without pretraining was still rising). Round 2 trained it longer; probe − DeepLabV3 on T is −0.005 [−0.019, +0.010], inconclusive. The claim is "no clear difference", not "better". |
| 5 | "Thousands of times larger" | Compares **trainable** parameters (4.6 k) with the **total** parameters of published models. Our full model is 85.7 M parameters and about 215 GFLOPs per 512 px image (`raw/a/teardown.json`). The saving is in training, labels and what must be uplinked, not in inference cost. It does not run on a Curiosity-class processor. |
| 6 | "Pre-registered rules" | The rules are written down before the runs and each file carries its own timestamp (`hipotesis_ronda2.json`: 6 Nov 2026 12:19), but git only has them from the commit that also holds the results, and the round-1 rules were saved again after the pilot. Read it as "decision rules written before the runs", not a formal registration. |
| 7 | "24 h gap" and "automatic leakage audit (0 duplicates)" | Both belong to **round 2** (`audit_splits` in `a4m_lib.py`). Round 1 splits validation by spacecraft clock without a gap and only removes training images whose file name appears in a test set (0 found). Round-2 audit (`raw/*/auditoria_datos.json`): no file in two sets; validation and T are ≥ 24.0 h from every training image; expert sets are ≥ 18.8 h from every training image; 0 near-duplicates removed in `a_T`. |
| 8 | H2 "adapting deep layers helps" | Early used learning rate 2e-4, middle and late 1e-3, each picked as the better of two values on the 2,000-image pilot (one seed) and **not swept again** (`P2T_*_lr_choice.json`). Zone and learning rate are therefore confounded. Mean mIoU on T (15 epochs): probe 0.841 (3 seeds), early 0.844, middle 0.874, late 0.878 (2 seeds each). Early is no better than the probe on T and late is +0.037 above it, so the supported reading is "adapting late or middle helps on T; adapting early adds nothing", and even that is exploratory. On the expert test all three zones add only +0.005 to +0.007 over the probe (round 1). |
| 9 | H1 "no alteration helps" | True in both rounds, but CLAHE is *equivalent* on M1 (−0.002) and a small *real drop* on T (−0.010 [−0.018, −0.004]); the margin is the 0.01 threshold, so it is borderline. H1 also bundles two predictions. The texture one **failed**: we predicted blur would lower mIoU and hit `sand`; it moved −0.002 on M1 (+0.000 on T) and the class that lost most was `soil`. Blur σ = 2 px is smaller than the 16 px patch, so it is a weak test of texture. |
| 10 | "mIoU 0.862 where three experts agree and 0.546 where only one labelled" | The second group is the pixels labelled in M1 but not in M3. Expert disagreement is **confounded** with visual difficulty; we cannot separate label ambiguity from hard pixels. This is why the poster says the ceiling "seems" to come from the labels. The gap appears in all four classes, and excluding an 8 px band around label borders raises mIoU by about +0.025 (consistent with border noise, not proof). |
| 11 | "The model does not know when it is wrong" | Mixes two things. The 0.674 gap is about **false alarms** (when the probe says big rock it is 82 % confident and right 15 % of the time on M1 + M2). The risky error for a rover is the **miss**: about 45 % of big-rock pixels are not detected. Both are reported, as separate numbers. |
| 12 | "M3 fell from 0.93 to 0.87 without a verified explanation" | Big rock appears in 5 of the 322 M3 images, so a few thousand false-positive pixels move its IoU and with it the mIoU. Without that class every DINOv3 model scores 0.96–0.97 on M3. M3 is also noisy across seeds (std on M3: probe 0.005 round 2 / 0.032 round 1, decoder 0.038–0.045). The exact cause for this pair of runs was **not isolated**, but M3 is not used to decide anything. |
| 13 | C1: Lovász raises big-rock IoU "but increases false positives"; Fig. 3 row "C1" | The Fig. 3 row is **big-rock IoU**, not global mIoU as the axis label says. It is a trade-off, not a free gain: +0.024 IoU for about double the false-positive pixels on M3 (52 k → 91 k in the 2×2 average). How to weigh a false alarm against a miss is a mission decision. |
| 14 | "Frozen foundation models on Mars have only been tried with DINOv2" | Should read: "**on AI4Mars specifically**, no controlled evaluation exists". Mars-Bench evaluates frozen feature extractors on other Martian datasets and left AI4Mars out because of its crowd labels (*not re-checked*). We use crowd labels only for training and for paired comparisons on T; every main conclusion is measured on the expert sets. |
| 15 | Header and text minor points | "1.689 billion Earth images" are web images (LVD-1689M). Fig. 5's y-axis reaches 1.2 by mistake (IoU ≤ 1). References [1], [2], [4], [6] are not cited in the text. |

## B. Answers to checks asked before the session

- **Which probe was evaluated on T?** The round-2 probe (`raw/a_T/P1_original`), trained on 12,997 images with T and validation held out (T mean mIoU 0.841). T was **not** in its training. The Fig. 2 probe bars are the round-1 probe.
- **Seed spread on T (population std):** probe 0.001; decoder (CE) 0.012; decoder + Lovász 0.003; LoRA early 0.0001, middle 0.006, late 0.002 (2 seeds each). Seeds share batch order, so these are understated.
- **Late vs. probe on T:** +0.037 (means above). Early vs. probe: +0.003.
- **Decision rule:** a difference is *real* if the paired block-bootstrap 95 % CI (blocks of 10 consecutive images, 2,000 resamples) excludes 0 and |Δ| ≥ 0.01 after Holm correction within each hypothesis; *equivalent* if the CI lies inside ±0.01. No formal power calculation; 2,000 images narrow the CI about 2.5× relative to 322.
- **AI use:** an AI assistant (Claude) wrote the code and helped draft the framing and text, as the course policy allowed; each decision is marked in [`DESIGN_LOG.md`](../DESIGN_LOG.md). The *Verificó* column there is for the authors to fill in.

## C. What would settle the open points (not done)

1. H2 with the same learning rate in every zone and more seeds.
2. Retrain on high-agreement pixels only to test H3 causally.
3. 1,024 px, other DINOv3 sizes, other rovers.
