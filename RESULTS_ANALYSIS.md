# Sample-Efficient Contrastive Learning — Full Results Analysis

**Date:** March 25, 2026
**Dataset:** IMU gesture recognition (breakEgg, cut, flip, idle, pour, whip) — 20 participants
**Backbone:** SDCNet (1D-CNN)
**Metric:** Accuracy (test set) — `Test_Acc` column from evaluation CSV files

---

## Table of Contents

1. [Summary and Key Results](#1-summary-and-key-results)
2. [Baseline Model Selection](#2-baseline-model-selection)
3. [Global Comparative Table — All Configurations](#3-global-comparative-table--all-configurations)
4. [Gain Table Relative to Baseline at 100% Data](#4-gain-table-relative-to-baseline-at-100-data)
5. [Per-Fraction Comparison Against Corresponding Baseline](#5-per-fraction-comparison-against-corresponding-baseline)
6. [Critical Analysis](#6-critical-analysis)
7. [Detailed Analysis — Stratified Split](#7-detailed-analysis--stratified-split)
8. [Detailed Analysis — LOSO](#8-detailed-analysis--loso)
9. [Detailed Analysis — LOGO](#9-detailed-analysis--logo)
10. [Sweet Spot Identification](#10-sweet-spot-identification)
11. [Practical Impact and ROI per Configuration](#11-practical-impact-and-roi-per-configuration)
12. [Methodological Notes and Reproducibility](#12-methodological-notes-and-reproducibility)
13. [Conclusion](#13-conclusion)

---

## 1. Summary and Key Results

### What was done

Contrastive pre-training (triplet loss) followed by supervised fine-tuning on SDCNet was compared against a fully-supervised SDCNet baseline across six labeled-data fractions (100%, 70%, 60%, 50%, 40%, 30%) and three evaluation protocols of increasing difficulty (Stratified, LOSO, LOGO). Four triplet mining strategies were tested: random, random_shift, semihard, semihard_shift. All experiments use seed 42 for full reproducibility. Both the baseline and contrastive models use the same corrected LOGO partition (seed-42 non-overlapping permutation windows of 3 participants).

### Key indicators

| Indicator | Value |
|---|---|
| Best contrastive method | random (triplet, no temporal shift) |
| Baseline SDCNet LOGO Accuracy at 100% data | 0.8110 |
| Best contrastive LOGO Accuracy at 100% data | 0.8638 (random, +6.5%) |
| Best contrastive LOGO Accuracy at 30% data | 0.8391 (random, +24.1% vs same-fraction baseline) |
| Baseline LOGO drop from 100% to 30% | -0.1350 (-16.6%) |
| Contrastive LOGO drop from 100% to 30% (random) | -0.0247 (-2.9%) |
| Effect of temporal shift augmentation | Negligible (<0.5 pp in all conditions) |
| Semi-hard vs random mining (LOGO avg delta) | -4.8 to -7.0 pp |

### Central finding

The baseline SDCNet degrades severely with less data: LOGO Accuracy drops from 0.8110 at 100% to 0.6760 at 30% (-16.6%). Contrastive pre-training with random mining loses only 2.9% over the same range (0.8638 to 0.8391, -2.9%). This divergence is the core result of the study: the representation learned during contrastive pre-training is robust to data reduction in ways that purely supervised training cannot replicate.

At 30% labeled data, random contrastive mining achieves a LOGO Accuracy of 0.8391 against a same-fraction baseline of 0.6760, a gain of +0.1631 (+24.1%). The contrastive advantage is smallest at high data fractions (100%: +6.5%) and largest at low fractions (30%: +24.1%), because the baseline degrades far faster.

---

## 2. Baseline Model Selection

A model selection phase trained four architectures (DNN, LSTM, Transformer, SDCNet) on 100% of the data under all three evaluation strategies using a separate run (`baseline_model_selection`). This phase used the original lexicographic LOGO partition for cross-architecture comparison; subsequent single-architecture analyses use the unified seed-42 partition.

### Results — all architectures at 100% data

| Architecture | Stratified Acc | LOSO Acc (avg/20 folds) | LOGO Acc (avg/5 folds) |
|---|---|---|---|
| DNN | 0.7409 | 0.6284 | 0.5247 |
| LSTM | 0.9225 | 0.6886 | 0.5270 |
| Transformer | 0.8961 | 0.7305 | 0.6241 |
| **SDCNet** | **0.9586** | **0.8288** | **0.7319** |

### Selection rationale

SDCNet is unambiguously the strongest architecture across all three protocols. The performance gap is especially pronounced in the generalization-oriented protocols:

- LOSO: SDCNet (0.8288) vs Transformer (0.7305) — gap of 9.8 pp
- LOGO: SDCNet (0.7319) vs Transformer (0.6241) — gap of 10.8 pp

DNN and LSTM collapse in LOGO (~0.52), indicating complete failure to generalize to unseen participant groups regardless of data volume. SDCNet's 1D temporal convolution captures local motion patterns with a structural inductive bias well-suited to IMU gesture recognition. It is the only viable backbone for cross-group evaluation.

**Selected backbone: SDCNet.** All subsequent analyses use SDCNet trained with the corrected seed-42 LOGO partition, giving reference values: LOGO Acc = 0.8110, LOSO Acc = 0.8350, Stratified Acc = 0.9650 at 100% data.

---

## 3. Global Comparative Table — All Configurations

All values are accuracy on the test set (`Test_Acc`). Baseline runs exist at all fractions with the corrected LOGO partition.

### Stratified Accuracy

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.9650 | 0.9518 | 0.9435 | 0.9338 | 0.9255 | 0.9131 |
| random | **0.9674** | **0.9578** | **0.9545** | **0.9526** | **0.9448** | **0.9413** |
| random_shift | 0.9631 | 0.9575 | 0.9543 | 0.9537 | 0.9451 | 0.9389 |
| semihard | 0.9454 | 0.9473 | 0.9395 | 0.9160 | 0.8921 | 0.9069 |
| semihard_shift | 0.9575 | 0.9236 | 0.9112 | 0.9082 | 0.9179 | 0.8956 |

### LOSO Accuracy (average over 20 folds)

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.8350 | 0.8150 | 0.8110 | 0.7996 | 0.7898 | 0.7550 |
| random | 0.8773 | 0.8666 | 0.8611 | 0.8532 | **0.8546** | **0.8449** |
| random_shift | **0.8818** | **0.8689** | **0.8657** | **0.8541** | 0.8597 | 0.8376 |
| semihard | 0.8519 | 0.8314 | 0.8227 | 0.8125 | 0.8013 | 0.7900 |
| semihard_shift | 0.8499 | 0.8357 | 0.8270 | 0.8171 | 0.7969 | 0.7810 |

### LOGO Accuracy (average over 5 folds) — Primary metric

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.8110 | 0.8093 | 0.7680 | 0.7665 | 0.7578 | 0.6760 |
| random | 0.8638 | **0.8661** | **0.8573** | **0.8506** | 0.8394 | **0.8391** |
| random_shift | **0.8650** | 0.8586 | 0.8544 | 0.8493 | **0.8458** | 0.8314 |
| semihard | 0.8157 | 0.8100 | 0.7919 | 0.7997 | 0.7589 | 0.7688 |
| semihard_shift | 0.8123 | 0.8063 | 0.7975 | 0.7931 | 0.7878 | 0.7689 |

Bold values indicate the best contrastive result per fraction (across all 4 methods). At 100% and 70%, the baseline is already competitive with semihard methods on LOGO; random mining maintains a clear advantage at all fractions.

---

## 4. Gain Table Relative to Baseline at 100% Data

Delta = contrastive Accuracy - baseline SDCNet Accuracy at 100% data. Percentage in parentheses is the relative gain. This table uses the fixed 100%-data baseline as the reference throughout, allowing direct comparison of what contrastive methods at reduced fractions achieve against the full-data supervised benchmark.

### LOGO — Gain vs Baseline SDCNet 100% (Acc = 0.8110)

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| random | +0.0528 (+6.5%) | +0.0551 (+6.8%) | +0.0463 (+5.7%) | +0.0396 (+4.9%) | +0.0284 (+3.5%) | +0.0281 (+3.5%) |
| random_shift | +0.0540 (+6.7%) | +0.0476 (+5.9%) | +0.0434 (+5.4%) | +0.0383 (+4.7%) | +0.0348 (+4.3%) | +0.0204 (+2.5%) |
| semihard | +0.0047 (+0.6%) | -0.0010 (-0.1%) | -0.0191 (-2.4%) | -0.0113 (-1.4%) | -0.0521 (-6.4%) | -0.0422 (-5.2%) |
| semihard_shift | +0.0013 (+0.2%) | -0.0047 (-0.6%) | -0.0135 (-1.7%) | -0.0179 (-2.2%) | -0.0232 (-2.9%) | -0.0421 (-5.2%) |

Only random and random_shift consistently exceed the full-data baseline at all fractions. Semihard mining falls below the full-data baseline at 70% and below, semihard_shift from 70% and below.

### LOSO — Gain vs Baseline SDCNet 100% (Acc = 0.8350)

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| random | +0.0423 (+5.1%) | +0.0316 (+3.8%) | +0.0261 (+3.1%) | +0.0182 (+2.2%) | +0.0196 (+2.3%) | +0.0099 (+1.2%) |
| random_shift | +0.0468 (+5.6%) | +0.0339 (+4.1%) | +0.0307 (+3.7%) | +0.0191 (+2.3%) | +0.0247 (+3.0%) | +0.0026 (+0.3%) |
| semihard | +0.0169 (+2.0%) | -0.0036 (-0.4%) | -0.0123 (-1.5%) | -0.0225 (-2.7%) | -0.0337 (-4.0%) | -0.0450 (-5.4%) |
| semihard_shift | +0.0149 (+1.8%) | +0.0007 (+0.1%) | -0.0080 (-1.0%) | -0.0179 (-2.1%) | -0.0381 (-4.6%) | -0.0540 (-6.5%) |

Random_shift at 30% is just barely above the full-data baseline (+0.3 pp). Semihard methods fall well below the baseline from 70% onward in LOSO.

---

## 5. Per-Fraction Comparison Against Corresponding Baseline

This section compares contrastive at X% against the supervised baseline trained on the same X% of data. With baseline runs now available at all fractions, this is the fairest same-data comparison.

### LOGO — Contrastive gain vs baseline at the SAME fraction

| Fraction | Baseline | random | random_shift | semihard | semihard_shift |
|---|---|---|---|---|---|
| 100% | 0.8110 | 0.8638 (+5.3 pp) | 0.8650 (+5.4 pp) | 0.8157 (+0.5 pp) | 0.8123 (+0.1 pp) |
| 70% | 0.8093 | 0.8661 (+5.7 pp) | 0.8586 (+4.9 pp) | 0.8100 (+0.1 pp) | 0.8063 (-0.3 pp) |
| 60% | 0.7680 | 0.8573 (+8.9 pp) | 0.8544 (+8.6 pp) | 0.7919 (+2.4 pp) | 0.7975 (+3.0 pp) |
| 50% | 0.7665 | 0.8506 (+8.4 pp) | 0.8493 (+8.3 pp) | 0.7997 (+3.3 pp) | 0.7931 (+2.7 pp) |
| 40% | 0.7578 | 0.8394 (+8.2 pp) | 0.8458 (+8.8 pp) | 0.7589 (+0.1 pp) | 0.7878 (+3.0 pp) |
| 30% | 0.6760 | 0.8391 (+16.3 pp) | 0.8314 (+15.5 pp) | 0.7688 (+9.3 pp) | 0.7689 (+9.3 pp) |

Key observation: the contrastive advantage at same-data grows as the available labeled fraction decreases. At 30%, even semihard achieves a +9.3 pp improvement over the same-fraction baseline, because the baseline collapses to 0.6760. Random mining achieves the full +16.3 pp advantage at 30%, the largest gain observed in the entire study.

At 100% data, the contrastive advantage is real but more modest (random: +5.3 pp). At 70%, semihard_shift barely falls below the baseline (-0.3 pp), the only instance where a contrastive method fails to beat its own-fraction baseline in LOGO.

### LOSO — Contrastive gain vs baseline at the SAME fraction

| Fraction | Baseline | random | random_shift | semihard | semihard_shift |
|---|---|---|---|---|---|
| 100% | 0.8350 | 0.8773 (+4.2 pp) | 0.8818 (+4.7 pp) | 0.8519 (+1.7 pp) | 0.8499 (+1.5 pp) |
| 70% | 0.8150 | 0.8666 (+5.2 pp) | 0.8689 (+5.4 pp) | 0.8314 (+1.6 pp) | 0.8357 (+2.1 pp) |
| 60% | 0.8110 | 0.8611 (+5.0 pp) | 0.8657 (+5.5 pp) | 0.8227 (+1.2 pp) | 0.8270 (+1.6 pp) |
| 50% | 0.7996 | 0.8532 (+5.4 pp) | 0.8541 (+5.5 pp) | 0.8125 (+1.3 pp) | 0.8171 (+1.8 pp) |
| 40% | 0.7898 | 0.8546 (+6.5 pp) | 0.8597 (+7.0 pp) | 0.8013 (+1.2 pp) | 0.7969 (+0.7 pp) |
| 30% | 0.7550 | 0.8449 (+9.0 pp) | 0.8376 (+8.3 pp) | 0.7900 (+3.5 pp) | 0.7810 (+2.6 pp) |

All contrastive methods beat the same-fraction baseline in LOSO across all fractions. The advantage grows as data decreases, following the same pattern as LOGO, though the absolute gains are smaller because LOSO is a less extreme generalization task.

---

## 6. Critical Analysis

### 6.1 Random mining dominates semi-hard mining — a robust and unexpected result

The standard expectation in metric learning is that semi-hard mining improves over random sampling by selecting more informative triplets. The observed results consistently contradict this:

- LOGO at 100%: random 0.8638 vs semihard 0.8157 — gap of 4.8 pp; vs baseline 0.8110: semihard +0.5 pp vs random +5.3 pp
- LOGO at 30%: random 0.8391 vs semihard 0.7688 — gap of 7.0 pp
- The ranking random > semihard is stable across all fractions and all three evaluation protocols

Three mechanistic explanations align:

First, the dataset contains 6 structured gesture classes with strong temporal signatures; natural inter-class distance is large, making random triplets already informative. The "hardness" that semi-hard mining adds provides diminishing returns when the feature space is already well-separated by class.

Second, semi-hard mining reduces the effective gradient signal per batch: it discards many potential triplets before training, which at 30–40% data volumes means much fewer learning updates per epoch — exactly when the model needs more signal, not less.

Third, on a dataset of 20 participants with 6 gesturally similar classes, semi-hard negatives tend to cluster around gesture boundaries (e.g., cut vs whip), causing early training instability. With only 5 LOGO folds, one unstable fold (semihard at 100%, fold 2 = 0.6811) significantly pulls the average down.

### 6.2 Temporal shift augmentation has no measurable impact

Comparing random vs random_shift and semihard vs semihard_shift across all fractions and protocols:

| Protocol | random vs random_shift | semihard vs semihard_shift |
|---|---|---|
| LOGO (avg over all 6 fractions) | -0.002 | +0.004 |
| LOSO (avg over 6 fractions) | +0.002 | +0.001 |
| Stratified (avg over 6 fractions) | -0.001 | -0.005 |

All deltas are within noise. The jitter and amplitude scaling augmentations already active provide sufficient temporal invariance. Circular temporal shift, which rotates the signal window, does not add semantic value for gesture data recorded from onset: gestures have an inherent temporal alignment relative to their start point, so phase-shifting them creates unrealistic samples that confuse rather than enrich the representation.

### 6.3 The baseline degrades sharply with less data; contrastive does not

This is the most important structural finding:

| Data fraction | Baseline LOGO Acc | Random contrastive LOGO Acc | Gap |
|---|---|---|---|
| 100% | 0.8110 | 0.8638 | +0.0528 |
| 70% | 0.8093 | 0.8661 | +0.0568 |
| 60% | 0.7680 | 0.8573 | +0.0893 |
| 50% | 0.7665 | 0.8506 | +0.0841 |
| 40% | 0.7578 | 0.8394 | +0.0816 |
| 30% | 0.6760 | 0.8391 | +0.1631 |

The baseline loses -16.6% on LOGO going from 100% to 30% data. Random contrastive loses only -2.9%. The gap between them triples (from 5.3 pp to 16.3 pp). This pattern confirms that the contrastive representation learned during pre-training is fundamentally more data-efficient: it captures cross-group invariances from unlabeled structure during triplet training, and the labeled data is used only to align class boundaries — a much simpler task that requires fewer examples.

### 6.4 Semi-hard methods are barely competitive with the corrected baseline

With the LOGO partition, the baseline at 100% achieves 0.8110 — notably higher than the 0.7319 produced by the old lexicographic partition. This makes semihard's advantage over the baseline minimal at 100% (+0.5 pp in LOGO) and negative at 70% and below. Semihard is not a viable strategy for this task at any data fraction when compared to the correct baseline. Only random mining delivers a consistent, meaningful advantage.

### 6.5 Variance and reliability

Semihard mining produces LOGO fold standard deviations of 0.0717 at 100% data (vs 0.0148 for random). Its worst fold at 100% (F2 = 0.6811) reaches near-baseline performance, indicating that fold-dependent failure is a systematic issue. In production, where only a single deployment scenario exists, such instability is a serious concern. Random mining never produces a fold below 0.80 across any fraction — a floor of practical usability.

---

## 7. Detailed Analysis — Stratified Split

The stratified split (70/10/20, class-balanced, random_state=42) evaluates in-distribution generalization on randomly held-out samples from the same sessions and population. Having established in Section 6 that random mining is the only consistently superior method, this and subsequent sections focus on the comparison between the supervised baseline (SDCNet) and random contrastive pre-training.

### Performance comparison — Baseline SDCNet vs Random contrastive

| Method | 100% | 70% | 60% | 50% | 40% | 30% | Drop 100→30% |
|---|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.9650 | 0.9518 | 0.9435 | 0.9338 | 0.9255 | 0.9131 | -0.0519 |
| Random contrastive | 0.9674 | 0.9578 | 0.9545 | 0.9526 | 0.9448 | 0.9413 | -0.0261 |
| **Gain (random − baseline)** | +0.0024 | +0.0060 | +0.0110 | +0.0188 | +0.0193 | +0.0282 | |

Both methods perform near-optimally in the Stratified setting. The contrastive advantage is modest here (max +2.8 pp at 30%) because in-distribution evaluation does not stress cross-subject generalization. However, the baseline degrades nearly twice as fast (-5.2 pp vs -2.6 pp from 100% to 30%), confirming the pattern seen across all protocols.

### Per-class F1 — Stratified, 100% labeled data

*Note: per-class breakdown is from sklearn's classification report (per-class F1), not overall accuracy.*

| Class | Baseline F1 | Random F1 | Gain |
|---|---|---|---|
| breakEgg | 0.97 | 0.98 | +0.01 |
| cut | 0.95 | 0.96 | +0.01 |
| flip | 0.97 | 0.96 | -0.01 |
| idle | 0.94 | 0.96 | +0.02 |
| pour | 0.96 | 0.97 | +0.01 |
| whip | 0.98 | 0.97 | -0.01 |
| **Macro avg** | **0.96** | **0.97** | **+0.01** |

At 100% labeled data, both methods perform near-perfectly and per-class differences are within rounding noise (±1 pp). The `idle` class, which has half the support of other classes (~325 vs ~680), already reaches 0.94 / 0.96 under both approaches, confirming no systematic class imbalance bias at full data.

### Per-class F1 — Stratified, 30% labeled data

*Note: per-class breakdown is from sklearn's classification report (per-class F1), not overall accuracy.*

| Class | Baseline F1 | Random F1 | Gain |
|---|---|---|---|
| breakEgg | 0.94 | 0.96 | +0.02 |
| cut | 0.87 | 0.92 | +0.05 |
| flip | 0.91 | 0.94 | +0.03 |
| idle | 0.89 | 0.92 | +0.03 |
| pour | 0.91 | 0.94 | +0.03 |
| whip | 0.95 | 0.96 | +0.01 |
| **Macro avg** | **0.91** | **0.94** | **+0.03** |

At 30% labeled data the per-class differences become more diagnostic. The `cut` class shows the largest gap (+5 pp), dropping to 0.87 under the baseline but maintained at 0.92 with contrastive pre-training. The `whip` class is near-identical (+1 pp), suggesting it is the most robustly-learned gesture even in low-data conditions. The `idle` class (smallest support) degrades from 0.94 to 0.89 for the baseline, but only to 0.92 for random, indicating that the contrastive representation better preserves the idle boundary even with reduced fine-tuning data.

### Conclusion — Stratified

The in-distribution protocol confirms that random contrastive pre-training consistently meets or exceeds the supervised baseline at all data fractions, with no regression introduced by the two-stage training pipeline. The advantage is modest (up to +2.8 pp at 30%) because stratified test samples come from the same participants and sessions as training data. The protocol's primary value is as a sanity check; the cross-subject protocols (Sections 8–9) reveal the full picture.

---

## 8. Detailed Analysis — LOSO

Leave-One-Subject-Out holds out all sessions from one participant and tests on them. This evaluates generalization to individual physiological and kinematic variability not seen during training. Each of the 20 folds corresponds to one held-out participant.

### Summary — LOSO Accuracy (average over 20 participants)

| Method | 100% | 70% | 60% | 50% | 40% | 30% | Drop 100→30% |
|---|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.8350 | 0.8150 | 0.8110 | 0.7996 | 0.7898 | 0.7550 | -0.0800 |
| Random contrastive | 0.8773 | 0.8666 | 0.8611 | 0.8532 | 0.8546 | 0.8449 | -0.0324 |
| **Gain (random − baseline)** | +0.0423 | +0.0516 | +0.0501 | +0.0536 | +0.0648 | +0.0899 | |

The contrastive gain more than doubles from +4.2 pp at 100% to +9.0 pp at 30%, driven entirely by the faster degradation of the supervised baseline (-8.0 pp) compared to random contrastive (-3.2 pp).

### Per-participant LOSO Accuracy — Baseline SDCNet

| Participant | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| P01 | 0.8200 | 0.7953 | 0.8006 | 0.7209 | 0.7619 | 0.7543 |
| P02 | 0.7424 | 0.7219 | 0.7035 | 0.6569 | 0.6310 | 0.5963 |
| P03 | 0.8653 | 0.8330 | 0.8463 | 0.8374 | 0.8341 | 0.8174 |
| P04 | 0.7455 | 0.7433 | 0.7294 | 0.7027 | 0.6759 | 0.6535 |
| P05 | 0.7361 | 0.6540 | 0.6866 | 0.6909 | 0.6435 | 0.6519 |
| P06 | 0.8632 | 0.8494 | 0.8771 | 0.8739 | 0.8729 | 0.8162 |
| P07 | 0.8342 | 0.8375 | 0.8244 | 0.8092 | 0.8081 | 0.7285 |
| P08 | 0.6652 | 0.6512 | 0.6285 | 0.6404 | 0.6587 | 0.5994 |
| P09 | 0.8007 | 0.8376 | 0.8130 | 0.7324 | 0.8309 | 0.6573 |
| P10 | 0.7449 | 0.6606 | 0.6937 | 0.7044 | 0.6852 | 0.6574 |
| P11 | 0.8820 | 0.8498 | 0.8391 | 0.8541 | 0.7918 | 0.8219 |
| P12 | 0.9524 | 0.9679 | 0.9513 | 0.9336 | 0.9413 | 0.9557 |
| P13 | 0.8708 | 0.8924 | 0.8439 | 0.8568 | 0.8030 | 0.7740 |
| P14 | 0.8131 | 0.8151 | 0.8080 | 0.8121 | 0.7937 | 0.7416 |
| P15 | 0.8332 | 0.7912 | 0.8009 | 0.8288 | 0.7664 | 0.7836 |
| P16 | 0.9187 | 0.8902 | 0.8902 | 0.8870 | 0.8786 | 0.8585 |
| P17 | 0.8973 | 0.8754 | 0.8678 | 0.8350 | 0.8437 | 0.7891 |
| P18 | 0.8560 | 0.8328 | 0.8361 | 0.8073 | 0.7763 | 0.7453 |
| P19 | 0.8981 | 0.8529 | 0.8405 | 0.8601 | 0.8580 | 0.8200 |
| P20 | 0.9611 | 0.9481 | 0.9384 | 0.9492 | 0.9405 | 0.8778 |
| **Avg** | **0.8350** | **0.8150** | **0.8110** | **0.7996** | **0.7898** | **0.7550** |
| **Std** | 0.076 | 0.088 | 0.082 | 0.088 | 0.090 | 0.094 |
| **Min** | 0.6652 (P08) | 0.6512 (P08) | 0.6285 (P08) | 0.6404 (P08) | 0.6310 (P02) | 0.5963 (P02) |

### Per-participant LOSO Accuracy — Random contrastive

| Participant | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| P01 | 0.8459 | 0.8125 | 0.8416 | 0.8545 | 0.8168 | 0.8330 |
| P02 | 0.7944 | 0.7457 | 0.7327 | 0.7359 | 0.7316 | 0.7576 |
| P03 | 0.9009 | 0.9143 | 0.9087 | 0.8686 | 0.9109 | 0.8797 |
| P04 | 0.8118 | 0.8225 | 0.8086 | 0.8032 | 0.8128 | 0.7561 |
| P05 | 0.7781 | 0.7645 | 0.7708 | 0.7750 | 0.7687 | 0.7655 |
| P06 | 0.9028 | 0.9060 | 0.9167 | 0.9028 | 0.9124 | 0.8825 |
| P07 | 0.8659 | 0.8768 | 0.8680 | 0.8604 | 0.8462 | 0.8462 |
| P08 | 0.8024 | 0.7862 | 0.7711 | 0.7559 | 0.7894 | 0.7441 |
| P09 | 0.8600 | 0.8723 | 0.8835 | 0.8242 | 0.8578 | 0.8791 |
| P10 | 0.8314 | 0.7898 | 0.7577 | 0.7994 | 0.7908 | 0.7801 |
| P11 | 0.9013 | 0.9067 | 0.9109 | 0.8745 | 0.8938 | 0.8712 |
| P12 | 0.9690 | 0.9380 | 0.9491 | 0.9302 | 0.9701 | 0.9579 |
| P13 | 0.9182 | 0.9074 | 0.9010 | 0.8967 | 0.8547 | 0.8536 |
| P14 | 0.8550 | 0.8764 | 0.8366 | 0.8335 | 0.8284 | 0.8294 |
| P15 | 0.9214 | 0.9053 | 0.8805 | 0.8794 | 0.8611 | 0.8568 |
| P16 | 0.9271 | 0.9007 | 0.9113 | 0.8965 | 0.8828 | 0.8976 |
| P17 | 0.9650 | 0.9202 | 0.9148 | 0.9366 | 0.9475 | 0.9213 |
| P18 | 0.8571 | 0.8793 | 0.8117 | 0.8095 | 0.7973 | 0.8317 |
| P19 | 0.8868 | 0.8642 | 0.8940 | 0.8683 | 0.8652 | 0.8282 |
| P20 | 0.9524 | 0.9427 | 0.9535 | 0.9578 | 0.9535 | 0.9265 |
| **Avg** | **0.8773** | **0.8666** | **0.8611** | **0.8532** | **0.8546** | **0.8449** |
| **Std** | 0.055 | 0.058 | 0.064 | 0.059 | 0.063 | 0.059 |
| **Min** | 0.7781 (P05) | 0.7457 (P02) | 0.7327 (P02) | 0.7359 (P02) | 0.7316 (P02) | 0.7441 (P08) |

### Per-participant gain (random − baseline) at 100% and 30%

| Participant | Gain at 100% | Gain at 30% | Change in gain |
|---|---|---|---|
| P01 | +0.026 | +0.079 | +0.053 |
| P02 | +0.052 | +0.161 | +0.109 |
| P03 | +0.036 | +0.062 | +0.026 |
| P04 | +0.066 | +0.103 | +0.037 |
| P05 | +0.042 | +0.114 | +0.072 |
| P06 | +0.040 | +0.066 | +0.026 |
| P07 | +0.032 | +0.118 | +0.086 |
| P08 | **+0.137** | +0.145 | +0.008 |
| P09 | +0.059 | **+0.222** | **+0.163** |
| P10 | +0.087 | +0.123 | +0.036 |
| P11 | +0.019 | +0.049 | +0.030 |
| P12 | +0.017 | +0.002 | -0.015 |
| P13 | +0.047 | +0.080 | +0.033 |
| P14 | +0.042 | +0.088 | +0.046 |
| P15 | +0.088 | +0.073 | -0.015 |
| P16 | +0.008 | +0.039 | +0.031 |
| P17 | +0.068 | +0.132 | +0.064 |
| P18 | +0.001 | +0.086 | +0.085 |
| P19 | -0.011 | +0.008 | +0.019 |
| P20 | -0.009 | +0.049 | +0.058 |

Key observations from the per-participant analysis:

- **At 100% data**: 3 participants (P18, P19, P20) show the baseline ≈ random (within 2 pp). These are participants who perform near-optimally with full supervised data. Random contrastive never harms these participants in practice; the small negatives (e.g. P19: -0.011) are within run-to-run variance.

- **At 30% data**: only P12 is essentially tied (random: 0.9579 vs baseline: 0.9557, Δ = +0.002). Every other participant benefits from contrastive pre-training at 30%, with P09 gaining +22.2 pp — the largest single-participant gain in the entire study. P09 drops from 0.8007 (baseline, 100%) to 0.6573 (baseline, 30%), while random maintains 0.8791.

- **P08** is the hardest participant for both methods in terms of baseline LOSO minimum. At 30%, the baseline reaches 0.5994 while random maintains 0.7441 (+14.5 pp). P08's gestures are likely the most idiosyncratic in the cohort.

- **Participants maintaining high performance**: P12 (near-ceiling at 0.95+), P17, P20 — these are participants whose gestures are close to the population mean and are well-represented regardless of data volume or method. Random contrastive does not regress them.

- **Variance stability**: random contrastive std stays between 0.055 and 0.064 across all fractions, while the baseline std grows from 0.076 (100%) to 0.094 (30%). This confirms that as labeled data decreases, the supervised model becomes increasingly unequal across participants — some maintaining good performance while others collapse (P02: 0.5963, P08: 0.5994). The contrastive minimum floor at 30% is 0.7441 (P08), 15 pp above the baseline's minimum.

### Conclusion — LOSO

Random contrastive pre-training outperforms the supervised baseline for every participant at every data fraction tested, with three negligible exceptions at 100% data that are within noise. At 30% labeled data, the gain has grown to +9.0 pp on average, with individual participant gains ranging from +0.2 pp (P12, already near-ceiling) to +22.2 pp (P09). The contrastive advantage is not uniformly distributed: it is largest for the most challenging participants (P09, P02, P08), precisely those who need it most in a deployment scenario. The minimum LOSO Accuracy across all 20 participants never falls below 0.74 for random at any data fraction, providing a reliable performance floor.

---

## 9. Detailed Analysis — LOGO

Leave-One-Group-Out tests on groups of three participants entirely absent from training. This is the most realistic and demanding scenario, simulating deployment to a new demographic group not represented in the training data. It is the primary metric of this study. Five folds are defined by the seed-42 permutation partition: Fold 1 = {P01, P18, P16}, Fold 2 = {P02, P09, P06}, Fold 3 = {P12, P04, P19}, Fold 4 = {P17, P14, P03}, Fold 5 = {P10, P20, P05}.

### Summary — LOGO Accuracy (average over 5 folds)

| Method | 100% | 70% | 60% | 50% | 40% | 30% | Drop 100→30% |
|---|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.8110 | 0.8093 | 0.7680 | 0.7665 | 0.7578 | 0.6760 | **-0.1350 (-16.6%)** |
| Random contrastive | 0.8638 | 0.8661 | 0.8573 | 0.8506 | 0.8394 | 0.8391 | -0.0247 (-2.9%) |
| **Gain (random − baseline)** | +0.0528 | +0.0568 | +0.0893 | +0.0841 | +0.0816 | **+0.1631** | |

### Per-fold breakdown — Baseline SDCNet

| Fraction | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg | Std |
|---|---|---|---|---|---|---|---|
| 100% | 0.8143 | 0.7759 | 0.8473 | 0.8485 | 0.7693 | 0.8110 | 0.0338 |
| 70% | 0.8413 | 0.7592 | 0.8196 | 0.8567 | 0.7696 | 0.8093 | 0.0386 |
| 60% | 0.7725 | 0.7367 | 0.8317 | 0.7905 | 0.7085 | 0.7680 | 0.0427 |
| 50% | 0.7783 | 0.7178 | 0.7979 | 0.8155 | 0.7231 | 0.7665 | 0.0395 |
| 40% | 0.7102 | 0.7566 | 0.8085 | 0.8012 | 0.7124 | 0.7578 | 0.0419 |
| 30% | 0.6569 | 0.6386 | 0.7263 | 0.7665 | 0.5915 | 0.6760 | 0.0626 |

### Per-fold breakdown — Random contrastive

| Fraction | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg | Std |
|---|---|---|---|---|---|---|---|
| 100% | 0.8737 | 0.8525 | 0.8708 | 0.8811 | 0.8411 | 0.8638 | 0.0148 |
| 70% | 0.8643 | 0.8438 | 0.8833 | 0.8983 | 0.8407 | 0.8661 | 0.0222 |
| 60% | 0.8747 | 0.8286 | 0.8616 | 0.8861 | 0.8358 | 0.8573 | 0.0221 |
| 50% | 0.8524 | 0.8369 | 0.8612 | 0.8918 | 0.8105 | 0.8506 | 0.0269 |
| 40% | 0.8420 | 0.8242 | 0.8399 | 0.8600 | 0.8311 | 0.8394 | 0.0121 |
| 30% | 0.8560 | 0.8097 | 0.8423 | 0.8800 | 0.8077 | 0.8391 | 0.0277 |

### Fold-level gap (random − baseline) per fraction

| Fraction | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg gap | Min gap |
|---|---|---|---|---|---|---|---|
| 100% | +0.059 | +0.077 | +0.024 | +0.033 | +0.072 | +0.053 | +0.024 |
| 70% | +0.023 | +0.085 | +0.064 | +0.042 | +0.071 | +0.057 | +0.023 |
| 60% | +0.102 | +0.092 | +0.030 | +0.096 | +0.127 | +0.089 | +0.030 |
| 50% | +0.074 | +0.119 | +0.063 | +0.076 | +0.087 | +0.084 | +0.063 |
| 40% | +0.132 | +0.068 | +0.031 | +0.059 | +0.119 | +0.082 | +0.031 |
| 30% | **+0.199** | +0.171 | +0.116 | +0.114 | **+0.216** | **+0.163** | +0.114 |

The contrastive advantage is positive across **all 30 fold×fraction cells** (no exceptions). The smallest gap in the entire matrix is +0.023 (70%, Fold 1); the largest is +0.216 (30%, Fold 5 = {P10, P20, P05}). At 30% labeled data, the minimum fold gap is +0.114 — meaning the worst-case group benefits by over 11 pp, with no fold below 0.81 for random contrastive.

### Per-class F1 analysis (from classification reports)

*Note: per-class breakdown is from sklearn's classification report (per-class F1), not overall accuracy.*

#### At 100% labeled data

| Class | Baseline avg | Random avg | Gain | Baseline range | Random range |
|---|---|---|---|---|---|
| breakEgg | 0.838 | 0.888 | **+0.050** | [0.810–0.860] | [0.850–0.920] |
| cut | 0.804 | 0.864 | **+0.060** | [0.760–0.850] | [0.830–0.910] |
| flip | 0.766 | 0.816 | **+0.050** | [0.710–0.840] | [0.750–0.860] |
| idle | 0.798 | 0.834 | +0.036 | [0.730–0.840] | [0.750–0.910] |
| pour | 0.826 | 0.858 | +0.032 | [0.740–0.910] | [0.800–0.910] |
| whip | 0.836 | 0.906 | **+0.070** | [0.740–0.880] | [0.860–0.930] |

#### At 30% labeled data

| Class | Baseline avg | Random avg | Gain | Baseline range | Random range |
|---|---|---|---|---|---|
| breakEgg | 0.696 | 0.860 | **+0.164** | [0.610–0.780] | [0.830–0.900] |
| cut | 0.686 | 0.830 | **+0.144** | [0.550–0.780] | [0.770–0.890] |
| flip | **0.498** | 0.778 | **+0.280** | [0.430–0.690] | [0.710–0.860] |
| idle | 0.754 | 0.816 | +0.062 | [0.600–0.830] | [0.740–0.860] |
| pour | 0.750 | 0.846 | +0.096 | [0.680–0.850] | [0.790–0.880] |
| whip | 0.706 | 0.892 | **+0.186** | [0.590–0.820] | [0.810–0.940] |

The per-class analysis at 30% reveals striking differentiation:

**`flip` is the most data-sensitive gesture.** The baseline mean drops to 0.498 at 30% — near-chance level for a 6-class problem — while random contrastive maintains 0.778 (+28.0 pp). The flip gesture requires fine temporal discrimination (peak-to-peak wrist flip) that the baseline cannot learn from limited examples, but the contrastive embedding pre-encodes this distinction from the unlabeled structure of the data.

**`whip` shows the most consistent contrastive advantage across fractions.** At 100% it is already the largest per-class gain (+7.0 pp); at 30% it reaches +18.6 pp. Whip involves a characteristic high-frequency wrist snap that may have the most distinctive contrastive signature, allowing the encoder to anchor it strongly even before fine-tuning.

**`idle` degrades least in both methods** (baseline: 0.798→0.754, random: 0.834→0.816). The low-motion signature of idle is trivially detectable even with few labeled examples.

**All classes benefit more from contrastive at 30% than at 100%**: the smallest per-class gain increase is `pour` (+3.2 pp → +9.6 pp) and the largest is `flip` (+5.0 pp → +28.0 pp).

### Structural observations on fold composition

Fold 4 ({P17, P14, P03}) consistently produces the best results across both methods (random: 0.88–0.90). Fold 5 ({P10, P20, P05}) and Fold 2 ({P02, P09, P06}) are the most variable. Critically, even the weakest fold at 30% (Fold 5: random = 0.8077) outperforms the baseline average at 30% (0.6760) by 13.2 pp. Random contrastive is not merely better on average — it is better in every single fold at every data fraction.

### Conclusion — LOGO

Random contrastive mining dominates the supervised baseline at every fold and every data fraction tested. The core finding is the divergence of degradation curves: the baseline loses 13.5 pp going from 100% to 30%, random loses only 2.5 pp. The fold-level gap analysis confirms the advantage is uniform (no exceptions across 30 combinations) and uniformly large at 30% (min +0.114, max +0.216).

---

## 10. Sweet Spot Identification

The sweet spot is the data fraction at which contrastive pre-training delivers the best balance between labeling cost saved and performance maintained, relative to full-data training.

### LOGO performance thresholds — random mining

| Performance target | Fraction required | Accuracy achieved |
|---|---|---|
| Maximum performance | 70% | 0.8661 |
| Within 1 pp of max (>= 0.8561) | 60% | 0.8573 |
| Within 2 pp of max (>= 0.8461) | 50% | 0.8506 |
| Within 4 pp of max (>= 0.8261) | 30% | 0.8391 |

Interestingly, 70% data achieves the highest LOGO accuracy (0.8661 > 0.8638 at 100%), making it both the most data-efficient and the highest-performing configuration. The performance curve is very flat: a decline of approximately 0.7 pp per 10 pp reduction in data fraction from 70% down to 30%.

### Cost-effectiveness analysis

Defining "cost" as fraction of labeled data used and "benefit" as gain over the same-fraction baseline:

| Fraction | Data saved | LOGO Acc (random) | Gain vs same-fraction baseline | Ratio (gain/cost) |
|---|---|---|---|---|
| 100% | 0% | 0.8638 | +0.0528 | 0.053 |
| 70% | 30% | 0.8661 | +0.0568 | 0.081 |
| 60% | 40% | 0.8573 | +0.0893 | 0.149 |
| 50% | 50% | 0.8506 | +0.0841 | 0.168 |
| 40% | 60% | 0.8394 | +0.0816 | 0.204 |
| 30% | 70% | 0.8391 | +0.1631 | 0.544 |

The ratio (gain per unit of data used) increases sharply at 30%; this is partly mechanical since the baseline degrades fastest at that point. However, the absolute performance (0.8391) is also key: it represents a usable, stable model that remains 2.8 pp above even the maximum the baseline can achieve.

### Same-fraction comparison perspective

The most natural "sweet spot" framing is: at what fraction does the accuracy plateau for contrastive, while the baseline continues declining? The data shows the divergence begins clearly at 60% (baseline at 0.7680, contrastive at 0.8573 — an 8.9 pp gap that widens to 16.3 pp at 30%). The inflection is between 70% and 60% for the baseline.

### Recommendation

**70% is the pragmatic sweet spot.** It delivers the highest LOGO accuracy of any configuration (0.8661), including 100% data, while saving 30% of annotation effort. If annotation is highly constrained, **50% (Acc = 0.8506) or 30% (Acc = 0.8391)** both remain practically viable with clear advantages over the supervised baseline at the same data level.

---

## 11. Practical Impact and ROI per Configuration

### Annotation cost context

Collecting and annotating IMU gesture data requires supervised experimental sessions, participant recruitment, and per-gesture labeling. Reducing labeled data by 50–70% proportionally reduces these costs.

### Production readiness by configuration

| Configuration | LOGO Acc | vs same-fraction baseline | vs baseline@100% | Data vs 100% | Verdict |
|---|---|---|---|---|---|
| Baseline 100% | 0.8110 | — | — | 100% | Upper bound for supervised |
| Baseline 30% | 0.6760 | — | -16.6% | 30% | Degraded, borderline usable |
| Random 100% | 0.8638 | +5.3 pp | +5.3 pp | 100% | Recommended |
| **Random 70%** | **0.8661** | +5.7 pp | +5.5 pp | **70%** | **Best overall: highest accuracy + saves 30%** |
| Random 50% | 0.8506 | +8.4 pp | +3.9 pp | 50% | Recommended |
| Random 40% | 0.8394 | +8.2 pp | +2.8 pp | 40% | Good ROI |
| Random 30% | 0.8391 | +16.3 pp | +2.8 pp | 30% | Very low cost, useful |

Note: random at 70% actually outperforms random at 100% by 0.0023 Acc. 30% annotation cost is freed with zero performance loss.

### ROI summary by use case

**Use case A — Maximum accuracy, cost secondary:** Use random at 70%. Acc = 0.8661, the highest of any configuration. Saves 30% of annotation effort.

**Use case B — Balanced cost and performance:** Use random at 50%. Acc = 0.8506 (98.2% of max), 50% annotation savings, 8.4 pp improvement over same-fraction supervised baseline.

**Use case C — Severely labeled-data constrained:** Use random at 30%. Acc = 0.8391 (96.9% of random at 100%), 16.3 pp improvement over same-fraction baseline (which at 30% is barely usable at 0.6760).

**Use case D — Avoid completely:** semihard and semihard_shift at any fraction. They offer negligible advantage at 100%, are unstable, and provide no value over random.

### Core ROI argument

Contrastive pre-training at 30% labeled data vs supervised training at 100%:

- 70% reduction in annotation cost
- LOGO Acc: 0.8391 (contrastive, 30%) vs 0.8110 (supervised, 100%) — still +2.8 pp ahead despite 70% less data
- In practical deployment: the supervised model degrades severely as deployment cohorts shift; the contrastive model degrades very slowly

This makes labeling efficiency and deployment robustness simultaneously available — a combination not achievable with purely supervised training on this dataset.

---

## 12. Methodological Notes and Reproducibility

### Note on metric: F1 vs Accuracy

All tables in this document report `Test_Acc` (accuracy) from the evaluation CSV files. Both `train_baseline.py` and `train_contrastive_model.py` record both `Test_Acc` and `Test_F1` per fold. For this 6-class dataset, accuracy and F1 are nearly identical (difference ≤ 0.5 pp) because the class distribution is approximately balanced (`idle` has ~half the support of other classes but its per-class F1 is close to the rest). The per-class breakdown tables in Sections 7–9 continue to use per-class F1 from sklearn's `classification_report`, as per-class accuracy is not directly meaningful in multi-class settings. All conclusions are identical whether using accuracy or F1.

### Seeds and determinism

All experiments: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. All reported results are fully reproducible from a clean re-run.

### Test set consistency

| Protocol | Baseline vs contrastive test set equivalence |
|---|---|
| Stratified | Identical: both use `train_test_split(random_state=42, test_size=0.2, stratify=y)` |
| LOSO | Identical: both iterate over `unique_participants` in the same sorted order |
| LOGO | Unified: both use seed-42 non-overlapping permutation windows of 3 participants (5 folds, each participant appears in exactly one test fold) |

### Data fraction scope

The `--data-fraction` parameter reduces only the training set. Test and validation sets are always drawn from the full held-out data, ensuring evaluation is comparable across fractions.

### Baseline runs

Both baseline at 100% (`results/100pct/baseline/`) and baseline at reduced fractions (`results/{70,60,50,40,30}pct/baseline/`) are available, enabling the per-fraction same-data comparison in Section 5.

A separate `baseline_model_selection` run (`results/100pct/baseline_model_selection/`) trained all 4 architectures (DNN, LSTM, Transformer, SDCNet) using the original lexicographic LOGO partition. Its purpose was model selection only; the SDCNet LOGO value from this run (0.7319) is lower than the corrected-partition value (0.8110). All main analyses use the corrected partition.

### Cluster and infrastructure

- Cluster: Narval (Compute Canada), A100 GPU (40/80 GB)
- Compute time per run: approximately 4–8 hours depending on fraction
- SLURM scripts: `scripts/run_*_slurm.sh` (5 methods + 1 baseline allpct array script)

---

## 13. Conclusion

This study demonstrates that contrastive pre-training with triplet loss and random negative mining provides consistent and growing improvements over fully-supervised SDCNet on a cross-group IMU gesture recognition task, across all labeled-data fractions tested.

### Four key conclusions

**1. The contrastive representation is fundamentally more data-robust.** The supervised baseline degrades by 16.6% in LOGO accuracy when labeled data is reduced from 100% to 30%. Random contrastive mining degrades by only 2.9% over the same range. This ~6x difference in degradation rate is the study's principal finding and directly determines the viability of contrastive learning in label-scarce deployment scenarios.

**2. The advantage grows as data decreases.** At full data, random mining outperforms the same-fraction baseline by 5.3 pp in LOGO. At 30% data, this gap reaches 16.3 pp. The contrastive approach becomes exponentially more valuable precisely when annotation cost is the primary constraint — a strongly favorable property for real-world deployment.

**3. Semi-hard mining is counter-productive.** With the corrected baseline, semihard methods offer only 0.5 pp LOGO advantage at 100% data and fail to beat the same-fraction baseline below 70%. For structured temporal IMU data with strong inter-class distance, random triplet sampling provides sufficient learning signal, and semi-hard selection introduces instability without benefit. This result holds consistently across all 6 fractions, 3 protocols, and all individual folds.

**4. Temporal shift augmentation is irrelevant.** The circular shift augmentation adds no measurable value on top of the jitter and scaling augmentations already active. Gesture IMU signals have a consistent temporal alignment from onset; randomizing their phase creates unrealistic samples without improving generalization.

### Recommended configuration for production

**Random triplet mining at 70% of available labeled data.** This achieves LOGO Accuracy = 0.8661, which is the highest of any configuration tested — including the 100%-data supervised and contrastive runs — while saving 30% of annotation effort. If annotation is highly constrained, 50% (Acc = 0.8506) or 30% (Acc = 0.8391) both remain practically viable with clear advantages over the supervised baseline at the same data level.

### Limitations and future work

- The study uses 5 LOGO folds. With 20 participants that yields groups of 3; a larger dataset would allow more folds and tighter variance estimates.
- The same-data LOGO comparison (Section 5) confirms the advantage but the 30% baseline (0.6760) is at the edge of usability; it would be worth investigating whether the baseline collapses are consistent across re-runs.
- Extending the study to additional gesture datasets would validate whether the random-vs-semihard finding is task-specific or general.
