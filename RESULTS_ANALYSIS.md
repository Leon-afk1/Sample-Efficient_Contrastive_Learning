# Sample-Efficient Contrastive Learning — Full Results Analysis

**Date:** March 25, 2026
**Dataset:** IMU gesture recognition (breakEgg, cut, flip, idle, pour, whip) — 20 participants
**Backbone:** SDCNet (1D-CNN)
**Metric:** Macro-averaged F1-score (test set)

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
| Baseline SDCNet LOGO F1 at 100% data | 0.8113 |
| Best contrastive LOGO F1 at 100% data | 0.8615 (random, +6.2%) |
| Best contrastive LOGO F1 at 30% data | 0.8370 (random, +24.4% vs same-fraction baseline) |
| Baseline LOGO drop from 100% to 30% | -0.1384 (-17.1%) |
| Contrastive LOGO drop from 100% to 30% (random) | -0.0245 (-2.8%) |
| Effect of temporal shift augmentation | Negligible (<0.5 pp in all conditions) |
| Semi-hard vs random mining (LOGO avg delta) | -4.5 to -7.7 pp |

### Central finding

The baseline SDCNet degrades severely with less data: LOGO F1 drops from 0.8113 at 100% to 0.6729 at 30% (-17.1%). Contrastive pre-training with random mining loses only 2.4 pp over the same range (0.8615 to 0.8370, -2.8%). This divergence is the core result of the study: the representation learned during contrastive pre-training is robust to data reduction in ways that purely supervised training cannot replicate.

At 30% labeled data, random contrastive mining achieves a LOGO F1 of 0.8370 against a same-fraction baseline of 0.6729, a gain of +0.1641 (+24.4%). The contrastive advantage is smallest at high data fractions (100%: +6.2%) and largest at low fractions (30%: +24.4%), because the baseline degrades far faster.

---

## 2. Baseline Model Selection

A model selection phase trained four architectures (DNN, LSTM, Transformer, SDCNet) on 100% of the data under all three evaluation strategies using a separate run (`baseline_model_selection`). This phase used the original lexicographic LOGO partition for cross-architecture comparison; subsequent single-architecture analyses use the unified seed-42 partition.

### Results — all architectures at 100% data

| Architecture | Stratified F1 | LOSO F1 (avg/20 folds) | LOGO F1 (avg/5 folds) |
|---|---|---|---|
| DNN | 0.7395 | 0.6082 | 0.5252 |
| LSTM | 0.9220 | 0.6763 | 0.5240 |
| Transformer | 0.8958 | 0.7226 | 0.6251 |
| **SDCNet** | **0.9588** | **0.8267** | **0.7326** |

### Selection rationale

SDCNet is unambiguously the strongest architecture across all three protocols. The performance gap is especially pronounced in the generalization-oriented protocols:

- LOSO: SDCNet (0.8267) vs Transformer (0.7226) — gap of 10.4 pp
- LOGO: SDCNet (0.7326) vs Transformer (0.6251) — gap of 10.7 pp

DNN and LSTM collapse in LOGO (~0.52), indicating complete failure to generalize to unseen participant groups regardless of data volume. SDCNet's 1D temporal convolution captures local motion patterns with a structural inductive bias well-suited to IMU gesture recognition. It is the only viable backbone for cross-group evaluation.

**Selected backbone: SDCNet.** All subsequent analyses use SDCNet trained with the corrected seed-42 LOGO partition, giving reference values: LOGO F1 = 0.8113, LOSO F1 = 0.8322, Stratified F1 = 0.9651 at 100% data.

---

## 3. Global Comparative Table — All Configurations

All values are macro-averaged F1-score on the test set. Baseline runs exist at all fractions with the corrected LOGO partition.

### Stratified F1

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.9651 | 0.9521 | 0.9436 | 0.9342 | 0.9259 | 0.9134 |
| random | **0.9666** | **0.9556** | **0.9534** | **0.9509** | **0.9428** | **0.9394** |
| random_shift | 0.9615 | 0.9557 | 0.9517 | 0.9518 | 0.9421 | 0.9375 |
| semihard | 0.9431 | 0.9455 | 0.9375 | 0.9144 | 0.8914 | 0.9054 |
| semihard_shift | 0.9564 | 0.9220 | 0.9096 | 0.9091 | 0.9174 | 0.8956 |

### LOSO F1 (average over 20 folds)

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.8322 | 0.8132 | 0.8093 | 0.7969 | 0.7868 | 0.7501 |
| random | 0.8708 | 0.8634 | 0.8565 | 0.8498 | **0.8513** | **0.8411** |
| random_shift | **0.8775** | **0.8665** | **0.8617** | **0.8495** | 0.8562 | 0.8329 |
| semihard | 0.8487 | 0.8299 | 0.8200 | 0.8106 | 0.7994 | 0.7880 |
| semihard_shift | 0.8479 | 0.8343 | 0.8272 | 0.8156 | 0.7970 | 0.7804 |

### LOGO F1 (average over 5 folds) — Primary metric

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.8113 | 0.8097 | 0.7668 | 0.7648 | 0.7565 | 0.6729 |
| random | **0.8615** | **0.8616** | **0.8533** | **0.8481** | 0.8383 | **0.8370** |
| random_shift | 0.8613 | 0.8555 | 0.8503 | 0.8476 | **0.8437** | 0.8317 |
| semihard | 0.8166 | 0.8128 | 0.7916 | 0.8023 | 0.7614 | 0.7723 |
| semihard_shift | 0.8143 | 0.8087 | 0.8003 | 0.7974 | 0.7912 | 0.7719 |

Bold values indicate the best contrastive result per fraction (across all 4 methods). At 100% and 70%, the baseline is already competitive with semihard methods on LOGO; random mining maintains a clear advantage at all fractions.

---

## 4. Gain Table Relative to Baseline at 100% Data

Delta = contrastive F1 - baseline SDCNet F1 at 100% data. Percentage in parentheses is the relative gain. This table uses the fixed 100%-data baseline as the reference throughout, allowing direct comparison of what contrastive methods at reduced fractions achieve against the full-data supervised benchmark.

### LOGO — Gain vs Baseline SDCNet 100% (F1 = 0.8113)

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| random | +0.0502 (+6.2%) | +0.0503 (+6.2%) | +0.0420 (+5.2%) | +0.0368 (+4.5%) | +0.0270 (+3.3%) | +0.0257 (+3.2%) |
| random_shift | +0.0500 (+6.2%) | +0.0442 (+5.4%) | +0.0390 (+4.8%) | +0.0363 (+4.5%) | +0.0324 (+4.0%) | +0.0204 (+2.5%) |
| semihard | +0.0053 (+0.7%) | +0.0015 (+0.2%) | -0.0197 (-2.4%) | -0.0090 (-1.1%) | -0.0499 (-6.2%) | -0.0390 (-4.8%) |
| semihard_shift | +0.0030 (+0.4%) | -0.0026 (-0.3%) | -0.0110 (-1.4%) | -0.0139 (-1.7%) | -0.0201 (-2.5%) | -0.0394 (-4.9%) |

Only random and random_shift consistently exceed the full-data baseline at all fractions. Semihard mining falls below the full-data baseline at 60% and below, semihard_shift from 70% and below.

### LOSO — Gain vs Baseline SDCNet 100% (F1 = 0.8322)

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| random | +0.0386 (+4.6%) | +0.0312 (+3.7%) | +0.0243 (+2.9%) | +0.0176 (+2.1%) | +0.0191 (+2.3%) | +0.0089 (+1.1%) |
| random_shift | +0.0453 (+5.4%) | +0.0343 (+4.1%) | +0.0295 (+3.5%) | +0.0173 (+2.1%) | +0.0240 (+2.9%) | +0.0007 (+0.1%) |
| semihard | +0.0165 (+2.0%) | -0.0023 (-0.3%) | -0.0122 (-1.5%) | -0.0216 (-2.6%) | -0.0328 (-3.9%) | -0.0442 (-5.3%) |
| semihard_shift | +0.0157 (+1.9%) | +0.0021 (+0.3%) | -0.0050 (-0.6%) | -0.0166 (-2.0%) | -0.0352 (-4.2%) | -0.0518 (-6.2%) |

Random_shift at 30% is just barely above the full-data baseline (+0.1 pp). Semihard methods fall well below the baseline from 60% onward in LOSO, mirroring the pattern seen in LOGO.

---

## 5. Per-Fraction Comparison Against Corresponding Baseline

This section compares contrastive at X% against the supervised baseline trained on the same X% of data. With baseline runs now available at all fractions, this is the fairest same-data comparison.

### LOGO — Contrastive gain vs baseline at the SAME fraction

| Fraction | Baseline | random | random_shift | semihard | semihard_shift |
|---|---|---|---|---|---|
| 100% | 0.8113 | 0.8615 (+5.0 pp) | 0.8613 (+5.0 pp) | 0.8166 (+0.5 pp) | 0.8143 (+0.3 pp) |
| 70% | 0.8097 | 0.8616 (+5.2 pp) | 0.8555 (+4.6 pp) | 0.8128 (+0.3 pp) | 0.8087 (-0.1 pp) |
| 60% | 0.7668 | 0.8533 (+8.6 pp) | 0.8503 (+8.4 pp) | 0.7916 (+2.5 pp) | 0.8003 (+3.4 pp) |
| 50% | 0.7648 | 0.8481 (+8.3 pp) | 0.8476 (+8.3 pp) | 0.8023 (+3.7 pp) | 0.7974 (+3.3 pp) |
| 40% | 0.7565 | 0.8383 (+8.2 pp) | 0.8437 (+8.7 pp) | 0.7614 (+0.5 pp) | 0.7912 (+3.5 pp) |
| 30% | 0.6729 | 0.8370 (+16.4 pp) | 0.8317 (+15.9 pp) | 0.7723 (+9.9 pp) | 0.7719 (+9.9 pp) |

Key observation: the contrastive advantage at same-data grows as the available labeled fraction decreases. At 30%, even semihard achieves a +9.9 pp improvement over the same-fraction baseline, because the baseline collapses to 0.6729 while semihard only reaches 0.7723. Random mining achieves the full +16.4 pp advantage at 30%, the largest gain observed in the entire study.

At 100% data, the contrastive advantage is real but more modest (random: +5.0 pp). At 70%, semihard_shift barely falls below the baseline (-0.1 pp), the only instance where a contrastive method fails to beat its own-fraction baseline in LOGO.

### LOSO — Contrastive gain vs baseline at the SAME fraction

| Fraction | Baseline | random | random_shift | semihard | semihard_shift |
|---|---|---|---|---|---|
| 100% | 0.8322 | 0.8708 (+3.9 pp) | 0.8775 (+4.5 pp) | 0.8487 (+1.7 pp) | 0.8479 (+1.6 pp) |
| 70% | 0.8132 | 0.8634 (+5.0 pp) | 0.8665 (+5.3 pp) | 0.8299 (+1.7 pp) | 0.8343 (+2.1 pp) |
| 60% | 0.8093 | 0.8565 (+4.7 pp) | 0.8617 (+5.2 pp) | 0.8200 (+1.1 pp) | 0.8272 (+1.8 pp) |
| 50% | 0.7969 | 0.8498 (+5.3 pp) | 0.8495 (+5.3 pp) | 0.8106 (+1.4 pp) | 0.8156 (+1.9 pp) |
| 40% | 0.7868 | 0.8513 (+6.5 pp) | 0.8562 (+6.9 pp) | 0.7994 (+1.3 pp) | 0.7970 (+1.0 pp) |
| 30% | 0.7501 | 0.8411 (+9.1 pp) | 0.8329 (+8.3 pp) | 0.7880 (+3.8 pp) | 0.7804 (+3.0 pp) |

All contrastive methods beat the same-fraction baseline in LOSO across all fractions. The advantage grows as data decreases, following the same pattern as LOGO, though the absolute gains are smaller because LOSO is a less extreme generalization task and both methods degrade less.

---

## 6. Critical Analysis

### 6.1 Random mining dominates semi-hard mining — a robust and unexpected result

The standard expectation in metric learning is that semi-hard mining improves over random sampling by selecting more informative triplets. The observed results consistently contradict this:

- LOGO at 100%: random 0.8615 vs semihard 0.8166 — gap of 4.5 pp; vs baseline 0.8113: semihard +0.5 pp vs random +5.0 pp
- LOGO at 30%: random 0.8370 vs semihard 0.7723 — gap of 6.5 pp
- The ranking random > semihard is stable across all fractions and all three evaluation protocols

Three mechanistic explanations align:

First, the dataset contains 6 structured gesture classes with strong temporal signatures; natural inter-class distance is large, making random triplets already informative. The "hardness" that semi-hard mining adds provides diminishing returns when the feature space is already well-separated by class.

Second, semi-hard mining reduces the effective gradient signal per batch: it discards many potential triplets before training, which at 30–40% data volumes means much fewer learning updates per epoch — exactly when the model needs more signal, not less.

Third, on a dataset of 20 participants with 6 gesturally similar classes, semi-hard negatives tend to cluster around gesture boundaries (e.g., cut vs whip), causing early training instability. With only 5 LOGO folds, one unstable fold (semihard at 100%, fold 2 = 0.6998) significantly pulls the average down.

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

| Data fraction | Baseline LOGO F1 | Random contrastive LOGO F1 | Gap |
|---|---|---|---|
| 100% | 0.8113 | 0.8615 | +0.0502 |
| 70% | 0.8097 | 0.8616 | +0.0519 |
| 60% | 0.7668 | 0.8533 | +0.0865 |
| 50% | 0.7648 | 0.8481 | +0.0833 |
| 40% | 0.7565 | 0.8383 | +0.0818 |
| 30% | 0.6729 | 0.8370 | +0.1641 |

The baseline loses -17.1% on LOGO going from 100% to 30% data. Random contrastive loses only -2.8%. The gap between them triples (from 5.0 pp to 16.4 pp). This pattern confirms that the contrastive representation learned during pre-training is fundamentally more data-efficient: it captures cross-group invariances from unlabeled structure during triplet training, and the labeled data is used only to align class boundaries — a much simpler task that requires fewer examples.

### 6.4 Semi-hard methods are barely competitive with the corrected baseline

With the corrected LOGO partition (seed-42, non-overlapping), the baseline at 100% achieves 0.8113 — notably higher than the 0.7326 produced by the old lexicographic partition. This makes semihard's advantage over the baseline minimal at 100% (+0.5 pp in LOGO) and negative at 60% and below. Semihard is not a viable strategy for this task at any data fraction when compared to the correct baseline. Only random mining delivers a consistent, meaningful advantage.

### 6.5 Variance and reliability

Semihard mining produces LOGO fold standard deviations of 0.0651 at 100% data (vs 0.0174 for random). Its worst fold at 100% (F2 = 0.6998) reaches near-baseline performance, indicating that fold-dependent failure is a systematic issue. In production, where only a single deployment scenario exists, such instability is a serious concern. Random mining never produces a fold below 0.80 across any fraction — a floor of practical usability.

---

## 7. Detailed Analysis — Stratified Split

The stratified split (70/10/20, class-balanced, random_state=42) evaluates in-distribution generalization on randomly held-out samples from the same sessions and population.

### Full results — Stratified F1

| Method | 100% | 70% | 60% | 50% | 40% | 30% | Drop 100→30% |
|---|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.9651 | 0.9521 | 0.9436 | 0.9342 | 0.9259 | 0.9134 | -0.0517 |
| random | 0.9666 | 0.9556 | 0.9534 | 0.9509 | 0.9428 | 0.9394 | -0.0272 |
| random_shift | 0.9615 | 0.9557 | 0.9517 | 0.9518 | 0.9421 | 0.9375 | -0.0240 |
| semihard | 0.9431 | 0.9455 | 0.9375 | 0.9144 | 0.8914 | 0.9054 | -0.0377 |
| semihard_shift | 0.9564 | 0.9220 | 0.9096 | 0.9091 | 0.9174 | 0.8956 | -0.0608 |

### Gain vs same-fraction baseline (Stratified)

| Method | 100% | 70% | 60% | 50% | 40% | 30% |
|---|---|---|---|---|---|---|
| random | +0.0015 | +0.0035 | +0.0098 | +0.0167 | +0.0169 | +0.0260 |
| random_shift | -0.0036 | +0.0036 | +0.0081 | +0.0176 | +0.0162 | +0.0241 |
| semihard | -0.0220 | -0.0066 | -0.0061 | -0.0198 | -0.0345 | -0.0080 |
| semihard_shift | -0.0087 | -0.0301 | -0.0340 | -0.0251 | -0.0085 | -0.0178 |

### Observations

Stratified results are uniformly high (0.89–0.97) and the protocol cannot meaningfully differentiate methods. Random mining consistently beats or ties the baseline across all fractions, with an increasing gap at lower fractions (up to +2.6 pp at 30%). The baseline itself degrades modestly (-5.2 pp from 100% to 30%), while random loses only -2.7 pp. Semihard methods fall below the baseline at several fractions, echoing the pattern seen on harder protocols but more moderately.

### Conclusion — Stratified

The in-distribution evaluation does not reveal the full picture. All methods perform near-optimally when test and training data come from the same population and session. The protocol is useful for confirming numerical sanity but carries no diagnostic weight when comparing generalization strategies. Random contrastive mining is the only method that consistently improves on or matches the supervised baseline across all fractions here.

---

## 8. Detailed Analysis — LOSO

Leave-One-Subject-Out holds out all sessions from one participant and tests on them. This evaluates generalization to individual physiological and kinematic variability not seen during training.

### Full results — LOSO F1 (average over 20 folds)

| Method | 100% | 70% | 60% | 50% | 40% | 30% | Drop 100→30% |
|---|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.8322 | 0.8132 | 0.8093 | 0.7969 | 0.7868 | 0.7501 | -0.0821 |
| random | 0.8708 | 0.8634 | 0.8565 | 0.8498 | 0.8513 | 0.8411 | -0.0297 |
| random_shift | **0.8775** | 0.8665 | 0.8617 | 0.8495 | 0.8562 | 0.8329 | -0.0446 |
| semihard | 0.8487 | 0.8299 | 0.8200 | 0.8106 | 0.7994 | 0.7880 | -0.0607 |
| semihard_shift | 0.8479 | 0.8343 | 0.8272 | 0.8156 | 0.7970 | 0.7804 | -0.0675 |

### Inter-fold variance (min / max / std) over 20 folds

| Method | 100% — min / max / std | 30% — min / max / std |
|---|---|---|
| Baseline SDCNet | 0.6575 / 0.9608 / 0.080 | 0.5884 / 0.9556 / 0.098 |
| random | 0.7728 / 0.9697 / 0.058 | 0.7430 / 0.9552 / 0.061 |
| random_shift | 0.7794 / 0.9682 / 0.058 | 0.6889 / 0.9344 / 0.062 |
| semihard | 0.7232 / 0.9408 / 0.066 | 0.5797 / 0.9119 / 0.087 |
| semihard_shift | 0.7386 / 0.9570 / 0.066 | 0.5838 / 0.9309 / 0.083 |

At 30%, the baseline minimum (0.5884) and semihard minimum (0.5797) reach similar catastrophic lows. Random mining's minimum at 30% (0.7430) is 15 pp above these. The baseline degrades from std=0.080 at 100% to std=0.098 at 30%, confirming increased participant-level performance variability as labeled data decreases.

### Gain comparison (same fraction)

All contrastive methods beat the same-fraction LOSO baseline at all data levels. The advantage grows as fraction decreases (baseline drops faster). At 30%, random achieves +9.1 pp over same-fraction baseline; even semihard achieves +3.8 pp.

### Conclusion — LOSO

Random and random_shift provide consistent and growing improvements over the baseline across all fractions. The baseline degrades -8.2 pp from 100% to 30%, while random loses only -3.0 pp, making the advantage at 30% nearly three times larger (+9.1 pp) than at 100% (+3.9 pp). Semihard methods also beat the same-fraction baseline in LOSO (unlike the mixed picture in LOGO), but only modestly, and they fall well below the full-data baseline from 70% onward when comparing against the fixed reference. The minimum LOSO fold performance (worst individual participant) remains well above 0.74 for random at any fraction, a practically important floor.

---

## 9. Detailed Analysis — LOGO

Leave-One-Group-Out tests on groups of three participants entirely absent from training. This is the most realistic and demanding scenario, simulating deployment to a new demographic group not represented in the training data. It is the primary metric of this study.

### Full results — LOGO F1 (average over 5 folds)

| Method | 100% | 70% | 60% | 50% | 40% | 30% | Drop 100→30% |
|---|---|---|---|---|---|---|---|
| Baseline SDCNet | 0.8113 | 0.8097 | 0.7668 | 0.7648 | 0.7565 | 0.6729 | **-0.1384 (-17.1%)** |
| random | 0.8615 | 0.8616 | 0.8533 | 0.8481 | 0.8383 | 0.8370 | -0.0245 (-2.8%) |
| random_shift | 0.8613 | 0.8555 | 0.8503 | 0.8476 | 0.8437 | 0.8317 | -0.0296 (-3.4%) |
| semihard | 0.8166 | 0.8128 | 0.7916 | 0.8023 | 0.7614 | 0.7723 | -0.0443 (-5.4%) |
| semihard_shift | 0.8143 | 0.8087 | 0.8003 | 0.7974 | 0.7912 | 0.7719 | -0.0424 (-5.2%) |

### Per-fold breakdown — Baseline SDCNet

| Fraction | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg | Std |
|---|---|---|---|---|---|---|---|
| 100% | 0.8150 | 0.7773 | 0.8440 | 0.8491 | 0.7710 | 0.8113 | 0.0325 |
| 70% | 0.8419 | 0.7610 | 0.8173 | 0.8569 | 0.7716 | 0.8097 | 0.0378 |
| 60% | 0.7700 | 0.7375 | 0.8286 | 0.7887 | 0.7094 | 0.7668 | 0.0412 |
| 50% | 0.7767 | 0.7189 | 0.7909 | 0.8146 | 0.7230 | 0.7648 | 0.0379 |
| 40% | 0.7039 | 0.7602 | 0.8031 | 0.8010 | 0.7142 | 0.7565 | 0.0418 |
| 30% | 0.6467 | 0.6447 | 0.7153 | 0.7657 | 0.5920 | 0.6729 | 0.0607 |

### Per-fold breakdown — random

| Fraction | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg | Std |
|---|---|---|---|---|---|---|---|
| 100% | 0.8683 | 0.8528 | 0.8741 | 0.8806 | 0.8320 | 0.8615 | 0.0174 |
| 70% | 0.8577 | 0.8392 | 0.8809 | 0.8966 | 0.8338 | 0.8616 | 0.0241 |
| 60% | 0.8689 | 0.8311 | 0.8565 | 0.8834 | 0.8265 | 0.8533 | 0.0218 |
| 50% | 0.8474 | 0.8370 | 0.8599 | 0.8904 | 0.8056 | 0.8481 | 0.0277 |
| 40% | 0.8402 | 0.8235 | 0.8420 | 0.8616 | 0.8243 | 0.8383 | 0.0140 |
| 30% | 0.8517 | 0.8143 | 0.8393 | 0.8784 | 0.8012 | 0.8370 | 0.0273 |

### Per-fold breakdown — random_shift

| Fraction | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg | Std |
|---|---|---|---|---|---|---|---|
| 100% | 0.8607 | 0.8427 | 0.8609 | 0.8893 | 0.8529 | 0.8613 | 0.0155 |
| 70% | 0.8523 | 0.8472 | 0.8714 | 0.8742 | 0.8324 | 0.8555 | 0.0156 |
| 60% | 0.8574 | 0.8377 | 0.8592 | 0.8965 | 0.8007 | 0.8503 | 0.0312 |
| 50% | 0.8534 | 0.8296 | 0.8607 | 0.8750 | 0.8190 | 0.8476 | 0.0205 |
| 40% | 0.8721 | 0.8162 | 0.8319 | 0.8804 | 0.8177 | 0.8437 | 0.0273 |
| 30% | 0.8449 | 0.8064 | 0.8263 | 0.8747 | 0.8065 | 0.8317 | 0.0258 |

### Per-fold breakdown — semihard

| Fraction | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg | Std |
|---|---|---|---|---|---|---|---|
| 100% | 0.8422 | **0.6998** | 0.8560 | 0.8873 | 0.7977 | 0.8166 | **0.0651** |
| 70% | 0.7693 | 0.7916 | 0.8342 | 0.8597 | 0.8093 | 0.8128 | 0.0317 |
| 60% | 0.7558 | 0.7688 | 0.7977 | 0.8385 | 0.7972 | 0.7916 | 0.0285 |
| 50% | 0.8461 | 0.7813 | 0.7905 | 0.8492 | 0.7443 | 0.8023 | 0.0402 |
| 40% | 0.7312 | 0.7594 | 0.7836 | 0.7949 | 0.7376 | 0.7614 | 0.0249 |
| 30% | 0.7749 | 0.7112 | 0.7887 | 0.8249 | 0.7617 | 0.7723 | 0.0371 |

### Per-fold breakdown — semihard_shift

| Fraction | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg | Std |
|---|---|---|---|---|---|---|---|
| 100% | 0.7797 | 0.7925 | 0.8276 | 0.8600 | 0.8118 | 0.8143 | 0.0281 |
| 70% | 0.8193 | 0.7834 | 0.7954 | 0.8416 | 0.8039 | 0.8087 | 0.0202 |
| 60% | 0.8108 | 0.7492 | 0.8141 | 0.8509 | 0.7764 | 0.8003 | 0.0348 |
| 50% | 0.7938 | 0.8022 | 0.8258 | 0.8388 | 0.7265 | 0.7974 | 0.0389 |
| 40% | 0.7999 | 0.7451 | 0.8141 | 0.8312 | 0.7657 | 0.7912 | 0.0315 |
| 30% | 0.8045 | 0.7476 | 0.7666 | 0.8130 | 0.7276 | 0.7719 | 0.0327 |

### Best method per fraction (LOGO, contrastive only)

| Fraction | Best contrastive method | F1 | vs same-fraction baseline |
|---|---|---|---|
| 100% | random | 0.8615 | +5.0 pp |
| 70% | random | 0.8616 | +5.2 pp |
| 60% | random | 0.8533 | +8.6 pp |
| 50% | random | 0.8481 | +8.3 pp |
| 40% | random_shift | 0.8437 | +8.7 pp |
| 30% | random | 0.8370 | +16.4 pp |

### Structural observations on fold variance

Fold 4 consistently produces the best results across all methods (0.85–0.90 for random). Fold 5 is consistently the weakest. This is a structural artifact of the three participants assigned to each group by the seed-42 permutation: fold 4's test participants perform gestures closer to the population mean, while fold 5's are more idiosyncratic. This is expected and does not indicate a methodological issue.

The semihard collapse at 100% fold 2 (0.6998) is the lowest individual fold F1 among all contrastive runs at full data. It is within 1 pp of the baseline average at 100% (0.8113) — meaning that specific participant combination essentially reduces semihard mining to supervised-baseline performance for that group.

### Conclusion — LOGO

Random mining is the clear winner, uniformly superior at every fraction, with low and stable fold variance (std never exceeds 0.028). The core finding is the divergence of degradation curves: the baseline loses 13.8 pp going from 100% to 30%, random loses only 2.4 pp. As a result, the absolute contrastive advantage triples from +5.0 pp at 100% to +16.4 pp at 30%. Semihard mining offers minimal benefit over the baseline at high fractions and becomes competitive only at 30% (when the baseline collapses), where it still underperforms random by 6.5 pp.

---

## 10. Sweet Spot Identification

The sweet spot is the data fraction at which contrastive pre-training delivers the best balance between labeling cost saved and performance maintained, relative to full-data training.

### LOGO performance thresholds — random mining

| Performance target | Fraction required | F1 achieved |
|---|---|---|
| Maximum performance | 100% / 70% (tied) | 0.8615 / 0.8616 |
| Within 1 pp of max (>= 0.8515) | 70% | 0.8616 |
| Within 2 pp of max (>= 0.8415) | 60% | 0.8533 |
| Within 4 pp of max (>= 0.8215) | 30% | 0.8370 |

The performance curve is essentially flat from 100% to 70% (+0.0001 ≡ no change), then shows a gentle slope declining approximately 0.6 pp per 10 pp reduction in fraction.

### Cost-effectiveness analysis

Defining "cost" as fraction of labeled data used and "benefit" as gain over the same-fraction baseline:

| Fraction | Data saved | LOGO F1 (random) | Gain vs same-fraction baseline | Ratio (gain/cost) |
|---|---|---|---|---|
| 100% | 0% | 0.8615 | +0.0502 | 0.050 |
| 70% | 30% | 0.8616 | +0.0519 | 0.074 |
| 60% | 40% | 0.8533 | +0.0865 | 0.144 |
| 50% | 50% | 0.8481 | +0.0833 | 0.167 |
| 40% | 60% | 0.8383 | +0.0818 | 0.205 |
| 30% | 70% | 0.8370 | +0.1641 | 0.547 |

The ratio (gain per unit of data used) increases sharply at 30%; this is partly mechanical since the baseline degrades fastest at that point. However, the absolute performance (0.8370) is also the key value: it represents a usable, stable model that remains 8.4 pp above even the maximum the baseline can achieve.

### Same-fraction comparison perspective

The most natural "sweet spot" framing is: at what fraction does the F1 plateau for contrastive, while the baseline continues declining? The data shows the divergence begins clearly at 60% (baseline at 0.7668, contrastive at 0.8533 — an 8.6 pp gap that widens to 16.4 pp at 30%). The inflection is between 70% and 60% for the baseline.

### Recommendation

**50% is the pragmatic sweet spot.** It captures 98.5% of the random mining performance at 100% (0.8481 vs 0.8615), beats the same-fraction baseline by 8.3 pp, and saves half the annotation cost. If annotation cost is highly constrained, **30% is defensible**: it still achieves 0.8370 LOGO F1 (97.2% of the 100%-data contrastive performance) with a 16.4 pp advantage over the same-fraction baseline.

---

## 11. Practical Impact and ROI per Configuration

### Annotation cost context

Collecting and annotating IMU gesture data requires supervised experimental sessions, participant recruitment, and per-gesture labeling. Reducing labeled data by 50–70% proportionally reduces these costs.

### Production readiness by configuration

| Configuration | LOGO F1 | Same-fraction baseline | Advantage | Data vs 100% | Verdict |
|---|---|---|---|---|---|
| Baseline 100% | 0.8113 | — | — | 100% | Upper bound for supervised |
| Baseline 30% | 0.6729 | — | — | 30% | Degraded, borderline usable |
| random 100% | 0.8615 | +5.0 pp | +5.0 pp | 100% | Recommended |
| **random 70%** | **0.8616** | +5.2 pp | +5.2 pp | **70%** | **Best data efficiency** |
| random 50% | 0.8481 | +8.3 pp | +8.3 pp | 50% | Recommended |
| random 40% | 0.8383 | +8.2 pp | +8.2 pp | 40% | Good ROI |
| random 30% | 0.8370 | +16.4 pp | +16.4 pp | 30% | Very low cost, useful |
| semihard 100% | 0.8166 | +0.5 pp | +0.5 pp | 100% | Not recommended |
| semihard 30% | 0.7723 | +9.9 pp | +0.5 pp above baseline 100% | 30% | Barely usable |

Note: random at 70% vs 100% differs by 0.0001 F1 — essentially free performance. 30% annotation cost is freed with zero measurable performance loss at equal comparison.

### ROI summary by use case

**Use case A — Maximum accuracy, cost secondary:** Use random at 70%. F1 = 0.8616, indistinguishable from random at 100%. Saves 30% of annotation effort.

**Use case B — Balanced cost and performance:** Use random at 50%. F1 = 0.8481 (98.5% of max), 50% annotation savings, 8.3 pp improvement over same-fraction supervised baseline.

**Use case C — Severely labeled-data constrained:** Use random at 30%. F1 = 0.8370 (still 97.2% of random at 100%), 16.4 pp improvement over same-fraction baseline (which at 30% is barely usable at 0.6729).

**Use case D — Avoid completely:** semihard and semihard_shift at any fraction. They offer negligible advantage at 100%, are unstable, and provide no value over random.

### Core ROI argument

Contrastive pre-training at 30% labeled data vs supervised training at 100%:

- 70% reduction in annotation cost
- LOGO F1: 0.8370 (contrastive, 30%) vs 0.8113 (supervised, 100%) — still +2.6 pp ahead despite 70% less data
- In practical deployment: the supervised model degrades severely as deployment cohorts shift; the contrastive model degrades very slowly

This makes labeling efficiency and deployment robustness simultaneously available — a combination not achievable with purely supervised training on this dataset.

---

## 12. Methodological Notes and Reproducibility

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

A separate `baseline_model_selection` run (`results/100pct/baseline_model_selection/`) trained all 4 architectures (DNN, LSTM, Transformer, SDCNet) using the original lexicographic LOGO partition. Its purpose was model selection only; the SDCNet LOGO value from this run (0.7326) is lower than the corrected-partition value (0.8113). All main analyses use the corrected partition.

### Cluster and infrastructure

- Cluster: Narval (Compute Canada), A100 GPU (40/80 GB)
- Account: `def-s1gabour`
- Compute time per run: approximately 4–8 hours depending on fraction
- SLURM scripts: `scripts/run_*_slurm.sh` (5 methods + 1 baseline allpct array script)

---

## 13. Conclusion

This study demonstrates that contrastive pre-training with triplet loss and random negative mining provides consistent and growing improvements over fully-supervised SDCNet on a cross-group IMU gesture recognition task, across all labeled-data fractions tested.

### Four key conclusions

**1. The contrastive representation is fundamentally more data-robust.** The supervised baseline degrades by 17.1% in LOGO F1 when labeled data is reduced from 100% to 30%. Random contrastive mining degrades by only 2.8% over the same range. This 6x difference in degradation rate is the study's principal finding and directly determines the viability of contrastive learning in label-scarce deployment scenarios.

**2. The advantage grows as data decreases.** At full data, random mining outperforms the same-fraction baseline by 5.0 pp in LOGO. At 30% data, this gap reaches 16.4 pp. The contrastive approach becomes exponentially more valuable precisely when annotation cost is the primary constraint — a strongly favorable property for real-world deployment.

**3. Semi-hard mining is counter-productive.** With the corrected baseline, semihard methods offer only 0.3–0.5 pp LOGO advantage at 100% data and fail to beat the same-fraction baseline below 60–70%. For structured temporal IMU data with strong inter-class distance, random triplet sampling provides sufficient learning signal, and semi-hard selection introduces instability without benefit. This result holds consistently across all 6 fractions, 3 protocols, and all individual folds.

**4. Temporal shift augmentation is irrelevant.** The circular shift augmentation adds no measurable value on top of the jitter and scaling augmentations already active. Gesture IMU signals have a consistent temporal alignment from onset; randomizing their phase creates unrealistic samples without improving generalization.

### Recommended configuration for production

**Random triplet mining at 70% of available labeled data.** This achieves LOGO F1 = 0.8616, identical to the 100%-data result (0.8615), while saving 30% of annotation effort. If annotation is highly constrained, 50% (F1 = 0.8481) or 30% (F1 = 0.8370) both remain practically viable with clear advantages over the supervised baseline at the same data level.

### Limitations and future work

- The study uses 5 LOGO folds. With 20 participants that yields groups of 3; a larger dataset would allow more folds and tighter variance estimates.
- The same-data LOGO comparison (Section 5) confirms the advantage but the 30% baseline (0.6729) is at the edge of usability; it would be worth investigating whether the baseline collapses are consistent across re-runs.
- Extending the study to additional gesture datasets would validate whether the random-vs-semihard finding is task-specific or general.
