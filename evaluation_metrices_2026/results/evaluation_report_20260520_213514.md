# EXIST 2026 — Evaluation Report

**Author:** Aryan Somnath Banerjee  
**Generated:** 2026-05-20 21:35:14  
**Runs evaluated:** 3 runs × 3 subtasks × 2 modes = 18 prediction files

> **Note:** Official test-set gold labels are not yet released for EXIST 2026.  
> **Section 1** compares each run against the official majority-class baseline (1053 shared test IDs).  
> **Section 2** measures inter-run consistency (Run 2 & 3 vs Run 1).  
> **Section 3** shows label distribution for each run.  
> `ICM` / `ICMSoft` / `ICMSoftNorm` are excluded here because they require distributional variation in the gold standard (not available with a constant baseline).  

---

## Section 1: Runs vs Majority-Class Baseline

Metrics: **F1-Macro, Precision, Recall** (hard) | **MAE, CrossEntropy** (soft)

### TASK2_1

#### TASK2_1 — HARD

| Run | FMeasure | Precision | Recall |
| :-- | :---: | :---: | :---: |
| Run 1 | 0.6854 | 1.0 | 0.5214 |
| Run 2 | 0.6854 | 1.0 | 0.5214 |
| Run 3 | 0.6662 | 1.0 | 0.4995 |


#### TASK2_1 — SOFT

| Run | MAE | CrossEntropy |
| :-- | :---: | :---: |
| Run 1 | 0.5387 | 2.1961 |
| Run 2 | 0.5415 | 2.1937 |
| Run 3 | 0.604 | 2.1545 |

### TASK2_2

#### TASK2_2 — HARD

| Run | FMeasure | Precision | Recall |
| :-- | :---: | :---: | :---: |
| Run 1 | 0.6474 | 1.0 | 0.4786 |
| Run 2 | 0.6474 | 1.0 | 0.4786 |
| Run 3 | 0.6671 | 1.0 | 0.5005 |


#### TASK2_2 — SOFT

| Run | MAE | CrossEntropy |
| :-- | :---: | :---: |
| Run 1 | 0.5071 | 5.946 |
| Run 2 | 0.3479 | 5.1999 |
| Run 3 | 0.4998 | 5.7638 |

### TASK2_3

#### TASK2_3 — HARD

| Run | FMeasure | Precision | Recall |
| :-- | :---: | :---: | :---: |
| Run 1 | 0.6474 | 1.0 | 0.4786 |
| Run 2 | 0.6474 | 1.0 | 0.4786 |
| Run 3 | 0.6474 | 1.0 | 0.4786 |


#### TASK2_3 — SOFT

| Run | MAE | CrossEntropy |
| :-- | :---: | :---: |
| Run 1 | 0.2013 | 5.3683 |
| Run 2 | 0.1898 | 1.7173 |
| Run 3 | 0.1898 | 1.7173 |


---

## Section 2: Inter-Run Agreement (Run 1 as Reference)

Run 2 and Run 3 are evaluated **against Run 1** to measure consistency.

### TASK2_1

#### TASK2_1 — HARD (vs Run 1)

| Run | FMeasure | Precision | Recall |
| :-- | :---: | :---: | :---: |
| Run 2 vs Run 1 | 1.0 | 1.0 | 1.0 |
| Run 3 vs Run 1 | 0.6874 | 0.6876 | 0.6879 |


#### TASK2_1 — SOFT (vs Run 1)

| Run | MAE | CrossEntropy |
| :-- | :---: | :---: |
| Run 2 vs Run 1 | 0.0335 | 0.4747 |
| Run 3 vs Run 1 | 0.2817 | 1.0419 |

### TASK2_2

#### TASK2_2 — HARD (vs Run 1)

| Run | FMeasure | Precision | Recall |
| :-- | :---: | :---: | :---: |
| Run 2 vs Run 1 | 1.0 | 1.0 | 1.0 |
| Run 3 vs Run 1 | 0.4178 | 0.5058 | 0.4816 |


#### TASK2_2 — SOFT (vs Run 1)

| Run | MAE | CrossEntropy |
| :-- | :---: | :---: |
| Run 2 vs Run 1 | 0.1895 | 3.6833 |
| Run 3 vs Run 1 | 0.184 | 2.0134 |

### TASK2_3

#### TASK2_3 — HARD (vs Run 1)

| Run | FMeasure | Precision | Recall |
| :-- | :---: | :---: | :---: |
| Run 2 vs Run 1 | 0.5792 | 0.4969 | 1.0 |
| Run 3 vs Run 1 | 0.5792 | 0.4969 | 1.0 |


#### TASK2_3 — SOFT (vs Run 1)

| Run | MAE | CrossEntropy |
| :-- | :---: | :---: |
| Run 2 vs Run 1 | 0.0298 | 0.4828 |
| Run 3 vs Run 1 | 0.0298 | 0.4828 |


---

## Section 3: Label Distribution Summary

Hard-label distributions across all runs (counts + percentages).

### TASK2_1

| Run | Label Distribution |
| :-- | :----------------- |
| Run 1 | NO:504 (47.9%)  |  YES:549 (52.1%) |
| Run 2 | NO:504 (47.9%)  |  YES:549 (52.1%) |
| Run 3 | NO:527 (50.0%)  |  YES:526 (50.0%) |

### TASK2_2

| Run | Label Distribution |
| :-- | :----------------- |
| Run 1 | DIRECT:455 (43.2%)  |  JUDGEMENTAL:94 (8.9%)  |  NO:504 (47.9%) |
| Run 2 | DIRECT:455 (43.2%)  |  JUDGEMENTAL:94 (8.9%)  |  NO:504 (47.9%) |
| Run 3 | DIRECT:156 (14.8%)  |  JUDGEMENTAL:370 (35.1%)  |  NO:527 (50.0%) |

### TASK2_3

| Run | Label Distribution |
| :-- | :----------------- |
| Run 1 | IDEOLOGICAL-INEQUALITY:17 (1.6%)  |  MISOGYNY-NON-SEXUAL-VIOLENCE:1 (0.1%)  |  NO:504 (47.6%)  |  OBJECTIFICATION:75 (7.1%)  |  SEXUAL-VIOLENCE:7 (0.7%)  |  STEREOTYPING-DOMINANCE:455 (43.0%) |
| Run 2 | IDEOLOGICAL-INEQUALITY:37 (2.7%)  |  MISOGYNY-NON-SEXUAL-VIOLENCE:41 (3.0%)  |  NO:504 (36.3%)  |  OBJECTIFICATION:325 (23.4%)  |  SEXUAL-VIOLENCE:26 (1.9%)  |  STEREOTYPING-DOMINANCE:456 (32.8%) |
| Run 3 | IDEOLOGICAL-INEQUALITY:37 (2.7%)  |  MISOGYNY-NON-SEXUAL-VIOLENCE:41 (3.0%)  |  NO:504 (36.3%)  |  OBJECTIFICATION:325 (23.4%)  |  SEXUAL-VIOLENCE:26 (1.9%)  |  STEREOTYPING-DOMINANCE:456 (32.8%) |

**Majority-class baseline distributions (for reference):**

| Task | Baseline Distribution |
| :--- | :-------------------- |
| TASK2_1 | YES:1053 (100.0%) |
| TASK2_2 | NO:1053 (100.0%) |
| TASK2_3 | NO:1053 (100.0%) |


---

## Notes on Metrics

| Metric | Range | Interpretation |
| :----- | :---- | :------------- |
| **F1-Macro** | 0 – 1 | ↑ higher is better; macro-avg across all classes |
| **Precision** | 0 – 1 | ↑ higher is better |
| **Recall** | 0 – 1 | ↑ higher is better |
| **MAE** | 0 – 1 | ↓ lower is better; mean absolute error of soft scores |
| **CrossEntropy** | 0 – ∞ | ↓ lower is better; divergence of predicted vs gold distributions |

> `N/A (zero-div)` = ICM-family metric undefined when gold is a single class (inherent limitation of the metric with majority-class baseline).  
> `N/A (const-gold)` = ICMSoft undefined when all gold soft scores are identical (zero variance).  
