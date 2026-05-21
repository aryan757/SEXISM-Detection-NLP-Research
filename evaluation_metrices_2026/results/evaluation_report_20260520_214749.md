# EXIST 2026 — Evaluation Report (Organiser Spec)

**Author:** Aryan Somnath Banerjee  
**Generated:** 2026-05-20 21:47:49

Metrics follow the **exact organiser specification**:

| Subtask | Hard metrics | Soft metrics | Hierarchy |
| :------ | :----------- | :----------- | :-------- |
| 2.1 | ICM, ICMNorm, F1 | ICMSoft, ICMSoftNorm, CrossEntropy | None |
| 2.2 | ICM, ICMNorm, F1 | ICMSoft, ICMSoftNorm, CrossEntropy | YES→[DIRECT, JUDGEMENTAL] |
| 2.3 | ICM, ICMNorm, F1 | ICMSoft, ICMSoftNorm | YES→[5 categories] |

---

## Section A — Training Experiments vs Official Gold

> Gold labels from `evaluation/golds/EXIST2025_training_task2_*.json`  
> Predictions from `EXIST_2026/training/outputs/*.json` (converted to PyEvALL format)  
> **ICM / ICMNorm / ICMSoft / ICMSoftNorm are all properly computed here.**

---

### TASK2_1

#### TASK2_1 — HARD
> Gold: `EXIST2025_training_task2_1_gold_hard.json` | Metrics: ICM, ICMNorm, FMeasure

| System | ICM | ICMNorm | FMeasure |
| :---- | :---: | :---: | :---: |
| moondream | -0.9745 | 0.0 | 0.0015 |
| qwen | -0.9741 | 0.0 | 0.0005 |
| unsloth_gemma | -0.9749 | 0.0 | 0.0 |
| unsloth_qwen | -0.9749 | 0.0 | 0.0 |


#### TASK2_1 — SOFT
> Gold: `EXIST2025_training_task2_1_gold_soft.json` | Metrics: ICMSoft, ICMSoftNorm, CrossEntropy

| System | ICMSoft | ICMSoftNorm | CrossEntropy |
| :---- | :---: | :---: | :---: |
| moondream | -3.1773 | 0.0006 | 0.0029 |
| qwen | -3.1766 | 0.0007 | 0.0004 |
| unsloth_gemma | -3.1794 | 0.0002 | 0.0008 |
| unsloth_qwen | -3.1794 | 0.0002 | 0.0008 |

### TASK2_2

#### TASK2_2 — HARD
> Gold: `EXIST2025_training_task2_2_gold_hard.json` | Metrics: ICM, ICMNorm, FMeasure

| System | ICM | ICMNorm | FMeasure |
| :---- | :---: | :---: | :---: |
| moondream | -1.4462 | 0.0 | 0.001 |
| qwen | -1.4454 | 0.0001 | 0.0005 |
| unsloth_gemma | -1.4465 | 0.0 | 0.0 |
| unsloth_qwen | -1.4465 | 0.0 | 0.0 |


#### TASK2_2 — SOFT
> Gold: `EXIST2025_training_task2_2_gold_soft.json` | Metrics: ICMSoft, ICMSoftNorm, CrossEntropy

| System | ICMSoft | ICMSoftNorm | CrossEntropy |
| :---- | :---: | :---: | :---: |
| moondream | -4.7773 | 0.0003 | 0.0043 |
| qwen | -4.7762 | 0.0004 | 0.0011 |
| unsloth_gemma | -4.7829 | 0.0 | 0.0058 |
| unsloth_qwen | -4.7829 | 0.0 | 0.0058 |

### TASK2_3

#### TASK2_3 — HARD
> Gold: `EXIST2025_training_task2_3_gold_hard.json` | Metrics: ICM, ICMNorm, FMeasure

| System | ICM | ICMNorm | FMeasure |
| :---- | :---: | :---: | :---: |
| moondream | -2.4533 | 0.0 | 0.0005 |
| qwen | -2.4508 | 0.0004 | 0.0012 |
| unsloth_gemma | -2.4537 | 0.0 | 0.0 |
| unsloth_qwen | -2.4537 | 0.0 | 0.0 |


#### TASK2_3 — SOFT
> Gold: `EXIST2025_training_task2_3_gold_soft.json` | Metrics: ICMSoft, ICMSoftNorm

| System | ICMSoft | ICMSoftNorm |
| :---- | :---: | :---: |
| moondream | -9.5752 | 0.0 |
| qwen | -9.5744 | 0.0001 |
| unsloth_gemma | -9.5756 | 0.0 |
| unsloth_qwen | -9.5756 | 0.0 |


---

## Section B — Test Submissions vs Majority-Class Baseline

> **Why no ICM here?** The majority-class baseline assigns the same label to every instance  
> (e.g. all `YES`). ICM normalisation = `(ICM_pred − (−ICM_gold)) / (ICM_gold − (−ICM_gold))`.  
> When gold is a single class: `ICM_gold = −log₂(1.0) = 0`, so denominator = `0`.  
> **This is not a bug** — ICM requires uncertainty in the gold distribution.  
> These results will be updated with proper ICM once organisers release the test-set gold.

| Subtask | Hard metrics used | Soft metrics used |
| :------ | :---------------- | :---------------- |
| All | FMeasure, Precision, Recall | MAE, CrossEntropy |

### TASK2_1

#### TASK2_1 — HARD
> Gold: majority-class baseline | Metrics: FMeasure, Precision, Recall

| System | FMeasure | Precision | Recall |
| :---- | :---: | :---: | :---: |
| Run 1 | 0.6854 | 1.0 | 0.5214 |
| Run 2 | 0.6854 | 1.0 | 0.5214 |
| Run 3 | 0.6662 | 1.0 | 0.4995 |


#### TASK2_1 — SOFT
> Gold: majority-class baseline | Metrics: MAE, CrossEntropy

| System | MAE | CrossEntropy |
| :---- | :---: | :---: |
| Run 1 | 0.5387 | 2.1961 |
| Run 2 | 0.5415 | 2.1937 |
| Run 3 | 0.604 | 2.1545 |

### TASK2_2

#### TASK2_2 — HARD
> Gold: majority-class baseline | Metrics: FMeasure, Precision, Recall

| System | FMeasure | Precision | Recall |
| :---- | :---: | :---: | :---: |
| Run 1 | 0.6474 | 1.0 | 0.4786 |
| Run 2 | 0.6474 | 1.0 | 0.4786 |
| Run 3 | 0.6671 | 1.0 | 0.5005 |


#### TASK2_2 — SOFT
> Gold: majority-class baseline | Metrics: MAE, CrossEntropy

| System | MAE | CrossEntropy |
| :---- | :---: | :---: |
| Run 1 | 0.5071 | 5.946 |
| Run 2 | 0.3479 | 5.1999 |
| Run 3 | 0.4998 | 5.7638 |

### TASK2_3

#### TASK2_3 — HARD
> Gold: majority-class baseline | Metrics: FMeasure, Precision, Recall

| System | FMeasure | Precision | Recall |
| :---- | :---: | :---: | :---: |
| Run 1 | 0.6474 | 1.0 | 0.4786 |
| Run 2 | 0.6474 | 1.0 | 0.4786 |
| Run 3 | 0.6474 | 1.0 | 0.4786 |


#### TASK2_3 — SOFT
> Gold: majority-class baseline | Metrics: MAE, CrossEntropy

| System | MAE | CrossEntropy |
| :---- | :---: | :---: |
| Run 1 | 0.2013 | 5.3683 |
| Run 2 | 0.1898 | 1.7173 |
| Run 3 | 0.1898 | 1.7173 |


---

## Section C — Label Distribution

### TASK2_1
| Run | Label Distribution |
| :-- | :----------------- |
| Run 1 | NO:504(48%)  |  YES:549(52%) |
| Run 2 | NO:504(48%)  |  YES:549(52%) |
| Run 3 | NO:527(50%)  |  YES:526(50%) |

### TASK2_2
| Run | Label Distribution |
| :-- | :----------------- |
| Run 1 | DIRECT:455(43%)  |  JUDGEMENTAL:94(9%)  |  NO:504(48%) |
| Run 2 | DIRECT:455(43%)  |  JUDGEMENTAL:94(9%)  |  NO:504(48%) |
| Run 3 | DIRECT:156(15%)  |  JUDGEMENTAL:370(35%)  |  NO:527(50%) |

### TASK2_3
| Run | Label Distribution |
| :-- | :----------------- |
| Run 1 | IDEOLOGICAL-INEQUALITY:17(2%)  |  MISOGYNY-NON-SEXUAL-VIOLENCE:1(0%)  |  NO:504(48%)  |  OBJECTIFICATION:75(7%)  |  SEXUAL-VIOLENCE:7(1%)  |  STEREOTYPING-DOMINANCE:455(43%) |
| Run 2 | IDEOLOGICAL-INEQUALITY:37(3%)  |  MISOGYNY-NON-SEXUAL-VIOLENCE:41(3%)  |  NO:504(36%)  |  OBJECTIFICATION:325(23%)  |  SEXUAL-VIOLENCE:26(2%)  |  STEREOTYPING-DOMINANCE:456(33%) |
| Run 3 | IDEOLOGICAL-INEQUALITY:37(3%)  |  MISOGYNY-NON-SEXUAL-VIOLENCE:41(3%)  |  NO:504(36%)  |  OBJECTIFICATION:325(23%)  |  SEXUAL-VIOLENCE:26(2%)  |  STEREOTYPING-DOMINANCE:456(33%) |

---

## Metric Reference

| Metric | Range | Better | Official? | Description |
| :----- | :---- | :----- | :-------: | :---------- |
| **ICM** | −∞ to +∞ | ↑ higher | ✅ Hard | Information Contrast Model — penalises wrong labels via information theory |
| **ICMNorm** | −1 to +1 | ↑ higher | ✅ Hard | ICM normalised; 0 = random, 1 = perfect |
| **FMeasure** | 0 to 1 | ↑ higher | ✅ Hard | Macro-avg F1 across all classes |
| **ICMSoft** | −∞ to +∞ | ↑ higher | ✅ Soft | Soft variant of ICM for probability distributions |
| **ICMSoftNorm** | −1 to +1 | ↑ higher | ✅ **Primary** | **Official primary metric for EXIST 2026 soft subtasks** |
| **CrossEntropy** | 0 to +∞ | ↓ lower | ✅ Soft | KL-divergence of predicted vs gold distributions |
| **Precision** | 0 to 1 | ↑ higher | ❌ | Used only in Section B (vs baseline) |
| **Recall** | 0 to 1 | ↑ higher | ❌ | Used only in Section B (vs baseline) |
| **MAE** | 0 to 1 | ↓ lower | ❌ | Used only in Section B soft (vs baseline) |
