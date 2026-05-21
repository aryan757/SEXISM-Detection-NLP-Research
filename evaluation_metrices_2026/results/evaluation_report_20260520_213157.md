# EXIST 2026 — Evaluation Report

**Author:** Aryan Somnath Banerjee  
**Generated:** 2026-05-20 21:31:57  
**Runs evaluated:** 3 runs × 3 subtasks × 2 modes (hard/soft) = 18 files

> **Note:** Official test-set gold labels are not yet released for EXIST 2026.
> Section 1 compares against the official **majority-class baseline** (same test IDs).
> Section 2 measures inter-run consistency (agreement between your runs).

## Section 1: Your Runs vs Majority-Class Baseline

> Evaluating each of your 3 runs against the official EXIST2025 test majority-class baseline.


### TASK2_1 — HARD

| Run | ERROR |
| --- | --- |
| Run 1 | float division by zero |
| Run 2 | float division by zero |
| Run 3 | float division by zero |

### TASK2_1 — SOFT

| Run | ERROR |
| --- | --- |
| Run 1 | cdf() not defined when sigma is zero |
| Run 2 | cdf() not defined when sigma is zero |
| Run 3 | cdf() not defined when sigma is zero |

### TASK2_2 — HARD

| Run | ERROR |
| --- | --- |
| Run 1 | float division by zero |
| Run 2 | float division by zero |
| Run 3 | float division by zero |

### TASK2_2 — SOFT

| Run | ERROR |
| --- | --- |
| Run 1 | cdf() not defined when sigma is zero |
| Run 2 | cdf() not defined when sigma is zero |
| Run 3 | cdf() not defined when sigma is zero |

### TASK2_3 — HARD

| Run | ERROR |
| --- | --- |
| Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |
| Run 2 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |
| Run 3 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |

### TASK2_3 — SOFT

| Run | ERROR |
| --- | --- |
| Run 1 | cdf() not defined when sigma is zero |
| Run 2 | cdf() not defined when sigma is zero |
| Run 3 | cdf() not defined when sigma is zero |


## Section 2: Inter-Run Agreement (Run 1 as Reference)

> Run 2 and Run 3 evaluated against Run 1 to measure consistency.


### TASK2_1 — HARD (vs Run 1)

| Comparison | ERROR |
| --- | --- |
| Run 2 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |
| Run 3 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |

### TASK2_1 — SOFT (vs Run 1)

| Comparison | ERROR |
| --- | --- |
| Run 2 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |
| Run 3 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |

### TASK2_2 — HARD (vs Run 1)

| Comparison | ERROR |
| --- | --- |
| Run 2 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |
| Run 3 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |

### TASK2_2 — SOFT (vs Run 1)

| Comparison | ERROR |
| --- | --- |
| Run 2 vs Run 1 | cdf() not defined when sigma is zero |
| Run 3 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |

### TASK2_3 — HARD (vs Run 1)

| Comparison | ERROR |
| --- | --- |
| Run 2 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |
| Run 3 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |

### TASK2_3 — SOFT (vs Run 1)

| Comparison | ERROR |
| --- | --- |
| Run 2 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |
| Run 3 vs Run 1 | 'PyEvALLDataframeReport' object has no attribute 'get_metric_dataframe' |