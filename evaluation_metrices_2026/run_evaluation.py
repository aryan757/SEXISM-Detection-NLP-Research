"""
EXIST 2026 — Official Evaluation Script (Organiser Spec)
=========================================================
Follows the EXACT metrics and hierarchies specified by the EXIST 2026 organisers:

  Hard subtasks 2.1 :  ICM, ICMNorm, FMeasure          (no hierarchy)
  Hard subtasks 2.2 :  ICM, ICMNorm, FMeasure           + hierarchy YES→[DIRECT,JUDGEMENTAL]
  Hard subtasks 2.3 :  ICM, ICMNorm, FMeasure           + hierarchy YES→[5 categories]
  Soft subtasks 2.1 :  ICMSoft, ICMSoftNorm, CrossEntropy  (no hierarchy)
  Soft subtasks 2.2 :  ICMSoft, ICMSoftNorm, CrossEntropy  + hierarchy
  Soft subtasks 2.3 :  ICMSoft, ICMSoftNorm             (NO CrossEntropy for 2.3)

TWO evaluation sections:

  SECTION A — Training-data evaluation (proper ICM with real gold labels)
    Source: EXIST_2026/training/outputs/*.json  →  converted to PyEvALL format
    Gold  : EXIST_2026_TESTING/evaluation/golds/EXIST2025_training_task2_*.json
    Note  : Only the training experiment outputs (moondream, qwen, etc.) are evaluated here.

  SECTION B — Test-set submissions vs Majority-Class Baseline (same 1053 IDs)
    Source: EXIST_2026_TESTING/exist2026_AryanSomnathBanerjee/
    Gold  : baselines/EXIST2025_test_task2_*_majority_class_*.json
    Note  : ICM unavailable here (single-class baseline → division by zero by design).
            These numbers are only meaningful once official test gold is released.
"""

import sys, json, warnings, tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE         = Path("/Users/aryan/Desktop/aryan_Exist_2026_dataset")
PRED_DIR     = BASE / "EXIST_2026_TESTING" / "exist2026_AryanSomnathBanerjee"
BASELINE_DIR = BASE / "EXIST_2026_TESTING" / "evaluation" / "baselines"
GOLD_DIR     = BASE / "EXIST_2026_TESTING" / "evaluation" / "golds"
TRAIN_OUT    = BASE / "EXIST_2026" / "training" / "outputs"
OUT_DIR      = BASE / "evaluation_metrices_2026" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "evaluation_metrices_2026" / "PyEvALL"))
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.utils.utils import PyEvALLUtils

# ── Organiser-specified metrics per task ─────────────────────────────────────
HARD_METRICS     = [MetricFactory.ICM.value, MetricFactory.ICMNorm.value, MetricFactory.FMeasure.value]
SOFT_21_METRICS  = [MetricFactory.ICMSoft.value, MetricFactory.ICMSoftNorm.value, MetricFactory.CrossEntropy.value]
SOFT_22_METRICS  = [MetricFactory.ICMSoft.value, MetricFactory.ICMSoftNorm.value, MetricFactory.CrossEntropy.value]
SOFT_23_METRICS  = [MetricFactory.ICMSoft.value, MetricFactory.ICMSoftNorm.value]  # no CE for 2.3

BASELINE_HARD_METRICS = [MetricFactory.FMeasure.value, MetricFactory.Precision.value, MetricFactory.Recall.value]
BASELINE_SOFT_METRICS = [MetricFactory.MAE.value, MetricFactory.CrossEntropy.value]

# ── Organiser-specified hierarchies ──────────────────────────────────────────
H22 = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
H23 = {"YES": ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
               "OBJECTIFICATION", "SEXUAL-VIOLENCE",
               "MISOGYNY-NON-SEXUAL-VIOLENCE"], "NO": []}

TASK_CONFIG = {
    "task2_1": {"hard_metrics": HARD_METRICS, "soft_metrics": SOFT_21_METRICS, "hierarchy": None},
    "task2_2": {"hard_metrics": HARD_METRICS, "soft_metrics": SOFT_22_METRICS, "hierarchy": H22},
    "task2_3": {"hard_metrics": HARD_METRICS, "soft_metrics": SOFT_23_METRICS, "hierarchy": H23},
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_params(hierarchy=None, report_type=None):
    p = {PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE}
    if hierarchy:
        p[PyEvALLUtils.PARAM_HIERARCHY] = hierarchy
    p[PyEvALLUtils.PARAM_REPORT] = report_type or PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME
    return p


def safe_eval_one(pred_path, gold_path, metric, hierarchy=None):
    """Evaluate a single metric safely. Returns (metric, value_or_status)."""
    params = make_params(hierarchy)
    try:
        report = PyEvALLEvaluation().evaluate(str(pred_path), str(gold_path), [metric], **params)
        df = report.df_average
        if df is not None and not df.empty:
            num_cols = [c for c in df.columns if c != "files"]
            if num_cols:
                val = df[num_cols[0]].iloc[0]
                try:
                    return metric, round(float(val), 4)
                except (ValueError, TypeError):
                    return metric, str(val)
        return metric, None
    except ZeroDivisionError:
        return metric, "N/A (single-class gold)"
    except Exception as e:
        msg = str(e)
        if "cdf" in msg or "sigma" in msg:
            return metric, "N/A (zero-variance gold)"
        return metric, f"ERR: {msg[:70]}"


def evaluate_file(pred_path, gold_path, metrics, hierarchy=None):
    return {m: v for m, v in (safe_eval_one(pred_path, gold_path, m, hierarchy) for m in metrics)}


def make_md_table(rows_dict, metric_keys):
    """rows_dict = {label: {metric: value}}"""
    hdr = "| System | " + " | ".join(metric_keys) + " |"
    sep = "| :---- | " + " | ".join([":---:"] * len(metric_keys)) + " |"
    lines = [hdr, sep]
    for label, results in rows_dict.items():
        cells = [str(results.get(k, "-")) for k in metric_keys]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def label_dist(fp):
    try:
        data = json.load(open(fp))
        c = Counter()
        for r in data:
            v = r.get("value", "?")
            if isinstance(v, list):
                for lbl in v: c[lbl] += 1
            else:
                c[str(v)] += 1
        total = sum(c.values())
        return "  |  ".join(f"{lbl}:{cnt}({100*cnt/total:.0f}%)" for lbl, cnt in sorted(c.items()))
    except Exception as e:
        return f"Error: {e}"


# ── Convert training experiment outputs → PyEvALL JSON ───────────────────────
CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
        "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]

def convert_training_output(records, model_name):
    """
    Convert training experiment JSON records to 6 PyEvALL-format lists:
    task2_1_hard, task2_1_soft, task2_2_hard, task2_2_soft, task2_3_hard, task2_3_soft
    """
    t21h, t21s, t22h, t22s, t23h, t23s = [], [], [], [], [], []
    tc = "EXIST2025"

    for r in records:
        mid = r["meme_id"]
        h21 = r.get("pred_2_1_hard", "NO")
        s21 = float(r.get("pred_2_1_soft", 0.5))

        # 2.1 hard
        t21h.append({"test_case": tc, "id": mid, "value": h21})
        # 2.1 soft
        t21s.append({"test_case": tc, "id": mid, "value": {"YES": round(s21, 4), "NO": round(1 - s21, 4)}})

        # 2.2 hard — if no sexism → NO
        h22 = r.get("pred_2_2_hard", "NO") if h21 == "YES" else "NO"
        t22h.append({"test_case": tc, "id": mid, "value": h22})
        # 2.2 soft
        pd = float(r.get("pred_2_2_p_direct") or 0.0)
        pj = float(r.get("pred_2_2_p_judgemental") or 0.0)
        total = pd + pj if (pd + pj) > 0 else 1.0
        no22 = round(1 - s21, 4) if h21 == "YES" else 1.0
        t22s.append({"test_case": tc, "id": mid, "value": {
            "DIRECT": round(pd / total * s21, 4),
            "JUDGEMENTAL": round(pj / total * s21, 4),
            "NO": round(no22, 4)
        }})

        # 2.3 hard
        h23 = r.get("pred_2_3_hard", [])
        if not h23 or h21 == "NO":
            h23 = ["NO"]
        t23h.append({"test_case": tc, "id": mid, "value": h23})
        # 2.3 soft
        scores = r.get("pred_2_3_scores", {})
        t23s.append({"test_case": tc, "id": mid, "value": {
            cat: round(float(scores.get(cat, 0.0)), 4) for cat in CATS
        }})

    return {"task2_1_hard": t21h, "task2_1_soft": t21s,
            "task2_2_hard": t22h, "task2_2_soft": t22s,
            "task2_3_hard": t23h, "task2_3_soft": t23s}


# ── Output storage ────────────────────────────────────────────────────────────
tsv_rows = []
md_lines = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("=" * 72)
print("  EXIST 2026 — Official Evaluation (Organiser Spec)")
print(f"  {now_str}")
print("=" * 72)

md_lines += [
    "# EXIST 2026 — Evaluation Report (Organiser Spec)",
    "",
    f"**Author:** Aryan Somnath Banerjee  ",
    f"**Generated:** {now_str}",
    "",
    "Metrics follow the **exact organiser specification**:",
    "",
    "| Subtask | Hard metrics | Soft metrics | Hierarchy |",
    "| :------ | :----------- | :----------- | :-------- |",
    "| 2.1 | ICM, ICMNorm, F1 | ICMSoft, ICMSoftNorm, CrossEntropy | None |",
    "| 2.2 | ICM, ICMNorm, F1 | ICMSoft, ICMSoftNorm, CrossEntropy | YES→[DIRECT, JUDGEMENTAL] |",
    "| 2.3 | ICM, ICMNorm, F1 | ICMSoft, ICMSoftNorm | YES→[5 categories] |",
    "",
    "---", "",
]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Training-data evaluation (PROPER ICM with real gold)
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━  SECTION A: Training Experiments vs Official Gold  ━━━\n")
md_lines += [
    "## Section A — Training Experiments vs Official Gold",
    "",
    "> Gold labels from `evaluation/golds/EXIST2025_training_task2_*.json`  ",
    "> Predictions from `EXIST_2026/training/outputs/*.json` (converted to PyEvALL format)  ",
    "> **ICM / ICMNorm / ICMSoft / ICMSoftNorm are all properly computed here.**",
    "", "---", "",
]

GOLD_MAP = {
    "task2_1_hard": GOLD_DIR / "EXIST2025_training_task2_1_gold_hard.json",
    "task2_1_soft": GOLD_DIR / "EXIST2025_training_task2_1_gold_soft.json",
    "task2_2_hard": GOLD_DIR / "EXIST2025_training_task2_2_gold_hard.json",
    "task2_2_soft": GOLD_DIR / "EXIST2025_training_task2_2_gold_soft.json",
    "task2_3_hard": GOLD_DIR / "EXIST2025_training_task2_3_gold_hard.json",
    "task2_3_soft": GOLD_DIR / "EXIST2025_training_task2_3_gold_soft.json",
}

TRAIN_MODELS = {
    "moondream":     TRAIN_OUT / "exist2026_results_moondream.json",
    "qwen":          TRAIN_OUT / "exist2026_results_qwen.json",
    "unsloth_gemma": TRAIN_OUT / "exist2026_results_unsloth_gemma.json",
    "unsloth_qwen":  TRAIN_OUT / "exist2026_results_unsloth_qwen.json",
}

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Convert each training model output to PyEvALL files
    converted = {}
    for model_name, fpath in TRAIN_MODELS.items():
        if not fpath.exists():
            continue
        data = json.load(open(fpath))
        records = data.get("results", data)
        if not isinstance(records, list) or len(records) == 0:
            continue
        pyevall_data = convert_training_output(records, model_name)
        converted[model_name] = {}
        for key, recs in pyevall_data.items():
            out = tmpdir / f"{model_name}_{key}.json"
            json.dump(recs, open(out, "w"), indent=2)
            converted[model_name][key] = out
        print(f"  Converted {model_name}: {len(records)} records")

    for task in ["task2_1", "task2_2", "task2_3"]:
        cfg = TASK_CONFIG[task]
        md_lines.append(f"### {task.upper()}")

        for mode in ["hard", "soft"]:
            key     = f"{task}_{mode}"
            gold    = GOLD_MAP[key]
            metrics = cfg["hard_metrics"] if mode == "hard" else cfg["soft_metrics"]
            hier    = cfg["hierarchy"]

            print(f"\n  {task.upper()} [{mode.upper()}]")
            md_lines.append(f"\n#### {task.upper()} — {mode.upper()}")
            md_lines.append(f"> Gold: `{gold.name}` | Metrics: {', '.join(metrics)}")
            md_lines.append("")

            model_results = {}
            for model_name, files in converted.items():
                if key not in files:
                    continue
                pred = files[key]
                result = evaluate_file(pred, gold, metrics, hier)
                model_results[model_name] = result
                vals = "  ".join(f"{m}={v}" for m, v in result.items())
                print(f"    {model_name}: {vals}")
                for m, v in result.items():
                    tsv_rows.append({"section": "training_vs_gold", "task": task,
                                     "mode": mode, "run": model_name,
                                     "gold": gold.name, "metric": m, "value": v})

            if model_results:
                all_keys = list(next(iter(model_results.values())).keys())
                md_lines += make_md_table(model_results, all_keys)
            md_lines.append("")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — Test submissions vs Majority-Class Baseline
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n━━━  SECTION B: Test Submissions vs Majority-Class Baseline  ━━━\n")
print("  NOTE: ICM excluded — baseline is single-class (all YES/NO), causing division by zero.")
print("  FMeasure, Precision, Recall used for hard; MAE, CrossEntropy for soft.\n")

md_lines += [
    "",
    "---", "",
    "## Section B — Test Submissions vs Majority-Class Baseline",
    "",
    "> **Why no ICM here?** The majority-class baseline assigns the same label to every instance  ",
    "> (e.g. all `YES`). ICM normalisation = `(ICM_pred − (−ICM_gold)) / (ICM_gold − (−ICM_gold))`.  ",
    "> When gold is a single class: `ICM_gold = −log₂(1.0) = 0`, so denominator = `0`.  ",
    "> **This is not a bug** — ICM requires uncertainty in the gold distribution.  ",
    "> These results will be updated with proper ICM once organisers release the test-set gold.",
    "",
    "| Subtask | Hard metrics used | Soft metrics used |",
    "| :------ | :---------------- | :---------------- |",
    "| All | FMeasure, Precision, Recall | MAE, CrossEntropy |",
    "",
]

BASELINE_MAP = {
    "task2_1_hard": BASELINE_DIR / "EXIST2025_test_task2_1_majority_class_hard.json",
    "task2_1_soft": BASELINE_DIR / "EXIST2025_test_task2_1_majority_class_soft.json",
    "task2_2_hard": BASELINE_DIR / "EXIST2025_test_task2_2_majority_class_hard.json",
    "task2_2_soft": BASELINE_DIR / "EXIST2025_test_task2_2_majority_class_soft.json",
    "task2_3_hard": BASELINE_DIR / "EXIST2025_test_task2_3_majority_class_hard.json",
    "task2_3_soft": BASELINE_DIR / "EXIST2025_test_task2_3_majority_class_soft.json",
}

for task in ["task2_1", "task2_2", "task2_3"]:
    cfg = TASK_CONFIG[task]
    md_lines.append(f"### {task.upper()}")

    for mode in ["hard", "soft"]:
        key     = f"{task}_{mode}"
        gold    = BASELINE_MAP[key]
        metrics = BASELINE_HARD_METRICS if mode == "hard" else BASELINE_SOFT_METRICS

        print(f"  {task.upper()} [{mode.upper()}]")
        md_lines.append(f"\n#### {task.upper()} — {mode.upper()}")
        md_lines.append(f"> Gold: majority-class baseline | Metrics: {', '.join(metrics)}")
        md_lines.append("")

        run_results = {}
        for run in [1, 2, 3]:
            pred = PRED_DIR / f"{task}_{mode}_AryanSomnathBanerjee_{run}.json"
            if not pred.exists():
                continue
            result = evaluate_file(pred, gold, metrics, hierarchy=None)
            run_results[f"Run {run}"] = result
            vals = "  ".join(f"{m}={v}" for m, v in result.items())
            print(f"    Run {run}: {vals}")
            for m, v in result.items():
                tsv_rows.append({"section": "test_vs_baseline", "task": task,
                                 "mode": mode, "run": run,
                                 "gold": "majority_class_baseline", "metric": m, "value": v})

        if run_results:
            all_keys = list(next(iter(run_results.values())).keys())
            md_lines += make_md_table(run_results, all_keys)
        md_lines.append("")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Label Distribution
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n━━━  SECTION C: Label Distribution  ━━━\n")
md_lines += ["", "---", "", "## Section C — Label Distribution", ""]

for task in ["task2_1", "task2_2", "task2_3"]:
    md_lines.append(f"### {task.upper()}")
    md_lines.append("| Run | Label Distribution |")
    md_lines.append("| :-- | :----------------- |")
    for run in [1, 2, 3]:
        p = PRED_DIR / f"{task}_hard_AryanSomnathBanerjee_{run}.json"
        if p.exists():
            dist = label_dist(p)
            print(f"  {task} Run {run}: {dist}")
            md_lines.append(f"| Run {run} | {dist} |")
    md_lines.append("")


# ═══════════════════════════════════════════════════════════════════════════
# Metric glossary
# ═══════════════════════════════════════════════════════════════════════════
md_lines += [
    "---", "",
    "## Metric Reference",
    "",
    "| Metric | Range | Better | Official? | Description |",
    "| :----- | :---- | :----- | :-------: | :---------- |",
    "| **ICM** | −∞ to +∞ | ↑ higher | ✅ Hard | Information Contrast Model — penalises wrong labels via information theory |",
    "| **ICMNorm** | −1 to +1 | ↑ higher | ✅ Hard | ICM normalised; 0 = random, 1 = perfect |",
    "| **FMeasure** | 0 to 1 | ↑ higher | ✅ Hard | Macro-avg F1 across all classes |",
    "| **ICMSoft** | −∞ to +∞ | ↑ higher | ✅ Soft | Soft variant of ICM for probability distributions |",
    "| **ICMSoftNorm** | −1 to +1 | ↑ higher | ✅ **Primary** | **Official primary metric for EXIST 2026 soft subtasks** |",
    "| **CrossEntropy** | 0 to +∞ | ↓ lower | ✅ Soft | KL-divergence of predicted vs gold distributions |",
    "| **Precision** | 0 to 1 | ↑ higher | ❌ | Used only in Section B (vs baseline) |",
    "| **Recall** | 0 to 1 | ↑ higher | ❌ | Used only in Section B (vs baseline) |",
    "| **MAE** | 0 to 1 | ↓ lower | ❌ | Used only in Section B soft (vs baseline) |",
    "",
]

# ── Save ──────────────────────────────────────────────────────────────────────
md_path  = OUT_DIR / f"evaluation_report_{timestamp}.md"
tsv_path = OUT_DIR / f"evaluation_results_{timestamp}.tsv"

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

if tsv_rows:
    headers = ["section", "task", "mode", "run", "gold", "metric", "value"]
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in tsv_rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")

print(f"\n{'='*72}")
print(f"  ✅  Markdown → {md_path}")
print(f"  ✅  TSV      → {tsv_path}")
print(f"{'='*72}\n")
