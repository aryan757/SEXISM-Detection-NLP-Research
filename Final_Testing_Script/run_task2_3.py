"""
EXIST 2026 — Subtask 2.3: Sexism Categorisation in Memes (multi-label)
=======================================================================
INPUT : EXIST2026_test_clean.json
        task2_1_hard_{TEAM}_{RUN}.json  ← 2.1 hard predictions (cascade)
OUTPUT: task2_3_hard_{TEAM}_{RUN}.json  ← hard labels (array of categories or ["NO"])
        task2_3_soft_{TEAM}_{RUN}.json  ← soft labels (6 independent probabilities)

NOTE: Must be run AFTER task 2.1.
      Non-sexist memes (label_21=NO) get value=["NO"] in hard
      and {NO:1.0, all_cats:0.0} in soft.

Valid hard label values (in arrays):
  "NO"
  "IDEOLOGICAL-INEQUALITY"
  "STEREOTYPING-DOMINANCE"
  "OBJECTIFICATION"
  "SEXUAL-VIOLENCE"
  "MISOGYNY-NON-SEXUAL-VIOLENCE"

Soft label probabilities are INDEPENDENT (do not need to sum to 1.0).

Usage:
    python run_task2_3.py \
        --test_path      path/to/EXIST2026_test_clean.json \
        --pred_21_path   ./submission/task2_1_hard_YOURTEAMNAME_1.json \
        --output_dir     ./submission \
        --model_key      qwen \
        --run_id         1
"""

import argparse
import json
import logging
import time
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser

from config   import TEAM_NAME, CATEGORIES_2_3
from prompts  import PROMPT_2_3
from utils    import (
    build_llm, safe_parse, load_test_data, write_output,
    make_hard_record_23, make_soft_record_23, output_filename,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# LOAD 2.1 PREDICTIONS
# ──────────────────────────────────────────────────────────────────────

def load_pred_21(pred_21_path: str) -> dict:
    with open(pred_21_path, encoding="utf-8") as f:
        records = json.load(f)
    result = {}
    for r in records:
        label = r["value"]
        soft  = 1.0 if label == "YES" else 0.0
        result[str(r["id"])] = {"label": label, "soft_score": soft}
    return result


# ──────────────────────────────────────────────────────────────────────
# CORE CLASSIFIER
# ──────────────────────────────────────────────────────────────────────

def classify_record_23(record: dict, pred_21: dict, chain) -> dict:
    """
    Run subtask 2.3 on a single test record.
    Returns {id_exist, hard_labels, scores}.
    """
    id_exist = str(record["id_EXIST"])
    text     = record.get("text", "").strip() or "[no text]"

    p21      = pred_21.get(id_exist, {"label": "NO", "soft_score": 0.0})
    label_21 = p21["label"]
    soft_21  = p21["soft_score"]

    # not sexist → all zeros
    if label_21 == "NO":
        log.info(f"[{id_exist}] label_21=NO → skipping LLM, all categories = 0")
        return {
            "id_exist":    id_exist,
            "hard_labels": ["NO"],
            "scores": {
                "NO": 1.0,
                "IDEOLOGICAL-INEQUALITY":       0.0,
                "STEREOTYPING-DOMINANCE":       0.0,
                "OBJECTIFICATION":              0.0,
                "SEXUAL-VIOLENCE":              0.0,
                "MISOGYNY-NON-SEXUAL-VIOLENCE": 0.0,
            },
        }

    try:
        raw = chain.invoke({
            "meme_text": text,
            "label_21":  label_21,
            "soft_21":   round(soft_21, 3),
        })
        pred = safe_parse(raw, "2.3")
    except Exception as e:
        log.error(f"[{id_exist}][2.3] LLM error: {e}")
        pred = {
            "labels": ["STEREOTYPING-DOMINANCE"],
            "p_ideological_inequality": 0.0,
            "p_stereotyping_dominance": 0.5,
            "p_objectification": 0.0,
            "p_sexual_violence": 0.0,
            "p_misogyny_non_sexual_violence": 0.0,
            "reasoning": "llm_error",
        }

    # extract per-category scores
    scores = {
        "NO":                            0.0,
        "IDEOLOGICAL-INEQUALITY":        float(pred.get("p_ideological_inequality", 0.0) or 0.0),
        "STEREOTYPING-DOMINANCE":        float(pred.get("p_stereotyping_dominance", 0.0) or 0.0),
        "OBJECTIFICATION":               float(pred.get("p_objectification", 0.0) or 0.0),
        "SEXUAL-VIOLENCE":               float(pred.get("p_sexual_violence", 0.0) or 0.0),
        "MISOGYNY-NON-SEXUAL-VIOLENCE":  float(pred.get("p_misogyny_non_sexual_violence", 0.0) or 0.0),
    }
    # clamp all to [0, 1]
    for k in scores:
        scores[k] = max(0.0, min(1.0, scores[k]))

    # hard labels = all categories >= 0.5
    hard_labels = pred.get("labels", [])
    # validate and clean hard labels
    valid_hard = [
        l for l in hard_labels
        if l in (CATEGORIES_2_3 + ["NO"])
    ]
    # fallback: if empty but IS sexist, pick highest scoring category
    if not valid_hard:
        cat_scores = {k: v for k, v in scores.items() if k != "NO"}
        best = max(cat_scores, key=cat_scores.get)
        valid_hard = [best]

    reasoning = pred.get("reasoning", "")
    log.info(f"[{id_exist}] labels={valid_hard} | {reasoning[:80]}")

    return {
        "id_exist":    id_exist,
        "hard_labels": valid_hard,
        "scores":      scores,
    }


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def run(test_path: str, pred_21_path: str, output_dir: str,
        model_key: str, run_id: int, max_records: int | None):

    records = load_test_data(test_path)
    pred_21 = load_pred_21(pred_21_path)
    if max_records:
        records = records[:max_records]
    log.info(f"Loaded {len(records)} test records")
    log.info(f"Loaded {len(pred_21)} task2.1 predictions from {pred_21_path}")
    log.info(f"Model: {model_key} | Run ID: {run_id} | Team: {TEAM_NAME}")

    llm    = build_llm(model_key)
    parser = JsonOutputParser()
    chain  = PROMPT_2_3 | llm | parser

    hard_out, soft_out = [], []

    for i, rec in enumerate(records):
        log.info(f"[{i+1}/{len(records)}] Processing {rec.get('id_EXIST','?')}")
        result = classify_record_23(rec, pred_21, chain)

        hard_out.append(make_hard_record_23(result["id_exist"], result["hard_labels"]))
        soft_out.append(make_soft_record_23(result["id_exist"], result["scores"]))
        time.sleep(0.05)

    out = Path(output_dir)
    hard_path = out / output_filename("task2", "3", "hard", run_id)
    soft_path = out / output_filename("task2", "3", "soft", run_id)

    write_output(hard_out, str(hard_path))
    write_output(soft_out, str(soft_path))

    log.info(f"\nDone! Files written:")
    log.info(f"  Hard → {hard_path}")
    log.info(f"  Soft → {soft_path}")

    return hard_out, soft_out


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXIST 2026 — Subtask 2.3")
    parser.add_argument("--test_path",    required=True, help="Path to EXIST2026_test_clean.json")
    parser.add_argument("--pred_21_path", required=True, help="Path to task2_1_hard_*_1.json")
    parser.add_argument("--output_dir",   default="./submission", help="Output directory")
    parser.add_argument("--model_key",    default="qwen", help="Model key from config.py")
    parser.add_argument("--run_id",       type=int, default=1, help="Run ID (1-3)")
    parser.add_argument("--max_records",  type=int, default=None, help="Limit records (for testing)")
    args = parser.parse_args()

    run(
        test_path    = args.test_path,
        pred_21_path = args.pred_21_path,
        output_dir   = args.output_dir,
        model_key    = args.model_key,
        run_id       = args.run_id,
        max_records  = args.max_records,
    )
