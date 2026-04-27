"""
EXIST 2026 — Subtask 2.2: Source Intention in Memes
====================================================
INPUT : EXIST2026_test_clean.json
        task2_1_hard_{TEAM}_{RUN}.json  ← 2.1 hard predictions (used to cascade)
OUTPUT: task2_2_hard_{TEAM}_{RUN}.json  ← hard labels (NO / DIRECT / JUDGEMENTAL)
        task2_2_soft_{TEAM}_{RUN}.json  ← soft labels ({NO, DIRECT, JUDGEMENTAL} floats)

NOTE: Task 2.2 must be run AFTER task 2.1.
      The 2.1 hard predictions are loaded so the LLM receives label_21 as context.
      Non-sexist memes (label_21=NO) get value="NO" in hard and {NO:1,DIRECT:0,JUDGEMENTAL:0} in soft.

Usage:
    python run_task2_2.py \
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

from config   import TEAM_NAME
from prompts  import PROMPT_2_2
from utils    import (
    build_llm, safe_parse, load_test_data, write_output,
    make_hard_record_22, make_soft_record_22, output_filename,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# LOAD 2.1 PREDICTIONS → dict {id_exist: {"label", "soft_score"}}
# ──────────────────────────────────────────────────────────────────────

def load_pred_21(pred_21_path: str) -> dict:
    """
    Load task2_1 hard JSON → dict keyed by id_exist.
    We also read soft file if available for soft_score, else default 0.5 / 1.0.
    """
    with open(pred_21_path, encoding="utf-8") as f:
        records = json.load(f)
    # hard file: value is "YES" or "NO"
    result = {}
    for r in records:
        label = r["value"]
        soft  = 1.0 if label == "YES" else 0.0
        result[str(r["id"])] = {"label": label, "soft_score": soft}
    return result


# ──────────────────────────────────────────────────────────────────────
# CORE CLASSIFIER
# ──────────────────────────────────────────────────────────────────────

def classify_record_22(record: dict, pred_21: dict, chain) -> dict:
    """
    Run subtask 2.2 on a single test record.
    Returns {"id_exist", "hard_label", "p_direct", "p_judgemental"}.
    """
    id_exist = str(record["id_EXIST"])
    text     = record.get("text", "").strip() or "[no text]"

    # get 2.1 result for this record
    p21       = pred_21.get(id_exist, {"label": "NO", "soft_score": 0.0})
    label_21  = p21["label"]
    soft_21   = p21["soft_score"]

    # if not sexist → skip LLM, return NO directly
    if label_21 == "NO":
        log.info(f"[{id_exist}] label_21=NO → skipping LLM, output=NO")
        return {
            "id_exist":     id_exist,
            "hard_label":   "NO",
            "p_direct":     0.0,
            "p_judgemental":0.0,
        }

    try:
        raw = chain.invoke({
            "meme_text": text,
            "label_21":  label_21,
            "soft_21":   round(soft_21, 3),
        })
        pred = safe_parse(raw, "2.2")
    except Exception as e:
        log.error(f"[{id_exist}][2.2] LLM error: {e}")
        pred = {"label": "DIRECT", "p_direct": 0.6, "p_judgemental": 0.4, "reasoning": "llm_error"}

    label      = pred.get("label", "DIRECT")
    p_direct   = float(pred.get("p_direct", 0.5) or 0.5)
    p_judgemental = float(pred.get("p_judgemental", 0.5) or 0.5)
    reasoning  = pred.get("reasoning", "")

    # validate label
    if label not in ("DIRECT", "JUDGEMENTAL", "NO"):
        label = "DIRECT" if p_direct >= 0.5 else "JUDGEMENTAL"

    # ensure probabilities are valid
    p_direct      = max(0.0, min(1.0, p_direct))
    p_judgemental = max(0.0, min(1.0, p_judgemental))
    total = p_direct + p_judgemental
    if total > 0:
        p_direct      = round(p_direct / total, 6)
        p_judgemental = round(1.0 - p_direct, 6)

    log.info(f"[{id_exist}] label={label}  p_direct={p_direct:.3f}  | {reasoning[:80]}")

    return {
        "id_exist":      id_exist,
        "hard_label":    label,
        "p_direct":      p_direct,
        "p_judgemental": p_judgemental,
    }


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def run(test_path: str, pred_21_path: str, output_dir: str,
        model_key: str, run_id: int, max_records: int | None):

    records  = load_test_data(test_path)
    pred_21  = load_pred_21(pred_21_path)
    if max_records:
        records = records[:max_records]
    log.info(f"Loaded {len(records)} test records")
    log.info(f"Loaded {len(pred_21)} task2.1 predictions from {pred_21_path}")
    log.info(f"Model: {model_key} | Run ID: {run_id} | Team: {TEAM_NAME}")

    llm    = build_llm(model_key)
    parser = JsonOutputParser()
    chain  = PROMPT_2_2 | llm | parser

    hard_out, soft_out = [], []

    for i, rec in enumerate(records):
        log.info(f"[{i+1}/{len(records)}] Processing {rec.get('id_EXIST','?')}")
        result = classify_record_22(rec, pred_21, chain)

        hard_out.append(make_hard_record_22(result["id_exist"], result["hard_label"]))
        soft_out.append(make_soft_record_22(
            result["id_exist"], result["p_direct"], result["p_judgemental"]
        ))
        time.sleep(0.05)

    out = Path(output_dir)
    hard_path = out / output_filename("task2", "2", "hard", run_id)
    soft_path = out / output_filename("task2", "2", "soft", run_id)

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
    parser = argparse.ArgumentParser(description="EXIST 2026 — Subtask 2.2")
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
