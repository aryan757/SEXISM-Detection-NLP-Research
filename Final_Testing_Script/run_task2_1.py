"""
EXIST 2026 — Subtask 2.1: Sexism Identification in Memes
=========================================================
INPUT : EXIST2026_test_clean.json
OUTPUT: task2_1_hard_{TEAM}_{RUN}.json   ← hard labels (YES / NO)
        task2_1_soft_{TEAM}_{RUN}.json   ← soft labels ({YES: float, NO: float})

Usage:
    python run_task2_1.py \
        --test_path  path/to/EXIST2026_test_clean.json \
        --output_dir ./submission \
        --model_key  qwen \
        --run_id     1
"""

import argparse
import json
import logging
import time
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser

from config   import TEAM_NAME
from prompts  import PROMPT_2_1
from utils    import (
    build_llm, safe_parse, load_test_data, write_output,
    make_hard_record_21, make_soft_record_21, output_filename,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CORE CLASSIFIER
# ──────────────────────────────────────────────────────────────────────

def classify_record_21(record: dict, chain) -> dict:
    """
    Run subtask 2.1 on a single test record.
    Returns {"id_exist", "hard_label", "soft_score"}.
    """
    id_exist = str(record["id_EXIST"])
    text     = record.get("text", "").strip() or "[no text]"

    try:
        raw = chain.invoke({"meme_text": text})
        pred = safe_parse(raw, "2.1")
    except Exception as e:
        log.error(f"[{id_exist}][2.1] LLM error: {e}")
        pred = {"label": "NO", "soft_score": 0.5, "reasoning": "llm_error"}

    label      = pred.get("label", "NO")
    soft_score = float(pred.get("soft_score", 0.5))
    reasoning  = pred.get("reasoning", "")

    # safety clamp
    label      = label if label in ("YES", "NO") else "NO"
    soft_score = max(0.0, min(1.0, soft_score))

    log.info(f"[{id_exist}] label={label}  soft={soft_score:.3f}  | {reasoning[:80]}")

    return {
        "id_exist":   id_exist,
        "hard_label": label,
        "soft_score": soft_score,
    }


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def run(test_path: str, output_dir: str, model_key: str, run_id: int, max_records: int | None):

    records = load_test_data(test_path)
    if max_records:
        records = records[:max_records]
    log.info(f"Loaded {len(records)} test records from {test_path}")
    log.info(f"Model: {model_key} | Run ID: {run_id} | Team: {TEAM_NAME}")

    # build LangChain chain
    llm    = build_llm(model_key)
    parser = JsonOutputParser()
    chain  = PROMPT_2_1 | llm | parser

    hard_out, soft_out = [], []

    for i, rec in enumerate(records):
        log.info(f"[{i+1}/{len(records)}] Processing {rec.get('id_EXIST', '?')}")
        result = classify_record_21(rec, chain)

        hard_out.append(make_hard_record_21(result["id_exist"], result["hard_label"]))
        soft_out.append(make_soft_record_21(result["id_exist"], result["soft_score"]))

        time.sleep(0.05)   # avoid hammering Ollama

    # write files
    out = Path(output_dir)
    hard_path = out / output_filename("task2", "1", "hard", run_id)
    soft_path = out / output_filename("task2", "1", "soft", run_id)

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
    parser = argparse.ArgumentParser(description="EXIST 2026 — Subtask 2.1")
    parser.add_argument("--test_path",   required=True,  help="Path to EXIST2026_test_clean.json")
    parser.add_argument("--output_dir",  default="./submission", help="Output directory")
    parser.add_argument("--model_key",   default="qwen", help="Model key from config.py")
    parser.add_argument("--run_id",      type=int, default=1, help="Run ID (1-3)")
    parser.add_argument("--max_records", type=int, default=None, help="Limit records (for testing)")
    args = parser.parse_args()

    run(
        test_path   = args.test_path,
        output_dir  = args.output_dir,
        model_key   = args.model_key,
        run_id      = args.run_id,
        max_records = args.max_records,
    )
