"""
EXIST 2026 — Master Runner
===========================
Runs ALL three subtasks in sequence and produces all 6 submission files:

    submission/
    ├── task2_1_hard_{TEAM}_{RUN}.json
    ├── task2_1_soft_{TEAM}_{RUN}.json
    ├── task2_2_hard_{TEAM}_{RUN}.json
    ├── task2_2_soft_{TEAM}_{RUN}.json
    ├── task2_3_hard_{TEAM}_{RUN}.json
    └── task2_3_soft_{TEAM}_{RUN}.json

Usage (full test run):
    python run_all.py \
        --test_path  path/to/EXIST2026_test_clean.json \
        --output_dir ./submission \
        --model_key  qwen \
        --run_id     1

Usage (quick test on first 10 records):
    python run_all.py \
        --test_path    path/to/EXIST2026_test_clean.json \
        --output_dir   ./submission \
        --model_key    qwen \
        --run_id       1 \
        --max_records  10

After this, validate with:
    python exist2025_format_val_V0.2.py
    (update the path inside that script to point to ./submission/)

Then zip the submission folder and upload to:
    https://forms.gle/5hY91c7aBv563oZM7
"""

import argparse
import logging
from pathlib import Path

from config import TEAM_NAME
from utils  import output_filename

import run_task2_1
import run_task2_2
import run_task2_3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_all(test_path: str, output_dir: str, model_key: str,
            run_id: int, max_records: int | None):

    log.info("=" * 60)
    log.info(f"EXIST 2026 — Full Pipeline")
    log.info(f"  Team      : {TEAM_NAME}")
    log.info(f"  Model     : {model_key}")
    log.info(f"  Run ID    : {run_id}")
    log.info(f"  Test data : {test_path}")
    log.info(f"  Output    : {output_dir}")
    if max_records:
        log.info(f"  Records   : {max_records} (limited)")
    log.info("=" * 60)

    # ── Step 1: Task 2.1 ─────────────────────────────────────────────
    log.info("\n▶ STEP 1/3 — Subtask 2.1: Sexism Identification")
    run_task2_1.run(
        test_path   = test_path,
        output_dir  = output_dir,
        model_key   = model_key,
        run_id      = run_id,
        max_records = max_records,
    )

    # path to the hard 2.1 predictions (needed by 2.2 and 2.3)
    pred_21_path = str(
        Path(output_dir) / output_filename("task2", "1", "hard", run_id)
    )

    # ── Step 2: Task 2.2 ─────────────────────────────────────────────
    log.info("\n▶ STEP 2/3 — Subtask 2.2: Source Intention")
    run_task2_2.run(
        test_path    = test_path,
        pred_21_path = pred_21_path,
        output_dir   = output_dir,
        model_key    = model_key,
        run_id       = run_id,
        max_records  = max_records,
    )

    # ── Step 3: Task 2.3 ─────────────────────────────────────────────
    log.info("\n▶ STEP 3/3 — Subtask 2.3: Sexism Categorisation")
    run_task2_3.run(
        test_path    = test_path,
        pred_21_path = pred_21_path,
        output_dir   = output_dir,
        model_key    = model_key,
        run_id       = run_id,
        max_records  = max_records,
    )

    # ── Summary ───────────────────────────────────────────────────────
    out = Path(output_dir)
    files = [
        output_filename("task2", "1", "hard", run_id),
        output_filename("task2", "1", "soft", run_id),
        output_filename("task2", "2", "hard", run_id),
        output_filename("task2", "2", "soft", run_id),
        output_filename("task2", "3", "hard", run_id),
        output_filename("task2", "3", "soft", run_id),
    ]

    log.info("\n" + "=" * 60)
    log.info("ALL DONE — Submission files:")
    for fname in files:
        fpath = out / fname
        status = "✓" if fpath.exists() else "✗ MISSING"
        log.info(f"  {status}  {fpath}")
    log.info("=" * 60)
    log.info(f"\nNext steps:")
    log.info(f"  1. Run format validator:  python exist2025_format_val_V0.2.py")
    log.info(f"  2. Create zip:  zip -r exist2026_{TEAM_NAME}.zip {output_dir}/")
    log.info(f"  3. Submit at:   https://forms.gle/5hY91c7aBv563oZM7")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXIST 2026 — Full Pipeline (all 3 subtasks)")
    parser.add_argument("--test_path",   required=True,  help="Path to EXIST2026_test_clean.json")
    parser.add_argument("--output_dir",  default="./submission", help="Output directory for JSON files")
    parser.add_argument("--model_key",   default="", help="Model key: moondream | qwen | llava | unsloth_gemma")
    parser.add_argument("--run_id",      type=int, default=1, help="Run number 1-3 (for multiple submissions)")
    parser.add_argument("--max_records", type=int, default=None, help="Limit number of records (useful for testing)")
    args = parser.parse_args()

    run_all(
        test_path   = args.test_path,
        output_dir  = args.output_dir,
        model_key   = args.model_key,
        run_id      = args.run_id,
        max_records = args.max_records,
    )
