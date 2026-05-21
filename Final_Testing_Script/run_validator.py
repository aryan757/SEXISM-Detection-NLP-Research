"""
EXIST 2026 — Format Validator (updated)
========================================
Drop this file next to your submission/ folder and run:

    python run_validator.py

It will call the official exist2025_format_val_V0.2.py logic
pointed at ./submission/ and print a clean pass/fail summary.
"""

import json
import jsonschema
from jsonschema import validate
from os import listdir
from os.path import isfile, join
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────
SUBMISSION_DIR = "/Users/aryan/Desktop/aryan_Exist_2026_dataset/Final_Testing_Script/exist2026_AryanSomnathBanerjee_run2"           # folder with your 6 JSON files
# ───────────────────────────────────────────────────────────────────────

ID         = "id"
TEST_CASE  = "test_case"
VALUE      = "value"

TASK2_1 = "task2_1"
TASK2_2 = "task2_2"
TASK2_3 = "task2_3"
TASK3_1 = "task3_1"
TASK3_2 = "task3_2"
TASK3_3 = "task3_3"

LIST_LABELS_SUBTASK1   = ["NO", "YES"]
LIST_LABELS_SUBTASK2_2 = ["NO", "JUDGEMENTAL", "DIRECT"]
LIST_LABELS_SUBTASK3   = [
    "NO",
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
    "SEXUAL-VIOLENCE",
    "OBJECTIFICATION",
]

FORMAT_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "test_case": {"type": "string"},
            "id":        {"type": "string"},
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    {"type": "integer"},
                    {
                        "type": "object",
                        "patternProperties": {"^.*$": {"type": "number"}},
                    },
                ]
            },
        },
        "required": ["test_case", "id", "value"],
        "additionalProperties": False,
    },
}


def parse_json(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as e:
        print(f"  ✗ JSON parse error: {e}")
        return False

    try:
        validate(instance=data, schema=FORMAT_JSON_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        print(f"  ✗ Schema validation error: {e.message}")
        return False

    return True


def validate_file(filepath: str, filename: str) -> list[str]:
    """
    Returns a list of error strings. Empty list = all good.
    """
    errors = []

    # ── filename format ───────────────────────────────────────────────
    parts = filename.replace(".json", "").split("_")
    if len(parts) != 5:
        errors.append(
            f"Filename must have exactly 5 underscore-separated parts "
            f"(got {len(parts)}): {filename}\n"
            f"  Expected format: task2_1_hard_TEAMNAME_1.json"
        )
        return errors   # can't continue without valid task name

    task = parts[0] + "_" + parts[1]   # e.g. "task2_1"

    # ── JSON schema ───────────────────────────────────────────────────
    if not parse_json(filepath):
        errors.append("JSON schema invalid — see error above")
        return errors

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── test_case field ───────────────────────────────────────────────
    wrong_tc = [r for r in data if r.get("test_case") != "EXIST2025"]
    if wrong_tc:
        errors.append(
            f"test_case must be 'EXIST2025'. "
            f"Found wrong values in {len(wrong_tc)} records. "
            f"Example: '{wrong_tc[0].get('test_case')}'"
        )

    # ── instance count ────────────────────────────────────────────────
    expected = None
    if task in (TASK2_1, TASK2_2, TASK2_3):
        expected = 1053
    elif task in (TASK3_1, TASK3_2, TASK3_3):
        expected = 674

    if expected and len(data) != expected:
        errors.append(
            f"Wrong number of instances: got {len(data)}, expected {expected}"
        )

    # ── ID range check ────────────────────────────────────────────────
    bad_ids = []
    data_es, data_en = {}, {}

    for rec in data:
        rid = rec.get(ID, "")
        try:
            rid_int = int(rid)
        except (ValueError, TypeError):
            bad_ids.append(rid)
            continue

        if task in (TASK2_1, TASK2_2, TASK2_3):
            if 310001 <= rid_int <= 310540:
                data_es[rid] = rec
            elif 410001 <= rid_int <= 410513:
                data_en[rid] = rec
            else:
                bad_ids.append(rid)
        elif task in (TASK3_1, TASK3_2, TASK3_3):
            if 320001 <= rid_int <= 320304:
                data_es[rid] = rec
            elif 420001 <= rid_int <= 420370:
                data_en[rid] = rec
            else:
                bad_ids.append(rid)

    if bad_ids:
        errors.append(f"IDs out of valid range: {bad_ids[:5]} {'...' if len(bad_ids)>5 else ''}")

    # ── Spanish / English count check ────────────────────────────────
    if task in (TASK2_1, TASK2_2, TASK2_3):
        if len(data_es) != 540:
            errors.append(f"Spanish meme count: got {len(data_es)}, expected 540")
        if len(data_en) != 513:
            errors.append(f"English meme count: got {len(data_en)}, expected 513")

    # ── value / label checks ─────────────────────────────────────────
    for rec in data:
        val = rec.get(VALUE)
        rid = rec.get(ID, "?")

        if isinstance(val, str):
            # hard label
            if task in (TASK2_1, TASK3_1):
                if val not in LIST_LABELS_SUBTASK1:
                    errors.append(f"[id={rid}] Invalid hard label for {task}: '{val}'. Must be YES or NO")
            elif task in (TASK2_2, TASK3_2):
                if val not in LIST_LABELS_SUBTASK2_2:
                    errors.append(f"[id={rid}] Invalid hard label for {task}: '{val}'. Must be NO, DIRECT or JUDGEMENTAL")

        elif isinstance(val, list):
            # hard multi-label (task 2.3 / 3.3)
            if task in (TASK2_3, TASK3_3):
                for lbl in val:
                    if lbl not in LIST_LABELS_SUBTASK3:
                        errors.append(f"[id={rid}] Invalid category label: '{lbl}'")
            else:
                errors.append(f"[id={rid}] Array value not expected for task {task}")

        elif isinstance(val, dict):
            # soft label
            if task in (TASK2_1, TASK3_1):
                if len(val) != 2:
                    errors.append(f"[id={rid}] Soft label must have exactly 2 keys (YES, NO), got {len(val)}")
                for k in val:
                    if k not in LIST_LABELS_SUBTASK1:
                        errors.append(f"[id={rid}] Invalid soft key: '{k}'. Must be YES or NO")
                total = sum(float(v) for v in val.values())
                if total > 1.001:
                    errors.append(f"[id={rid}] Soft probabilities sum > 1.0: {total:.4f}")

            elif task in (TASK2_2, TASK3_2):
                if len(val) != 3:
                    errors.append(f"[id={rid}] Soft label must have exactly 3 keys, got {len(val)}")
                for k in val:
                    if k not in LIST_LABELS_SUBTASK2_2:
                        errors.append(f"[id={rid}] Invalid soft key: '{k}'")
                total = sum(float(v) for v in val.values())
                if total > 1.001:
                    errors.append(f"[id={rid}] Soft probabilities sum > 1.0: {total:.4f}")

            elif task in (TASK2_3, TASK3_3):
                if len(val) != 6:
                    errors.append(f"[id={rid}] Soft label must have exactly 6 keys, got {len(val)}")
                for k in val:
                    if k not in LIST_LABELS_SUBTASK3:
                        errors.append(f"[id={rid}] Invalid soft key: '{k}'")
                # Note: for multi-label, sum does NOT need to be 1.0 — no sum check here

        else:
            errors.append(f"[id={rid}] Unexpected value type: {type(val)}")

    return errors


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    sub = Path(SUBMISSION_DIR)
    if not sub.exists():
        print(f"ERROR: Folder not found → {sub.resolve()}")
        return

    files = sorted([f for f in sub.iterdir() if f.suffix == ".json"])
    if not files:
        print(f"No JSON files found in {sub.resolve()}")
        return

    print(f"\nValidating {len(files)} files in: {sub.resolve()}")
    print("=" * 60)

    all_passed   = True
    passed_files = []
    failed_files = []

    for fpath in files:
        print(f"\n  FILE: {fpath.name}")
        errors = validate_file(str(fpath), fpath.name)

        if not errors:
            print(f"  ✅ PASSED — no errors")
            passed_files.append(fpath.name)
        else:
            all_passed = False
            failed_files.append(fpath.name)
            for err in errors:
                print(f"  ✗  {err}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed : {len(passed_files)} / {len(files)}")
    print(f"  Failed : {len(failed_files)} / {len(files)}")

    if passed_files:
        print("\n  ✅ PASSED:")
        for f in passed_files:
            print(f"      {f}")

    if failed_files:
        print("\n  ✗  FAILED:")
        for f in failed_files:
            print(f"      {f}")

    if all_passed:
        print("\n🎉 ALL FILES PASSED — ready to submit!")
        print("\nNext steps:")
        print("  1. Rename folder:  mv submission exist2026_YOURTEAMNAME")
        print("  2. Zip it:         zip -r exist2026_YOURTEAMNAME.zip exist2026_YOURTEAMNAME/")
        print("  3. Submit at:      https://forms.gle/5hY91c7aBv563oZM7")
    else:
        print("\n⚠️  Fix the errors above, then re-run this script.")
        print("   If test_case or spaces are the issue, run:  python fix_and_rename.py")


if __name__ == "__main__":
    main()
