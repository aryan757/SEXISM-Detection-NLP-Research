"""
EXIST 2026 — Fix Submission Files
===================================
This script fixes TWO issues found in the submission folder:

  1. test_case field is wrong (e.g. "Aryan Somnath Banerjee") → must be "EXIST2025"
  2. Filenames have spaces (e.g. "task2_1_hard_Aryan Somnath Banerjee_1.json")
     → must be "task2_1_hard_AryanSomnathBanerjee_1.json"

Run this ONCE from inside your Final_Testing_Script folder:

    python fix_and_rename.py

It will:
  - Read every JSON in ./submission/
  - Replace every "test_case" value with "EXIST2025"
  - Save the fixed JSON back
  - Rename the file to remove spaces from team name
"""

import json
import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────
SUBMISSION_DIR  = "/Users/aryan/Desktop/aryan_Exist_2026_dataset/Final_Testing_Script/submission"          # folder with your 6 JSON files
CORRECT_TEST_CASE = "EXIST2025"           # must be exactly this string
# ─────────────────────────────────────────────────────────────────────


def fix_json_file(filepath: Path) -> bool:
    """Load JSON, fix all test_case values, save back. Returns True if changed."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for record in data:
        if record.get("test_case") != CORRECT_TEST_CASE:
            record["test_case"] = CORRECT_TEST_CASE
            changed = True

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return changed


def fix_filename(filepath: Path) -> Path:
    """
    Remove spaces from filename.
    e.g. task2_1_hard_Aryan Somnath Banerjee_1.json
      →  task2_1_hard_AryanSomnathBanerjee_1.json
    """
    original_name = filepath.name
    fixed_name    = original_name.replace(" ", "")

    if original_name == fixed_name:
        return filepath  # no change needed

    new_path = filepath.parent / fixed_name
    filepath.rename(new_path)
    return new_path


def main():
    sub_dir = Path(SUBMISSION_DIR)

    if not sub_dir.exists():
        print(f"ERROR: Submission folder not found: {sub_dir.resolve()}")
        return

    json_files = sorted([f for f in sub_dir.iterdir() if f.suffix == ".json"])

    if not json_files:
        print(f"No JSON files found in {sub_dir.resolve()}")
        return

    print(f"Found {len(json_files)} JSON files in {sub_dir.resolve()}")
    print("=" * 60)

    for fpath in json_files:
        print(f"\nProcessing: {fpath.name}")

        # Step 1 — fix test_case inside JSON
        changed = fix_json_file(fpath)
        if changed:
            print(f"  ✓ Fixed test_case → EXIST2025")
        else:
            print(f"  ✓ test_case already correct")

        # Step 2 — rename file to remove spaces
        new_path = fix_filename(fpath)
        if new_path != fpath:
            print(f"  ✓ Renamed: {fpath.name}")
            print(f"         → {new_path.name}")
        else:
            print(f"  ✓ Filename already has no spaces")

    print("\n" + "=" * 60)
    print("Done! Fixed files:")
    for f in sorted(sub_dir.iterdir()):
        if f.suffix == ".json":
            print(f"  {f.name}")

    print("\nNext step → run the validator:")
    print("  python exist2025_format_val_V0.2.py")


if __name__ == "__main__":
    main()
