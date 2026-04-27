"""
EXIST 2026 — Shared Utilities
LLM builder, JSON parse helpers, soft/hard gold helpers, output file writers.
"""

import json
import re
import logging
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

from config import (
    OLLAMA_BASE_URL, MODELS, TEAM_NAME, TEST_CASE,
    LLM_TEMPERATURE, LLM_NUM_PREDICT, CATEGORIES_2_3,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# LLM BUILDER
# ──────────────────────────────────────────────────────────────────────

def build_llm(model_key: str) -> ChatOllama:
    """Return a configured ChatOllama instance."""
    model_name = MODELS[model_key]
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        format="json",
        num_predict=LLM_NUM_PREDICT,
    )


def get_parser() -> JsonOutputParser:
    return JsonOutputParser()


# ──────────────────────────────────────────────────────────────────────
# SAFE JSON PARSE (handles LLM markdown fences / raw strings)
# ──────────────────────────────────────────────────────────────────────

def safe_parse(raw, subtask: str) -> dict:
    """
    If LLM returned a string instead of dict (parser failure), recover.
    Returns a safe fallback dict on failure.
    """
    if isinstance(raw, dict):
        return raw
    try:
        cleaned = re.sub(r"```json|```", "", str(raw)).strip()
        return json.loads(cleaned)
    except Exception:
        log.warning(f"[{subtask}] JSON parse failure. Raw: {str(raw)[:200]!r}")
        if subtask == "2.1":
            return {"label": "NO", "soft_score": 0.5, "reasoning": "parse_error"}
        elif subtask == "2.2":
            return {"label": None, "p_direct": None, "p_judgemental": None, "reasoning": "parse_error"}
        else:
            return {
                "labels": [],
                "p_ideological_inequality": 0.0,
                "p_stereotyping_dominance": 0.0,
                "p_objectification": 0.0,
                "p_sexual_violence": 0.0,
                "p_misogyny_non_sexual_violence": 0.0,
                "reasoning": "parse_error",
            }


# ──────────────────────────────────────────────────────────────────────
# DATA LOADER
# ──────────────────────────────────────────────────────────────────────

def load_test_data(test_json_path: str) -> list[dict]:
    """
    Load the EXIST 2026 test JSON.
    Handles both dict-of-records and list-of-records formats.
    Returns a flat list of record dicts.
    """
    with open(test_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return list(raw.values())
    return raw


# ──────────────────────────────────────────────────────────────────────
# SOFT/HARD GOLD HELPERS (used during training evaluation only)
# ──────────────────────────────────────────────────────────────────────

def compute_soft_gold_21(labels: list[str]) -> float:
    """Fraction of YES votes."""
    return labels.count("YES") / len(labels)

def compute_hard_gold_21(labels: list[str]) -> str:
    """Majority vote; tie → NO per EXIST convention."""
    return "YES" if labels.count("YES") > len(labels) / 2 else "NO"

def compute_soft_gold_22(labels: list[str]) -> dict:
    """Fraction of DIRECT vs JUDGEMENTAL among valid (non-dash) labels."""
    valid = [l for l in labels if l not in ("-", "UNKNOWN")]
    if not valid:
        return {"p_direct": None, "p_judgemental": None}
    return {
        "p_direct":      valid.count("DIRECT") / len(valid),
        "p_judgemental": valid.count("JUDGEMENTAL") / len(valid),
    }

def compute_soft_gold_23(labels_per_annotator: list) -> dict:
    """Per-category fraction of annotators who flagged it."""
    n = len(labels_per_annotator)
    result = {}
    for cat in CATEGORIES_2_3:
        count = sum(1 for ann_labels in labels_per_annotator if cat in ann_labels)
        result[cat] = count / n
    return result


# ──────────────────────────────────────────────────────────────────────
# OUTPUT FILE WRITERS
# These write the exact JSON format required by EXIST 2026 / PyEvALL
# ──────────────────────────────────────────────────────────────────────

def write_output(records: list[dict], out_path: str):
    """Write a list of {test_case, id, value} dicts to a JSON file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"Written {len(records)} records → {out_path}")


def make_hard_record_21(id_exist: str, label: str) -> dict:
    return {"test_case": TEST_CASE, "id": str(id_exist), "value": label}

def make_soft_record_21(id_exist: str, p_yes: float) -> dict:
    p_yes = round(max(0.0, min(1.0, p_yes)), 6)
    p_no  = round(1.0 - p_yes, 6)
    return {"test_case": TEST_CASE, "id": str(id_exist), "value": {"YES": p_yes, "NO": p_no}}


def make_hard_record_22(id_exist: str, label: str) -> dict:
    # label must be "NO", "DIRECT", or "JUDGEMENTAL"
    return {"test_case": TEST_CASE, "id": str(id_exist), "value": label}

def make_soft_record_22(id_exist: str, p_direct: float, p_judgemental: float) -> dict:
    # Must sum to 1.0, three keys including NO
    p_d = round(max(0.0, min(1.0, p_direct or 0.0)), 6)
    p_j = round(max(0.0, min(1.0, p_judgemental or 0.0)), 6)
    # normalise so they sum to 1
    total = p_d + p_j
    if total > 0:
        p_d = round(p_d / total, 6)
        p_j = round(1.0 - p_d, 6)
    p_no = 0.0
    return {"test_case": TEST_CASE, "id": str(id_exist),
            "value": {"DIRECT": p_d, "JUDGEMENTAL": p_j, "NO": p_no}}


def make_hard_record_23(id_exist: str, labels: list[str]) -> dict:
    # labels is a list e.g. ["IDEOLOGICAL-INEQUALITY", "OBJECTIFICATION"] or ["NO"]
    if not labels:
        labels = ["NO"]
    return {"test_case": TEST_CASE, "id": str(id_exist), "value": labels}

def make_soft_record_23(id_exist: str, scores: dict) -> dict:
    # scores: dict of category → float (independent, don't sum to 1)
    # Must include NO and all 5 categories
    value = {
        "NO":                            round(scores.get("NO", 0.0), 6),
        "IDEOLOGICAL-INEQUALITY":        round(scores.get("IDEOLOGICAL-INEQUALITY", 0.0), 6),
        "STEREOTYPING-DOMINANCE":        round(scores.get("STEREOTYPING-DOMINANCE", 0.0), 6),
        "OBJECTIFICATION":               round(scores.get("OBJECTIFICATION", 0.0), 6),
        "SEXUAL-VIOLENCE":               round(scores.get("SEXUAL-VIOLENCE", 0.0), 6),
        "MISOGYNY-NON-SEXUAL-VIOLENCE":  round(scores.get("MISOGYNY-NON-SEXUAL-VIOLENCE", 0.0), 6),
    }
    return {"test_case": TEST_CASE, "id": str(id_exist), "value": value}


# ──────────────────────────────────────────────────────────────────────
# OUTPUT FILENAME HELPER
# ──────────────────────────────────────────────────────────────────────

def output_filename(task: str, subtask: str, label_type: str, run_id: int = 1) -> str:
    """
    Builds the official EXIST 2026 filename.
    e.g. task2_1_hard_YOURTEAMNAME_1.json
    """
    return f"{task}_{subtask}_{label_type}_{TEAM_NAME}_{run_id}.json"
