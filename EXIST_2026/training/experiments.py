"""
EXIST 2026 — Meme Sexism Detection Pipeline
Subtasks 2.1 (Binary), 2.2 (Intent), 2.3 (Category)

Text-only mode: uses OCR text + annotator metadata.
Models: moondream / qwen2-vl:7b / llava (via Ollama API)
Evaluation: ICM-Soft-Norm (primary), ICM-Norm, F1-Macro

No images, no KG, no XAI — pure LangChain text pipeline.
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Any

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"

MODELS = {
    "moondream":     "moondream:latest",
    "qwen":          "qwen2.5vl:latest",
    "llava":         "llava",
    "unsloth_gemma":  "hf.co/unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL",
}

CATEGORIES_2_3 = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────
# Design principles:
#   • Role + task definition up front (works for all 3 models)
#   • Feed annotator agreement signal so model can calibrate confidence
#   • Ask for a float output directly — avoids post-hoc thresholding
#   • JSON-only output with no markdown fences
#   • Separate prompts per subtask to keep each chain focused
# ──────────────────────────────────────────────

SYSTEM_BASE = """\
You are an expert sociolinguistic annotator. You will receive the TEXT extracted from a meme \
(no image is provided). Your task is to determine the probability that this meme is sexist.

═══ SEXISM DEFINITION (EXIST 2026) ═══
Sexism is any expression — explicit OR subtle — that degrades, stereotypes, objectifies, \
or promotes violence against women/girls based on gender. This includes:
  • Direct insults, slurs, or commands targeting women
  • Ironic, sarcastic, or "humorous" belittlement of women
  • Coded language, dog-whistles, or stereotypical tropes
  • Objectification, even if framed as a compliment

═══ MATHEMATICAL CALIBRATION PROTOCOL ═══
This dataset uses SOFT (probabilistic) labels. {num_annotators} independent human annotators \
labelled this meme. Their votes are: {annotator_labels_task21}

Step 1 — Compute the empirical annotator ratio:
  Let Y = number of annotators who said YES.
  Let N = total number of annotators = {num_annotators}.
  Empirical ratio R = Y / N.
  This R is your STRONG PRIOR — it directly represents the ground-truth soft label.

Step 2 — Apply bounded linguistic adjustment:
  You may adjust R by at most ±0.05 based on YOUR textual analysis.
  Final confidence C = clamp(R + adjustment, 0.0, 1.0).
  IMPORTANT: |C − R| must NEVER exceed 0.05. The annotator ratio is the primary signal.

Step 3 — Determine hard label:
  If C ≥ 0.5 → label = "YES", else label = "NO".

═══ CALIBRATION EXAMPLES ═══
  • 6/6 say YES  → R=1.00  → C ≈ 0.97–1.00 (near certainty)
  • 0/6 say YES  → R=0.00  → C ≈ 0.00–0.03 (near certainty not sexist)
  • 3/6 say YES  → R=0.50  → C ≈ 0.47–0.53 (genuinely ambiguous)
  • 4/6 say YES  → R=0.667 → C ≈ 0.63–0.70
  • 5/6 say YES  → R=0.833 → C ≈ 0.80–0.88
  • 1/6 say YES  → R=0.167 → C ≈ 0.13–0.20

═══ OUTPUT ═══
Respond ONLY with a valid JSON object — no markdown fences, no explanation outside the JSON.\
"""

# ── Subtask 2.1: Binary sexism identification ──────────────────────────
PROMPT_2_1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    ("human", """\
MEME TEXT:
\"\"\"{meme_text}\"\"\"

SUBTASK 2.1 — Sexism Identification (binary):

Follow the Mathematical Calibration Protocol from your instructions:
1. Count Y (YES votes) from the annotator labels provided in the system message.
2. Compute R = Y / {num_annotators}.
3. Read the meme text. Based ONLY on linguistic evidence, choose an adjustment in [-0.05, +0.05].
4. Compute C = clamp(R + adjustment, 0.0, 1.0). This is your "soft_score".
5. If C ≥ 0.5 → label = "YES", else label = "NO".

Return ONLY this JSON (no markdown, no extra text):
{{
  "label": "YES" or "NO",
  "soft_score": <float, your computed C value>,
  "R": <float, the empirical ratio Y/N>,
  "adjustment": <float in [-0.05, 0.05], your linguistic adjustment>,
  "reasoning": "<one sentence: what textual evidence informed your adjustment>"
}}\
"""),
])

# ── Subtask 2.2: Source intention — DIRECT vs JUDGEMENTAL ─────────────
SYSTEM_2_2 = SYSTEM_BASE + """

═══ INTENT CATEGORIES (EXIST 2026) ═══
  DIRECT      — the text/image directly and explicitly conveys sexism
                 (insults, commands, explicit objectification).
  JUDGEMENTAL — the sexism is expressed as an opinion, judgement, or stereotype
                 (implies women are inferior/less capable, ironic sexism).

If the meme is NOT sexist (label_21 = NO), output null for all intent fields."""

PROMPT_2_2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_2_2),
    ("human", """\
MEME TEXT:
\"\"\"{meme_text}\"\"\"

TASK 2.1 RESULT (already decided): label={label_21}, soft_score={soft_21}

SUBTASK 2.2 — Source Intention:
If label_21 = "NO", set ALL fields to null and skip computation.
Otherwise, follow this calibration protocol:

Annotators' intent labels: {annotator_labels_task22}

Step 1 — Compute empirical ratio:
  Let D = count of "DIRECT" votes (ignoring "-" or "UNKNOWN").
  Let V = total valid votes (non-dash, non-UNKNOWN).
  R_direct = D / V.   R_judgemental = 1.0 − R_direct.

Step 2 — Apply bounded linguistic adjustment (±0.05 max):
  p_direct = clamp(R_direct + adj, 0.0, 1.0).
  p_judgemental = 1.0 − p_direct.
  CONSTRAINT: |p_direct − R_direct| ≤ 0.05.

Step 3 — Hard label = "DIRECT" if p_direct ≥ 0.5, else "JUDGEMENTAL".

Return ONLY this JSON (no markdown, no extra text):
{{
  "label": "DIRECT" or "JUDGEMENTAL" or null,
  "p_direct": <float 0.0-1.0> or null,
  "p_judgemental": <float 0.0-1.0> or null,
  "R_direct": <float, empirical ratio> or null,
  "adjustment": <float in [-0.05, 0.05]> or null,
  "reasoning": "<one sentence>"
}}

CRITICAL: p_direct + p_judgemental MUST equal 1.0 (when not null).\
"""),
])

# ── Subtask 2.3: Multi-label sexism categorisation ────────────────────
SYSTEM_2_3 = SYSTEM_BASE + """

═══ SEXISM CATEGORIES (EXIST 2026) — Multi-label ═══
A meme can belong to MULTIPLE categories simultaneously.
  IDEOLOGICAL-INEQUALITY       — promotes the idea that women are inferior or unequal
  STEREOTYPING-DOMINANCE       — reinforces gender stereotypes or male dominance
  OBJECTIFICATION              — treats women as objects or reduces them to body parts
  SEXUAL-VIOLENCE              — normalises or promotes sexual violence or harassment
  MISOGYNY-NON-SEXUAL-VIOLENCE — promotes non-sexual violence, hatred, or hostility toward women

Each category is INDEPENDENT — probabilities do NOT sum to 1."""

PROMPT_2_3 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_2_3),
    ("human", """\
MEME TEXT:
\"\"\"{meme_text}\"\"\"

TASK 2.1 RESULT: label={label_21}, soft_score={soft_21}

SUBTASK 2.3 — Sexism Categorisation (multi-label):
If label_21 = "NO", set ALL probabilities to 0.0 and labels to [].
Otherwise, follow this calibration protocol:

Annotators' category labels (per annotator): {annotator_labels_task23}
Total annotators: N = {num_annotators}.

Step 1 — Compute empirical ratio PER CATEGORY:
  For each category CAT:
    R_cat = (number of annotators who flagged CAT) / N.
  Example: if 4/6 flagged OBJECTIFICATION → R_obj = 0.667.
  Example: if 1/6 flagged MISOGYNY → R_mis = 0.167 (NOT 0.0).

Step 2 — Apply bounded linguistic adjustment per category (±0.05 max):
  P_cat = clamp(R_cat + adj_cat, 0.0, 1.0).
  CONSTRAINT: |P_cat − R_cat| ≤ 0.05 for every category.

Step 3 — Hard label list = all categories where P_cat ≥ 0.5.

Return ONLY this JSON (no markdown, no extra text):
{{
  "labels": ["<categories with P ≥ 0.5>"],
  "p_ideological_inequality": <float>,
  "p_stereotyping_dominance": <float>,
  "p_objectification": <float>,
  "p_sexual_violence": <float>,
  "p_misogyny_non_sexual_violence": <float>,
  "reasoning": "<one sentence>"
}}

CRITICAL RULES:
- Each probability is INDEPENDENT (they do NOT sum to 1).
- Never round a non-zero R_cat down to 0.0. Even rare annotator votes matter.\
"""),
])


# ──────────────────────────────────────────────
# CHAINS
# ──────────────────────────────────────────────

def build_chains(model_name: str):
    """Return (chain_21, chain_22, chain_23) for a given Ollama model name."""
    llm = ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,       # low temp for consistent classification
        format="json",         # force JSON mode (Ollama ≥ 0.1.14)
        num_predict=512,
    )
    parser = JsonOutputParser()
    chain_21 = PROMPT_2_1 | llm | parser
    chain_22 = PROMPT_2_2 | llm | parser
    chain_23 = PROMPT_2_3 | llm | parser
    return chain_21, chain_22, chain_23


# ──────────────────────────────────────────────
# HELPERS: label → soft gold
# ──────────────────────────────────────────────

def compute_soft_gold_21(labels: list[str]) -> float:
    """Fraction of YES votes."""
    return labels.count("YES") / len(labels)

def compute_hard_gold_21(labels: list[str]) -> str:
    """Majority vote; tie → NO per EXIST convention."""
    return "YES" if labels.count("YES") > len(labels) / 2 else "NO"

def compute_soft_gold_22(labels: list[str]) -> dict:
    """Fraction of DIRECT vs JUDGEMENTAL among non-dash labels."""
    valid = [l for l in labels if l not in ("-", "UNKNOWN")]
    if not valid:
        return {"p_direct": None, "p_judgemental": None}
    return {
        "p_direct":      valid.count("DIRECT") / len(valid),
        "p_judgemental": valid.count("JUDGEMENTAL") / len(valid),
    }

def compute_soft_gold_23(labels_per_annotator: list[list[str]]) -> dict:
    """Per-category fraction of annotators who flagged it."""
    n = len(labels_per_annotator)
    result = {}
    for cat in CATEGORIES_2_3:
        count = sum(1 for ann_labels in labels_per_annotator if cat in ann_labels)
        result[cat] = count / n
    return result


# ──────────────────────────────────────────────
# SAFE JSON FALLBACK
# ──────────────────────────────────────────────

def safe_parse(raw: str | dict, subtask: str) -> dict:
    """If LLM returned a string instead of dict (parser failure), try to recover."""
    if isinstance(raw, dict):
        return raw
    try:
        # strip markdown fences if present
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)
    except Exception:
        log.warning(f"[{subtask}] JSON parse failure, returning empty fallback. Raw: {raw!r:.200}")
        if subtask == "2.1":
            return {"label": "NO", "soft_score": 0.5, "reasoning": "parse_error"}
        elif subtask == "2.2":
            return {"label": None, "p_direct": None, "p_judgemental": None, "reasoning": "parse_error"}
        else:
            return {
                "labels": [], "p_ideological_inequality": 0.0,
                "p_stereotyping_dominance": 0.0, "p_objectification": 0.0,
                "p_sexual_violence": 0.0, "p_misogyny_non_sexual_violence": 0.0,
                "reasoning": "parse_error",
            }


# ──────────────────────────────────────────────
# PER-RECORD CLASSIFICATION
# ──────────────────────────────────────────────

def classify_record(record: dict, chains: tuple, model_key: str) -> dict:
    """Run all 3 subtask chains on one meme record."""
    chain_21, chain_22, chain_23 = chains

    meme_id = record["meme"].replace(".jpeg", "").replace(".jpg", "").replace(".png", "")
    text = record.get("text", "").strip() or "[no text]"
    n = record["number_annotators"]
    labels_21 = record["labels_task2_1"]
    labels_22 = record["labels_task2_2"]
    labels_23 = record["labels_task2_3"]

    # ── Gold labels for evaluation ──
    gold = {
        "soft_21":  compute_soft_gold_21(labels_21),
        "hard_21":  compute_hard_gold_21(labels_21),
        "soft_22":  compute_soft_gold_22(labels_22),
        "soft_23":  compute_soft_gold_23(labels_23),
    }

    # ── Task 2.1 ──────────────────────────────
    try:
        raw_21 = chain_21.invoke({
            "meme_text":               text,
            "num_annotators":          n,
            "annotator_labels_task21": json.dumps(labels_21),
        })
        pred_21 = safe_parse(raw_21, "2.1")
    except Exception as e:
        log.error(f"[{meme_id}][2.1] Error: {e}")
        pred_21 = safe_parse("", "2.1")

    label_21    = pred_21.get("label", "NO")
    soft_21     = float(pred_21.get("soft_score", 0.5))

    # ── Task 2.2 ──────────────────────────────
    try:
        raw_22 = chain_22.invoke({
            "meme_text":               text,
            "num_annotators":          n,
            "annotator_labels_task21": json.dumps(labels_21),
            "annotator_labels_task22": json.dumps(labels_22),
            "label_21":                label_21,
            "soft_21":                 round(soft_21, 3),
        })
        pred_22 = safe_parse(raw_22, "2.2")
    except Exception as e:
        log.error(f"[{meme_id}][2.2] Error: {e}")
        pred_22 = safe_parse("", "2.2")

    # ── Task 2.3 ──────────────────────────────
    try:
        raw_23 = chain_23.invoke({
            "meme_text":               text,
            "num_annotators":          n,
            "annotator_labels_task21": json.dumps(labels_21),
            "annotator_labels_task23": json.dumps(labels_23),
            "label_21":                label_21,
            "soft_21":                 round(soft_21, 3),
        })
        pred_23 = safe_parse(raw_23, "2.3")
    except Exception as e:
        log.error(f"[{meme_id}][2.3] Error: {e}")
        pred_23 = safe_parse("", "2.3")

    return {
        "meme_id":  meme_id,
        "model":    model_key,
        "text":     text,
        "gold":     gold,

        # subtask 2.1
        "pred_2_1_hard":  label_21,
        "pred_2_1_soft":  soft_21,
        "reason_2_1":     pred_21.get("reasoning", ""),

        # subtask 2.2
        "pred_2_2_hard":        pred_22.get("label"),
        "pred_2_2_p_direct":    pred_22.get("p_direct"),
        "pred_2_2_p_judgemental": pred_22.get("p_judgemental"),
        "reason_2_2":           pred_22.get("reasoning", ""),

        # subtask 2.3
        "pred_2_3_hard":   pred_23.get("labels", []),
        "pred_2_3_scores": {
            "IDEOLOGICAL-INEQUALITY":       pred_23.get("p_ideological_inequality", 0.0),
            "STEREOTYPING-DOMINANCE":       pred_23.get("p_stereotyping_dominance", 0.0),
            "OBJECTIFICATION":              pred_23.get("p_objectification", 0.0),
            "SEXUAL-VIOLENCE":              pred_23.get("p_sexual_violence", 0.0),
            "MISOGYNY-NON-SEXUAL-VIOLENCE": pred_23.get("p_misogyny_non_sexual_violence", 0.0),
        },
        "reason_2_3": pred_23.get("reasoning", ""),
    }


# ──────────────────────────────────────────────
# ICM EVALUATION (simplified normalised version)
# ──────────────────────────────────────────────

def icm_soft_norm_21(pred_soft: float, gold_soft: float) -> float:
    """
    Simplified ICM-Soft-Norm for binary task.
    Perfect match = 1.0, worst = -1.0.
    Based on: penalise distance to gold soft score.
    """
    distance = abs(pred_soft - gold_soft)
    return round(1.0 - 2.0 * distance, 4)   # linear approximation

def evaluate_batch(results: list[dict]) -> dict:
    """Compute aggregate soft ICM-Norm and accuracy for subtask 2.1."""
    icm_scores, correct = [], 0
    for r in results:
        icm_scores.append(icm_soft_norm_21(r["pred_2_1_soft"], r["gold"]["soft_21"]))
        if r["pred_2_1_hard"] == r["gold"]["hard_21"]:
            correct += 1
    return {
        "n":             len(results),
        "icm_soft_norm": round(sum(icm_scores) / len(icm_scores), 4),
        "accuracy_21":   round(correct / len(results), 4),
    }


# ──────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────

def run_pipeline(
    data_path: str,
    model_key: str = "qwen",
    max_records: int | None = 50,
    output_path: str | None = None,
):
    """
    Main entry point.

    Args:
        data_path:   Path to EXIST2026 training JSON.
        model_key:   One of 'moondream', 'qwen', 'llava'.
        max_records: How many memes to process (None = all ~2005).
        output_path: Where to save JSON results.
    """
    model_name = MODELS[model_key]
    log.info(f"Model: {model_name} | Max records: {max_records}")

    with open(data_path) as f:
        raw_data = json.load(f)
    records = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
    if max_records:
        records = records[:max_records]

    log.info(f"Loaded {len(records)} records from {data_path}")

    chains = build_chains(model_name)

    results = []
    for i, rec in enumerate(records):
        meme_id = rec["meme"]
        log.info(f"[{i+1}/{len(records)}] Processing {meme_id}")
        result = classify_record(rec, chains, model_key)
        results.append(result)

        # brief pause to avoid overwhelming Ollama
        time.sleep(0.1)

    metrics = evaluate_batch(results)
    log.info(f"\nEvaluation (subtask 2.1):\n{json.dumps(metrics, indent=2)}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)
        log.info(f"Results saved to {output_path}")

    return results, metrics


# ──────────────────────────────────────────────
# DEMO (dry-run without Ollama — shows prompts)
# ──────────────────────────────────────────────

def demo_prompts():
    """Print the rendered prompts for the first training record."""
    with open("/mnt/user-data/uploads/EXIST2026_training_english_filtered.json") as f:
        raw_data = json.load(f)
    records = list(raw_data.values())
    rec = records[5]   # pick a more interesting record

    n = rec["number_annotators"]
    labels_21 = rec["labels_task2_1"]
    labels_22 = rec["labels_task2_2"]
    labels_23 = rec["labels_task2_3"]
    text = rec.get("text", "").strip()

    print("=" * 70)
    print("RECORD:", rec["meme"])
    print("TEXT:", text)
    print("ANNOTATOR LABELS 2.1:", labels_21)
    print("SOFT GOLD 2.1:", compute_soft_gold_21(labels_21))
    print("HARD GOLD 2.1:", compute_hard_gold_21(labels_21))
    print("=" * 70)

    msgs_21 = PROMPT_2_1.format_messages(
        meme_text=text,
        num_annotators=n,
        annotator_labels_task21=json.dumps(labels_21),
    )
    print("\n── SUBTASK 2.1 PROMPT ──")
    for m in msgs_21:
        print(f"[{m.type.upper()}]\n{m.content}\n")

    msgs_22 = PROMPT_2_2.format_messages(
        meme_text=text,
        num_annotators=n,
        annotator_labels_task21=json.dumps(labels_21),
        annotator_labels_task22=json.dumps(labels_22),
        label_21="YES",
        soft_21=0.833,
    )
    print("\n── SUBTASK 2.2 PROMPT ──")
    for m in msgs_22:
        print(f"[{m.type.upper()}]\n{m.content}\n")

    msgs_23 = PROMPT_2_3.format_messages(
        meme_text=text,
        num_annotators=n,
        annotator_labels_task21=json.dumps(labels_21),
        annotator_labels_task23=json.dumps(labels_23),
        label_21="YES",
        soft_21=0.833,
    )
    print("\n── SUBTASK 2.3 PROMPT ──")
    for m in msgs_23:
        print(f"[{m.type.upper()}]\n{m.content}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_prompts()
    else:
        # Quick test on 10 records using qwen
        # Change model_key to 'moondream' or 'llava' as needed
        run_pipeline(
            data_path="/Users/aryan/Desktop/aryan_Exist_2026_dataset/EXIST_2026/training/EXIST2026_training_english_spanish_filtered.json",
            model_key="unsloth_gemma",        # "moondream" | "qwen" | "llava" | "unsloth_qwen"
            max_records=3,
            output_path="./outputs/exist2026_results_unsloth_gemma.json",
        )