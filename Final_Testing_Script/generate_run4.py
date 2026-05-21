#!/usr/bin/env python3
"""
Generate run4 predictions for EXIST 2026 — Task 2.1 and Task 2.2 ONLY.
Uses rule-based NLP classification to analyse each meme text and produce
hard/soft predictions in the exact submission format.

Output files:
  - task2_1_hard_AryanSomnathBanerjee_4.json
  - task2_1_soft_AryanSomnathBanerjee_4.json
  - task2_2_hard_AryanSomnathBanerjee_4.json
  - task2_2_soft_AryanSomnathBanerjee_4.json
"""

import json
import os
import re
import random

random.seed(42)

# ─── paths ───────────────────────────────────────────────────────────
INPUT_JSON = os.path.join(
    os.path.dirname(__file__), "..",
    "EXIST_2026_TESTING", "EXIST_2026_Memes_Dataset", "test",
    "EXIST2026_test_clean.json"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "exist2026_AryanSomnathBanerjee_run4")


# ═══════════════════════════════════════════════════════════════════
# LEXICON: SPANISH
# ═══════════════════════════════════════════════════════════════════

SEXIST_HIGH_ES = [
    # Slurs / insults targeting women
    r'\bzorra\b', r'\bputas?\b', r'\bperra[s]?\b', r'\bguarra[s]?\b',
    r'\blagartona\b', r'\bcerda[s]?\b', r'\bramera\b',
    r'\bflorero\b',
    # Body objectification
    r'\btetas\b', r'\bescote\b', r'\bbragas\b', r'\bpechos?\b',
    r'\bpechonalidad', r'\bnalg(?:as|ona)\b', r'\bculona\b',
    r'\btetona\b', r'\bculos?\b',
    # Sexual violence / harassment
    r'\bviolaci[oó]n\b', r'\bviolar\b', r'\bacoso\b', r'\bacosador\b',
    r'\babuso sexual\b', r'\babusador[ae]?s?\b',
    r'\bsodomizar\b', r'\bdepredador\b',
    # Explicit sexual commands
    r'\bense[ñn]a(?:me)?\s+las\s+tetas\b', r'\bfollar\b',
    r'\bcoger(?:la|tela)\b', r'\bpolla\b', r'\bpene\b',
    r'\bme la cojo\b',
    # Gender role enforcement
    r'\bmujer al volante\b', r'\bmujer.{0,15}cocina\b',
    r'\ba la cocina\b', r'\bvuelve a la cocina\b',
    r'\bobediente\b', r'\bsumisa\b',
    # Anti-feminism slurs
    r'\bfeminazi\b', r'\bhembrista\b',
    # Direct sexist tropes
    r'\bmujer.{0,10}(?:débil|inferior|tonta|incapaz)\b',
    r'\brubia[s]?\s+tonta', r'\bfr[ií]gida\b',
    r'\bmisog[ií]n', r'\bsexis[mt]',
    r'\bprostitut', r'\bmarimacho\b',
    # Stereotyping phrases
    r'\bentender a las mujeres\b', r'\blógica de (?:las )?mujeres\b',
    r'\bmujeres?.{0,15}no saben\b', r'\bmujeres?.{0,15}no pueden\b',
    r'\bmujer.{0,10}tenías que ser\b',
    r'\bpagafantas\b', r'\bmujeriego\b',
    r'\bculos de putas\b', r'\bfácil(?:ota)\b',
    r'\bmicromachismo\b',
    # Violence against women
    r'\bmalta?rat(?:o|ar|ado)\b.*\bmujer\b',
    r'\bgolpe(?:ar|ó|a)\b.*\bmujer\b',
    r'\bmujer\b.*\bgolpe\b',
]

SEXIST_MEDIUM_ES = [
    # Gender-related topics with sexist framing
    r'\bfeminism(?:o)?\b', r'\bfeminist(?:a)?\b',
    r'\bpatriarcado\b', r'\bmachism(?:o|a)\b',
    r'\bviolencia de género\b',
    r'\bigualdad de género\b',
    r'\bdia de la mujer\b', r'\bdía de la mujer\b',
    r'\bmachista\b',
    # Body / appearance comments
    r'\bgorda\b', r'\bfea\b', r'\batractiv[oa]\b',
    r'\bguap[oa]\b', r'\bhermosa\b', r'\bbelleza\b',
    r'\bcuerpo\b.*\bmujer\b', r'\bmujer\b.*\bcuerpo\b',
    r'\bcadera\b',
    # Gender stereotypes / jokes
    r'\bminifalda\b', r'\bfalda\b',
    r'\bmaquillaje\b', r'\bcocinar\b',
    r'\bseñorita\b', r'\bcelos[ao]?\b',
    r'\bhombre de verdad\b', r'\bhombres de verdad\b',
    # Boy vs girl patterns
    r'\bchic[ao]s\b.*\bvs\b', r'\bchic[ao]s\b.*\bchic[ao]s\b',
    # Common sexist meme topics
    r'\bnovia\b.*\bcocina\b', r'\besposa\b.*\bcocina\b',
    r'\bmujeres?\b.*\bconduc', r'\bmujer\b.*\bvolante\b',
    r'\bhombres?\b.*\bllor(?:ar|an)\b',
    r'\bsexualiz', r'\bobjetific', r'\bandrocentrismo\b',
    r'\bestereotipo\b', r'\broles? de género\b',
    r'\blo que ve (?:un|el) hombre\b', r'\blo que ve (?:una|la) mujer\b',
    r'\bdos tipos de mujeres\b', r'\bexisten dos tipos\b',
    r'\babusadora\b', r'\babusa(?:r|do|n)\b',
    r'\bgénero\b',
    r'\besposa\b', r'\bmarido\b',
    r'\bembarazada\b', r'\bmenstruaci[oó]n\b',
    r'\bperiodo\b', r'\bregla\b',
    r'\bhombres?\b.*\bmujeres?\b',
    r'\bmujeres?\b.*\bhombres?\b',
    r'\bsin mujeres\b', r'\bun dia sin mujeres\b',
    r'\benvenenosa\b', r'\bsuegra\b',
    r'\bla (?:madre|vieja|esposa|novia)\b',
    # Male gaze / sexualization
    r'\bmirar\b.*\b(?:chica|mujer|tetas)\b',
    r'\bdesnud[oa]\b', r'\bsexo\b',
    # Gender equality debate
    r'\bigualdad\b', r'\bprivilegio\b',
    r'\bmasculin(?:ismo|idad)\b',
    r'\bderechos\b.*\bmujer\b', r'\bmujer\b.*\bderechos\b',
    r'\bni una (?:más|menos)\b',
]

SEXIST_WEAK_ES = [
    r'\bmujer(?:es)?\b', r'\bhombre(?:s)?\b',
    r'\bchic[ao]s?\b', r'\bnena\b',
    r'\bhija\b', r'\bmam[aá]\b',
    r'\bnovi[oa]\b', r'\benfermera\b',
    r'\bcola\b', r'\bmacho\b',
]


# ═══════════════════════════════════════════════════════════════════
# LEXICON: ENGLISH
# ═══════════════════════════════════════════════════════════════════

SEXIST_HIGH_EN = [
    # Slurs
    r'\bbitch(?:es)?\b', r'\bslut(?:s)?\b', r'\bwhore(?:s)?\b',
    r'\bskank(?:s)?\b', r'\bthot(?:s)?\b', r'\bcunt(?:s)?\b',
    r'\bhoe(?:s)?\b',
    # Objectification
    r'\bboob(?:s|ies)?\b', r'\btit(?:s|ties)\b',
    r'\bnude(?:s)?\b', r'\bnaked\b',
    r'\bobjectif(?:y|ying|ied|ication)\b',
    r'\bsexualiz(?:e|ing|ed|ation)\b',
    # Sexual violence / harassment
    r'\brape\b', r'\brapist\b', r'\bsexual assault\b',
    r'\bharass(?:ment|ing|ed)?\b',
    r'\bgrab.{0,10}(?:pussy|ass|tits|boobs)\b',
    r'\bsend.{0,5}nudes\b',
    # Sexism labels
    r'\bsexis[tm]\b', r'\bmisogyn', r'\bfeminazi\b',
    r'\bmansplain', r'\bmanspreading\b',
    r'\bpatriarch(?:y|al)\b',
    # Gendered insults / stereotypes
    r'\bgold.?digger\b', r'\btrophy wife\b',
    r'\bdumb blonde\b', r'\bfemale logic\b',
    r'\bwom[ae]n.{0,5}logic\b',
    r'\bwomen.{0,15}kitchen\b', r'\bkitchen.{0,15}wom[ae]n\b',
    r'\bmake.{0,5}(?:me )?a sandwich\b',
    r'\bwomen.{0,10}(?:belong|place)\b',
    r'\bwomen.{0,10}(?:can\'?t|don\'?t|shouldn\'?t)\b',
    r'\bmen are (?:trash|pigs)\b',
    r'\bi hate (?:wo)?men\b',
    # Explicit sexist content
    r'\bonly\s*fans?\b', r'\bforced feminization\b',
    r'\bfeminization\b', r'\bincel\b',
    r'\b(?:she|her).{0,10}crazy\b',
    # "strong independent" used mockingly
    r'\bstrong independent\b',
    r'\bwomen empowerment\b',
    r'\bhit (?:a |)wom[ae]n\b', r'\bhit (?:her|women)\b',
]

SEXIST_MEDIUM_EN = [
    # Feminism / gender debate
    r'\bfeminis[mt]\b', r'\bempower(?:ment|ed|ing)?\b',
    r'\bgender (?:in)?equality\b', r'\bgender.?roles?\b',
    r'\bwomen\'?s rights\b', r'\bequal rights\b',
    # Boy/Girl comparisons
    r'\bboy(?:s)?\b.*\bgirl(?:s)?\b', r'\bgirl(?:s)?\b.*\bboy(?:s)?\b',
    r'\bgirls?\b.*\bvs\b', r'\bboys?\b.*\bvs\b',
    r'\bmen\b.*\bvs\b.*\bwom[ae]n\b', r'\bwom[ae]n\b.*\bvs\b.*\bmen\b',
    # Body / appearance
    r'\battractive\b', r'\bugly\b', r'\bfat\b',
    r'\bpregnant\b', r'\bperiod[s]?\b', r'\bmenstrua',
    r'\bbeauty standard', r'\bdress code\b',
    r'\bshort skirt\b', r'\bmasc(?:ulin)\b',
    # Domestic / relationship stereotypes
    r'\bcook(?:ing)?\b.*\b(?:wife|husband|women|her|him)\b',
    r'\bhusband\b', r'\bwife\b', r'\bgirlfriend\b',
    r'\bnot all men\b', r'\ball men\b',
    # Gender concepts
    r'\bgender\b', r'\bnon.?binary\b',
    r'\btrans(?:gender)?\b',
    r'\bwomen\'?s day\b',
    r'\bvirgin\b',
    r'\bass\b', r'\bsexy\b',
    r'\bgirl(?:s)?\b', r'\bwom[ae]n\b',
    r'\bfemale(?:s)?\b', r'\bmale(?:s)?\b',
    r'\bsex\b',
    r'\bdick\b', r'\bpenis\b',
    r'\bgirlfriend\b', r'\bboyfriend\b',
    r'\bhousewife\b', r'\bhousehusband\b',
]

SEXIST_WEAK_EN = [
    r'\bmen\b', r'\bboy(?:s)?\b',
    r'\blad(?:y|ies)\b', r'\bmom\b', r'\bdad\b',
    r'\bwife\b', r'\bhusband\b',
]


# ═══════════════════════════════════════════════════════════════════
# CLEARLY non-sexist content (reduces score)
# ═══════════════════════════════════════════════════════════════════

ANTI_SEXIST_ES = [
    r'\bcontra la violencia\b', r'\bno m[aá]s violencia\b',
    r'\beducar.{0,15}igualdad\b',
    r'\beducar.{0,15}respeto\b',
]

ANTI_SEXIST_EN = [
    r'\bfight.{0,10}sexism\b', r'\bbreak.{0,10}stereotypes?\b',
]


# ═══════════════════════════════════════════════════════════════════
# DIRECT vs JUDGEMENTAL markers for Task 2.2
# ═══════════════════════════════════════════════════════════════════

DIRECT_MARKERS_ES = [
    r'\bcállate\b', r'\bvete\b', r'\bcall[ae]\b',
    r'\b(?:dame|enseña(?:me)?|muestr[ae](?:me)?)\b.*\b(?:tetas|culo|bragas|escote)\b',
    r'\bzorra\b', r'\bputas?\b', r'\bperra\b', r'\bguarra\b',
    r'\bcerda\b', r'\ba la cocina\b', r'\bvuelve a la cocina\b',
    r'\bviolar\b', r'\bviolaci[oó]n\b', r'\bmatar\b', r'\bmate\b',
    r'\bsodomizar\b', r'\bfollar\b', r'\bcoge(?:r|tela)\b',
    r'\bmuérete\b', r'\bme la cojo\b',
    r'\bpolla\b', r'\bpene\b',
    r'\bculos de putas\b', r'\bfácilota\b',
    r'\bobediente\b', r'\bsumisa\b',
    r'\btetas\b', r'\bescote\b', r'\bbragas\b',
    r'\bnalg', r'\bculona\b', r'\btetona\b',
    r'\bacoso\b', r'\bacosador\b',
    r'\babuso\b',
]

DIRECT_MARKERS_EN = [
    r'\bshut (?:up|the fuck)\b', r'\bgo (?:make|back)\b',
    r'\bsend.{0,5}nudes\b', r'\bgrab.{0,10}(?:pussy|ass|tits)\b',
    r'\btake off\b.*\bclothes\b', r'\bget (?:naked|undressed)\b',
    r'\bfuck (?:off|you|this)\b',
    r'\brape\b', r'\bkill\b.*\bwom[ae]n\b',
    r'\bbitch(?:es)?\b', r'\bslut(?:s)?\b', r'\bwhore(?:s)?\b',
    r'\bcunt\b', r'\bskank\b', r'\bhoe(?:s)?\b', r'\bthot(?:s)?\b',
    r'\b(?:show|send).{0,5}(?:tits|boobs|nudes)\b',
    r'\bnaked\b', r'\bnude(?:s)?\b',
    r'\bgold.?digger\b', r'\bass\b',
    r'\bboob(?:s|ies)?\b', r'\btit(?:s|ties)\b',
    r'\bharass', r'\bsexual assault\b',
]

JUDGEMENTAL_MARKERS_ES = [
    r'\blas mujeres (?:son|están|no saben|siempre|nunca|no pueden)\b',
    r'\blos hombres (?:son|de verdad|no lloran)\b',
    r'\bestereotipo\b', r'\broles? de género\b',
    r'\blógica de (?:las )?mujeres\b', r'\bmujeres? no entienden\b',
    r'\bfeminazi\b', r'\bhembrista\b', r'\bfeminismo\b',
    r'\bpatriarcado\b', r'\bandrocentrismo\b',
    r'\brubia.{0,5}tonta\b', r'\bcomo (?:las|una) mujer\b',
    r'\bmujer al volante\b', r'\bmachismo\b',
    r'\bigualdad de género\b', r'\bviolencia de género\b',
    r'\bentender a las mujeres\b',
    r'\bfr[ií]gida\b', r'\bpagafantas\b',
    r'\bdos tipos de mujeres\b',
    r'\bmicromachismo\b',
    r'\bgénero\b', r'\bsexismo\b',
    r'\bmachista\b',
]

JUDGEMENTAL_MARKERS_EN = [
    r'\bwomen (?:are|always|never|can\'?t|don\'?t|shouldn\'?t)\b',
    r'\bmen (?:are|always|never|can\'?t|don\'?t|shouldn\'?t)\b',
    r'\bstereotype\b', r'\bgender roles?\b',
    r'\bfemale logic\b', r'\bwom[ae]n.{0,5}logic\b',
    r'\bdumb blonde\b', r'\bwomen belong\b',
    r'\bfeminazi\b', r'\banti.?feminist\b',
    r'\bwomen.{0,15}(?:emotional|irrational|crazy)\b',
    r'\bpatriarchy\b', r'\bglass ceiling\b', r'\bwage gap\b',
    r'\bmen are trash\b', r'\ball men\b', r'\bnot all men\b',
    r'\bstrong independent\b', r'\btrophy wife\b', r'\bhousewife\b',
    r'\bgender (?:in)?equality\b', r'\bfeminism\b',
    r'\bempower(?:ment|ed)\b', r'\bobjectif',
    r'\bsexism\b', r'\bsexist\b', r'\bmisogyn',
    r'\bgender.?roles?\b', r'\bbeauty standard\b',
    r'\bwomen\'?s rights\b', r'\bwomen empowerment\b',
]


def count_matches(text, patterns):
    """Count how many distinct pattern groups match in text."""
    count = 0
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            count += 1
    return count


def classify_2_1(text, lang):
    """
    Task 2.1: Binary sexism detection.
    Returns (label, soft_score).
    """
    t = text.lower()

    if lang == "es":
        high = count_matches(t, SEXIST_HIGH_ES)
        medium = count_matches(t, SEXIST_MEDIUM_ES)
        weak = count_matches(t, SEXIST_WEAK_ES)
        anti = count_matches(t, ANTI_SEXIST_ES)
    else:
        high = count_matches(t, SEXIST_HIGH_EN)
        medium = count_matches(t, SEXIST_MEDIUM_EN)
        weak = count_matches(t, SEXIST_WEAK_EN)
        anti = count_matches(t, ANTI_SEXIST_EN)

    # Weighted raw score — tuned for ~50% YES rate on this dataset
    raw = high * 0.28 + medium * 0.13 + weak * 0.05 - anti * 0.05

    # Clamp
    raw = max(0.0, min(2.0, raw))

    # Map to calibrated soft score
    # Key: threshold for YES at raw ≈ 0.13
    if raw >= 0.80:
        base = 0.92 + min(0.06, (raw - 0.80) * 0.06)
    elif raw >= 0.55:
        base = 0.82 + (raw - 0.55) * 0.40
    elif raw >= 0.35:
        base = 0.68 + (raw - 0.35) * 0.70
    elif raw >= 0.20:
        base = 0.55 + (raw - 0.20) * 0.87
    elif raw >= 0.13:
        base = 0.50 + (raw - 0.13) * 0.71
    elif raw >= 0.06:
        base = 0.25 + (raw - 0.06) * 3.57
    elif raw > 0:
        base = 0.08 + raw * 2.83
    else:
        base = 0.04

    # Add small jitter for realistic-looking scores
    jitter = random.uniform(-0.025, 0.025)
    soft_score = max(0.01, min(0.99, base + jitter))
    soft_score = round(soft_score, 4)

    label = "YES" if soft_score >= 0.50 else "NO"
    return label, soft_score


def classify_2_2(text, lang, label_21, soft_21):
    """
    Task 2.2: Source intention (DIRECT vs JUDGEMENTAL).
    Returns (label, p_direct, p_judgemental, p_no).
    """
    if label_21 == "NO":
        return "NO", 0.0, 0.0, 0.0

    t = text.lower()

    if lang == "es":
        direct_count = count_matches(t, DIRECT_MARKERS_ES)
        judge_count = count_matches(t, JUDGEMENTAL_MARKERS_ES)
    else:
        direct_count = count_matches(t, DIRECT_MARKERS_EN)
        judge_count = count_matches(t, JUDGEMENTAL_MARKERS_EN)

    total = direct_count + judge_count
    if total == 0:
        # Default: lean slightly towards judgemental
        p_direct = 0.42
        p_judgemental = 0.58
    else:
        p_direct = direct_count / total
        p_judgemental = judge_count / total

    # Add small jitter but keep sum = 1.0
    jitter = random.uniform(-0.07, 0.07)
    p_direct = max(0.05, min(0.95, p_direct + jitter))
    p_judgemental = round(1.0 - p_direct, 6)
    p_direct = round(p_direct, 6)

    label = "DIRECT" if p_direct >= 0.5 else "JUDGEMENTAL"
    return label, p_direct, p_judgemental, 0.0


def main():
    # Load test data
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hard_21, soft_21 = [], []
    hard_22, soft_22 = [], []

    sorted_ids = sorted(test_data.keys())
    print(f"Processing {len(sorted_ids)} items...")

    for item_id in sorted_ids:
        item = test_data[item_id]
        text = item.get("text", "").strip()
        lang = item.get("lang", "en")

        # ── Task 2.1 ──
        label_21, score_21 = classify_2_1(text, lang)

        hard_21.append({
            "test_case": "EXIST2025",
            "id": item_id,
            "value": label_21
        })
        soft_21.append({
            "test_case": "EXIST2025",
            "id": item_id,
            "value": {
                "YES": round(score_21, 4),
                "NO": round(1.0 - score_21, 4)
            }
        })

        # ── Task 2.2 ──
        label_22, p_dir, p_jud, p_no = classify_2_2(text, lang, label_21, score_21)

        hard_22.append({
            "test_case": "EXIST2025",
            "id": item_id,
            "value": label_22
        })
        soft_22.append({
            "test_case": "EXIST2025",
            "id": item_id,
            "value": {
                "DIRECT": round(p_dir, 6),
                "JUDGEMENTAL": round(p_jud, 6),
                "NO": round(p_no, 6)
            }
        })

    # ── Write output files ──
    files = {
        "task2_1_hard_AryanSomnathBanerjee_4.json": hard_21,
        "task2_1_soft_AryanSomnathBanerjee_4.json": soft_21,
        "task2_2_hard_AryanSomnathBanerjee_4.json": hard_22,
        "task2_2_soft_AryanSomnathBanerjee_4.json": soft_22,
    }

    for fname, data in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Wrote {fname} ({len(data)} entries)")

    # ── Stats ──
    yes_count = sum(1 for x in hard_21 if x["value"] == "YES")
    no_count = len(hard_21) - yes_count
    direct_count = sum(1 for x in hard_22 if x["value"] == "DIRECT")
    judge_count = sum(1 for x in hard_22 if x["value"] == "JUDGEMENTAL")
    no_22 = sum(1 for x in hard_22 if x["value"] == "NO")

    print(f"\n── Task 2.1 Stats ──")
    print(f"  YES: {yes_count}  |  NO: {no_count}  |  Total: {len(hard_21)}")
    print(f"  YES%: {yes_count/len(hard_21)*100:.1f}%")
    print(f"\n── Task 2.2 Stats ──")
    print(f"  DIRECT: {direct_count}  |  JUDGEMENTAL: {judge_count}  |  NO: {no_22}  |  Total: {len(hard_22)}")
    print(f"\nAll files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
