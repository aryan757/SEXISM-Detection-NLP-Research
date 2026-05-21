"""
EXIST 2026 — Sensor-Aware Soft Score Fusion
============================================
Takes your EXISTING soft score JSON files (run2) +
sensor data from test JSON → produces FINAL updated soft + hard JSONs.

Two modes:
  Mode A (LLM): Uses local Ollama (llama3.2 / mistral / any model)
                to reason about sensor signals and adjust confidence
  Mode B (Rule): Pure Python rule-based fusion (no LLM, instant)

Run Mode B first to verify it works, then switch to Mode A for better results.

Usage:
  pip install requests numpy
  python sensor_fusion_final.py --mode rule    # fast, no LLM needed
  python sensor_fusion_final.py --mode llm     # uses Ollama
  python sensor_fusion_final.py --mode llm --model llama3.2
"""

import json, os, sys, argparse, requests, time, math
import numpy as np
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths to match your folder structure
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Path to your test JSON (contains meme IDs + sensor data)
    "test_json":  "/Users/aryan/Desktop/aryan_Exist_2026_dataset/EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/test/EXIST2026_test_clean.json",

    # Folder containing your existing 6 soft/hard JSON files (run2)
    "input_dir":  "exist2026_AryanSomnathBanerjee_run2",

    # Where to write the NEW fused output files
    "output_dir": "exist2026_AryanSomnathBanerjee_run3",

    # Your team name (used in filenames)
    "team_name":  "AryanSomnathBanerjee",
    "run_id":     3,

    # Ollama settings
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": "llama3.1",   # change to: mistral, qwen2.5, llama3.1 etc.

    # Fusion weight: how much sensor adjusts the VLM score
    # 0.0 = ignore sensor completely, 1.0 = sensor dominates
    "fusion_alpha": 0.35,
}

CATS = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
]

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: LOAD EXISTING SOFT SCORES
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_scores(input_dir, team_name, run_id):
    """
    Load your 6 existing JSON files and return one dict per meme:
    {
      "310001": {
        "t21_yes": 0.058, "t21_no": 0.941,
        "t22_direct": 0.0, "t22_judg": 0.0, "t22_no": 1.0,
        "t23_IDEOLOGICAL-INEQUALITY": 0.0, ...,
        "t21_hard": "NO",
        "t22_hard": "NO",
        "t23_hard": ["NO"]
      }
    }
    """
    scores = defaultdict(dict)
    files = {
        "t21_soft": f"task2_1_soft_{team_name}_{run_id}.json",
        "t21_hard": f"task2_1_hard_{team_name}_{run_id}.json",
        "t22_soft": f"task2_2_soft_{team_name}_{run_id}.json",
        "t22_hard": f"task2_2_hard_{team_name}_{run_id}.json",
        "t23_soft": f"task2_3_soft_{team_name}_{run_id}.json",
        "t23_hard": f"task2_3_hard_{team_name}_{run_id}.json",
    }
    for key, fname in files.items():
        fpath = os.path.join(input_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] Not found: {fpath}")
            continue
        with open(fpath) as f:
            records = json.load(f)
        for rec in records:
            mid = str(rec["id"])
            v   = rec["value"]
            if key == "t21_soft":
                scores[mid]["t21_yes"] = float(v.get("YES", 0.5))
                scores[mid]["t21_no"]  = float(v.get("NO",  0.5))
            elif key == "t21_hard":
                scores[mid]["t21_hard"] = str(v)
            elif key == "t22_soft":
                scores[mid]["t22_direct"] = float(v.get("DIRECT",0))
                scores[mid]["t22_judg"]   = float(v.get("JUDGEMENTAL",0))
                scores[mid]["t22_no"]     = float(v.get("NO",0))
            elif key == "t22_hard":
                scores[mid]["t22_hard"] = str(v)
            elif key == "t23_soft":
                for cat in CATS:
                    scores[mid][f"t23_{cat}"] = float(v.get(cat, 0.0))
                scores[mid]["t23_no"] = float(v.get("NO", 0.0))
            elif key == "t23_hard":
                scores[mid]["t23_hard"] = v if isinstance(v, list) else [v]

    print(f"  Loaded scores for {len(scores)} memes from {input_dir}")
    return dict(scores)

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: EXTRACT SENSOR FEATURES FROM TEST JSON
# ══════════════════════════════════════════════════════════════════════════════

def safe_mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(v)) if v else None

def extract_sensor_summary(record):
    """
    Returns a flat dict of aggregated sensor features for one meme.
    Aggregates across all subjects (mean across users).
    Returns None if no sensor data present.
    """
    s = record.get("sensorial", {})
    if not s or not s.get("users"):
        return None

    feats = {}

    # ── Eye Tracking ─────────────────────────────────────────────────────────
    et_users = s.get("modalities",{}).get("ET",{}).get("by_user",{})
    if et_users:
        rt_vals   = [u.get("reaction_time") for u in et_users.values()]
        pup_l     = [u.get("3d_eye_states_pupil diameter left [mm]_mean") for u in et_users.values()]
        pup_r     = [u.get("3d_eye_states_pupil diameter right [mm]_mean") for u in et_users.values()]
        fix_c     = [u.get("fixations_count") for u in et_users.values()]
        fix_dur   = [u.get("fixations_duration_mean_ns") for u in et_users.values()]
        sac_c     = [u.get("saccades_count") for u in et_users.values()]
        blink_c   = [u.get("blinks_count") for u in et_users.values()]

        feats["et_reaction_time_ms"]      = safe_mean(rt_vals)
        feats["et_pupil_left_mean_mm"]    = safe_mean(pup_l)
        feats["et_pupil_right_mean_mm"]   = safe_mean(pup_r)
        feats["et_fixations_count"]       = safe_mean(fix_c)
        feats["et_fixation_dur_mean_ns"]  = safe_mean(fix_dur)
        feats["et_saccades_count"]        = safe_mean(sac_c)
        feats["et_blinks_count"]          = safe_mean(blink_c)

    # ── Heart Rate ────────────────────────────────────────────────────────────
    hr_users = s.get("modalities",{}).get("HR",{}).get("by_user",{})
    if hr_users:
        feats["hr_mean"]     = safe_mean([u.get("garmin_hr_mean") for u in hr_users.values()])
        feats["hr_std"]      = safe_mean([u.get("garmin_hr_std")  for u in hr_users.values()])
        feats["hr_max"]      = safe_mean([u.get("garmin_hr_max")  for u in hr_users.values()])
        feats["hr_min"]      = safe_mean([u.get("garmin_hr_min")  for u in hr_users.values()])

    # ── EEG — summarise per band across all channels ──────────────────────────
    eeg_users = s.get("modalities",{}).get("EEG",{}).get("by_user",{})
    if eeg_users:
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            all_vals = []
            for uid, uv in eeg_users.items():
                for ch in range(16):
                    key = f"EXG_Channel_{ch}_{band}_power"
                    val = uv.get(key)
                    if val is not None:
                        all_vals.append(val)
            feats[f"eeg_{band.lower()}_mean"] = safe_mean(all_vals)

    # ── Derived arousal score (0–1) ───────────────────────────────────────────
    arousal_signals = []

    rt = feats.get("et_reaction_time_ms")
    if rt is not None:
        # Normalise: 5000ms=low, 20000ms=high
        arousal_signals.append(min(max((rt - 5000) / 15000, 0), 1))

    hr_std = feats.get("hr_std")
    if hr_std is not None:
        # Normalise: 0=low, 3=high
        arousal_signals.append(min(max(hr_std / 3.0, 0), 1))

    pup = feats.get("et_pupil_left_mean_mm")
    if pup is not None:
        # Normalise: 2mm=low, 5mm=high
        arousal_signals.append(min(max((pup - 2.0) / 3.0, 0), 1))

    theta = feats.get("eeg_theta_mean")
    if theta is not None:
        # Normalise: -2=low, 2=high
        arousal_signals.append(min(max((theta + 2) / 4.0, 0), 1))

    fix = feats.get("et_fixations_count")
    if fix is not None:
        # Normalise: 10=low, 60=high
        arousal_signals.append(min(max((fix - 10) / 50.0, 0), 1))

    feats["derived_arousal_score"] = float(np.mean(arousal_signals)) if arousal_signals else 0.5

    return feats

def load_sensor_data(test_json_path):
    """Load all sensor features from test JSON."""
    with open(test_json_path) as f:
        data = json.load(f)
    
    sensor_map = {}
    no_sensor  = 0
    for mid, record in data.items():
        feats = extract_sensor_summary(record)
        if feats:
            sensor_map[str(mid)] = feats
        else:
            no_sensor += 1

    print(f"  Sensor data: {len(sensor_map)} memes have it, {no_sensor} do not")
    return sensor_map

# ══════════════════════════════════════════════════════════════════════════════
# PART 3A: RULE-BASED FUSION (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def rule_based_adjustment(vlm_yes_prob, sensor_feats, alpha=0.35):
    """
    Adjust VLM P(YES) using sensor arousal score.

    Logic:
      arousal_score ∈ [0, 1]
      arousal → 1 means body was disturbed → push P(YES) UP
      arousal → 0 means body was calm      → push P(YES) DOWN

    Formula:
      adjusted = vlm_prob + alpha * (arousal_score - 0.5) * 2 * vlm_prob * (1-vlm_prob)
      The term vlm_prob*(1-vlm_prob) ensures we only adjust uncertain predictions,
      not ones already near 0 or 1.
    """
    if sensor_feats is None:
        return vlm_yes_prob

    arousal = sensor_feats.get("derived_arousal_score", 0.5)

    # Adjustment: positive if arousal > 0.5, negative if arousal < 0.5
    delta = alpha * (arousal - 0.5) * 2.0 * vlm_yes_prob * (1.0 - vlm_yes_prob)
    adjusted = vlm_yes_prob + delta

    return float(np.clip(adjusted, 0.001, 0.999))

def rule_fuse_all(scores, sensor_map, alpha=0.35):
    """Apply rule-based fusion to all memes for all 3 subtasks."""
    fused = {}
    for mid, sc in scores.items():
        sen = sensor_map.get(str(mid))
        f   = dict(sc)  # copy

        # ── Task 2.1 ─────────────────────────────────────────────────────────
        old_yes = sc.get("t21_yes", 0.5)
        new_yes = rule_based_adjustment(old_yes, sen, alpha)
        new_no  = 1.0 - new_yes
        f["t21_yes_fused"] = new_yes
        f["t21_no_fused"]  = new_no

        # ── Task 2.2 ─────────────────────────────────────────────────────────
        # Only adjust if meme is sexist; use arousal to shift between DIRECT/JUDGEMENTAL
        if new_yes >= 0.5:
            old_d = sc.get("t22_direct", 0.5)
            old_j = sc.get("t22_judg",   0.5)
            if sen:
                arousal = sen.get("derived_arousal_score", 0.5)
                # High arousal → more likely DIRECT (aggressive intent)
                direct_adj = rule_based_adjustment(old_d, sen, alpha * 0.5)
                judg_adj   = 1.0 - direct_adj
            else:
                direct_adj, judg_adj = old_d, old_j
            # Normalise
            tot = direct_adj + judg_adj
            f["t22_direct_fused"] = direct_adj / tot if tot > 0 else 0.5
            f["t22_judg_fused"]   = judg_adj   / tot if tot > 0 else 0.5
            f["t22_no_fused"]     = 0.0
        else:
            f["t22_direct_fused"] = 0.0
            f["t22_judg_fused"]   = 0.0
            f["t22_no_fused"]     = 1.0

        # ── Task 2.3 ─────────────────────────────────────────────────────────
        for cat in CATS:
            old_cat = sc.get(f"t23_{cat}", 0.0)
            new_cat = rule_based_adjustment(old_cat, sen, alpha * 0.6)
            f[f"t23_{cat}_fused"] = new_cat

        fused[mid] = f

    return fused

# ══════════════════════════════════════════════════════════════════════════════
# PART 3B: LLM-BASED FUSION (Ollama)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert in physiological signals and computational sexism detection.
You are given:
1. Soft classification scores from a Vision-Language Model (VLM) for a meme
2. Physiological sensor readings from real humans who viewed the same meme

Your task: Adjust the VLM confidence scores based on what the sensor data tells us
about how the human body actually reacted to this meme.

SENSOR INTERPRETATION RULES:
- HIGH arousal_score (>0.65): Body was disturbed → confidence this is sexist should increase
- LOW arousal_score (<0.35): Body was calm → confidence should decrease  
- MEDIUM (0.35-0.65): No strong signal → keep VLM score mostly unchanged

WHAT EACH SENSOR MEASURES:
- et_reaction_time_ms: Time to decide (ms). >12000=hesitation=disturbed, <5000=quick/unbothered
- hr_std: Heart rate variability. >1.5=emotional turbulence, <0.8=calm
- et_pupil_left_mean_mm: Pupil dilation. >3.5mm=high load, <2.5mm=low load
- eeg_theta_mean: Brain emotional processing. >0.5=triggered, near 0=neutral
- et_fixations_count: Gaze fixations. >45=intensive study, <15=quick glance

ADJUSTMENT RULES:
- Only adjust by maximum ±0.20 from the original VLM score
- For Task 2.1 (sexist/not): if arousal_score>0.65 AND VLM_YES<0.7, increase YES by 0.05-0.15
- For Task 2.2 (intent): high arousal → shift slightly toward DIRECT (aggressive)
- For Task 2.3 (category): adjust each category score by its relevance to the sensor pattern

OUTPUT FORMAT (strict JSON, nothing else):
{
  "task21_yes": <float 0-1>,
  "task21_no": <float 0-1>,
  "task22_direct": <float 0-1>,
  "task22_judgemental": <float 0-1>,
  "task22_no": <float 0-1>,
  "task23_IDEOLOGICAL-INEQUALITY": <float 0-1>,
  "task23_STEREOTYPING-DOMINANCE": <float 0-1>,
  "task23_OBJECTIFICATION": <float 0-1>,
  "task23_SEXUAL-VIOLENCE": <float 0-1>,
  "task23_MISOGYNY-NON-SEXUAL-VIOLENCE": <float 0-1>,
  "reasoning": "<one sentence why you adjusted>"
}

CONSTRAINTS:
- task21_yes + task21_no MUST sum to exactly 1.0
- task22_direct + task22_judgemental + task22_no MUST sum to exactly 1.0
- task23 scores are INDEPENDENT (do NOT need to sum to 1)
- All values MUST be between 0.001 and 0.999
- If no sensor data is available, return the original VLM scores unchanged
"""

def build_user_prompt(mid, sc, sen):
    """Build the per-meme prompt for Ollama."""
    vlm_section = f"""
MEME ID: {mid}

VLM SOFT SCORES (from your existing predictions):
  Task 2.1 — Sexism identification:
    P(YES = sexist)  = {sc.get('t21_yes', 0.5):.6f}
    P(NO  = not sexist) = {sc.get('t21_no', 0.5):.6f}

  Task 2.2 — Source intention:
    P(DIRECT)      = {sc.get('t22_direct', 0.0):.6f}
    P(JUDGEMENTAL) = {sc.get('t22_judg',   0.0):.6f}
    P(NO)          = {sc.get('t22_no',     1.0):.6f}

  Task 2.3 — Category (independent probabilities):
    IDEOLOGICAL-INEQUALITY      = {sc.get('t23_IDEOLOGICAL-INEQUALITY', 0.0):.6f}
    STEREOTYPING-DOMINANCE      = {sc.get('t23_STEREOTYPING-DOMINANCE', 0.0):.6f}
    OBJECTIFICATION             = {sc.get('t23_OBJECTIFICATION', 0.0):.6f}
    SEXUAL-VIOLENCE             = {sc.get('t23_SEXUAL-VIOLENCE', 0.0):.6f}
    MISOGYNY-NON-SEXUAL-VIOLENCE= {sc.get('t23_MISOGYNY-NON-SEXUAL-VIOLENCE', 0.0):.6f}
"""

    if sen is None:
        sensor_section = "\nSENSOR DATA: Not available for this meme. Return original VLM scores.\n"
    else:
        sensor_section = f"""
PHYSIOLOGICAL SENSOR DATA (averaged across {2} subjects who viewed this meme):

  EYE TRACKING:
    Reaction time          = {sen.get('et_reaction_time_ms', 'N/A')} ms
    Pupil left mean        = {sen.get('et_pupil_left_mean_mm', 'N/A')} mm
    Pupil right mean       = {sen.get('et_pupil_right_mean_mm', 'N/A')} mm
    Fixation count         = {sen.get('et_fixations_count', 'N/A')}
    Fixation duration mean = {sen.get('et_fixation_dur_mean_ns', 'N/A')} ns
    Saccade count          = {sen.get('et_saccades_count', 'N/A')}
    Blink count            = {sen.get('et_blinks_count', 'N/A')}

  HEART RATE:
    Mean HR                = {sen.get('hr_mean', 'N/A')} bpm
    HR std deviation       = {sen.get('hr_std', 'N/A')} ← KEY: >1.5 = emotional turbulence
    Max HR                 = {sen.get('hr_max', 'N/A')} bpm
    Min HR                 = {sen.get('hr_min', 'N/A')} bpm

  EEG BRAIN WAVES (mean across 16 channels):
    Delta power (deep processing)    = {sen.get('eeg_delta_mean', 'N/A')}
    Theta power (emotional memory)   = {sen.get('eeg_theta_mean', 'N/A')} ← KEY: >0.5 = triggered
    Alpha power (relaxation)         = {sen.get('eeg_alpha_mean', 'N/A')}
    Beta  power (active thinking)    = {sen.get('eeg_beta_mean', 'N/A')}
    Gamma power (high cognition)     = {sen.get('eeg_gamma_mean', 'N/A')}

  DERIVED AROUSAL SCORE = {sen.get('derived_arousal_score', 0.5):.4f}
  (0=very calm, 0.5=neutral, 1=highly disturbed)
  Interpretation: {'HIGH AROUSAL — body was disturbed by this meme' if sen.get('derived_arousal_score',0.5) > 0.65 else ('LOW AROUSAL — body was calm' if sen.get('derived_arousal_score',0.5) < 0.35 else 'NEUTRAL AROUSAL — no clear signal')}
"""

    return vlm_section + sensor_section + "\nNow output the adjusted JSON:"

def call_ollama(prompt, model, url, timeout=60):
    """Call Ollama API and return parsed JSON response."""
    payload = {
        "model":  model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp = consistent, deterministic output
            "top_p": 0.9,
            "num_predict": 400,
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json().get("response","").strip()
        # Extract JSON block
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        return None
    except Exception as e:
        print(f"    [Ollama error] {e}")
        return None

def llm_fuse_all(scores, sensor_map, config):
    """Use Ollama LLM to fuse VLM scores with sensor data."""
    fused   = {}
    total   = len(scores)
    ok_cnt  = 0
    fallback= 0

    print(f"\n  Running LLM fusion on {total} memes with {config['ollama_model']}...")
    print(f"  (This will take ~{total * 2 // 60} minutes. Ctrl+C to stop and use rule fallback.)\n")

    for i, (mid, sc) in enumerate(scores.items()):
        sen = sensor_map.get(str(mid))

        # Build prompt
        user_prompt = build_user_prompt(mid, sc, sen)

        # Call Ollama
        result = call_ollama(user_prompt, config["ollama_model"],
                             config["ollama_url"], timeout=90)

        if result and "task21_yes" in result:
            ok_cnt += 1
            # Validate and clip
            t21_yes = float(np.clip(result.get("task21_yes", sc.get("t21_yes",0.5)), 0.001, 0.999))
            t21_no  = 1.0 - t21_yes

            t22_d  = float(np.clip(result.get("task22_direct",      sc.get("t22_direct",0)), 0, 1))
            t22_j  = float(np.clip(result.get("task22_judgemental",  sc.get("t22_judg",0)),  0, 1))
            t22_no = float(np.clip(result.get("task22_no",           sc.get("t22_no",1)),    0, 1))
            t22_sum = t22_d + t22_j + t22_no
            if t22_sum > 0:
                t22_d /= t22_sum; t22_j /= t22_sum; t22_no /= t22_sum

            f = dict(sc)
            f["t21_yes_fused"] = t21_yes
            f["t21_no_fused"]  = t21_no
            f["t22_direct_fused"] = t22_d
            f["t22_judg_fused"]   = t22_j
            f["t22_no_fused"]     = t22_no
            f["llm_reasoning"]    = result.get("reasoning", "")

            for cat in CATS:
                key_out = f"task23_{cat}"
                f[f"t23_{cat}_fused"] = float(np.clip(
                    result.get(key_out, sc.get(f"t23_{cat}", 0.0)), 0.001, 0.999))
        else:
            # Fallback to rule-based for this meme
            fallback += 1
            rule_result = rule_fuse_all({mid: sc}, {mid: sen} if sen else {}, config["fusion_alpha"])
            f = rule_result[mid]

        fused[mid] = f

        # Progress
        if (i+1) % 10 == 0 or (i+1) == total:
            print(f"  [{i+1}/{total}] LLM ok={ok_cnt}  fallback={fallback}")

    print(f"\n  LLM fusion complete: {ok_cnt} LLM, {fallback} rule-fallback")
    return fused

# ══════════════════════════════════════════════════════════════════════════════
# PART 4: WRITE OUTPUT JSON FILES (PyEvALL format)
# ══════════════════════════════════════════════════════════════════════════════

def write_output_files(fused, output_dir, team_name, run_id):
    """Write all 6 PyEvALL-format JSON files from fused predictions."""
    os.makedirs(output_dir, exist_ok=True)
    TCASE = "EXIST2025"   # must stay EXIST2025 per guidelines
    ids   = list(fused.keys())

    # Helper to write one file
    def write(records, fname):
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  ✓ {fname}  ({len(records)} memes)")

    # ── Task 2.1 SOFT ─────────────────────────────────────────────────────────
    recs = [{"test_case": TCASE, "id": mid,
             "value": {
                 "YES": round(fused[mid]["t21_yes_fused"], 6),
                 "NO":  round(fused[mid]["t21_no_fused"],  6)
             }} for mid in ids]
    write(recs, f"task2_1_soft_{team_name}_{run_id}.json")

    # ── Task 2.1 HARD ─────────────────────────────────────────────────────────
    recs = [{"test_case": TCASE, "id": mid,
             "value": "YES" if fused[mid]["t21_yes_fused"] >= 0.5 else "NO"}
            for mid in ids]
    write(recs, f"task2_1_hard_{team_name}_{run_id}.json")

    # ── Task 2.2 SOFT ─────────────────────────────────────────────────────────
    recs = [{"test_case": TCASE, "id": mid,
             "value": {
                 "DIRECT":      round(fused[mid].get("t22_direct_fused", 0.0), 6),
                 "JUDGEMENTAL": round(fused[mid].get("t22_judg_fused",   0.0), 6),
                 "NO":          round(fused[mid].get("t22_no_fused",     1.0), 6),
             }} for mid in ids]
    write(recs, f"task2_2_soft_{team_name}_{run_id}.json")

    # ── Task 2.2 HARD ─────────────────────────────────────────────────────────
    recs = []
    for mid in ids:
        d  = fused[mid].get("t22_direct_fused", 0.0)
        j  = fused[mid].get("t22_judg_fused",   0.0)
        no = fused[mid].get("t22_no_fused",     1.0)
        if no >= max(d, j):
            val = "NO"
        else:
            val = "DIRECT" if d >= j else "JUDGEMENTAL"
        recs.append({"test_case": TCASE, "id": mid, "value": val})
    write(recs, f"task2_2_hard_{team_name}_{run_id}.json")

    # ── Task 2.3 SOFT ─────────────────────────────────────────────────────────
    recs = []
    for mid in ids:
        val = {cat: round(fused[mid].get(f"t23_{cat}_fused", 0.0), 6) for cat in CATS}
        val["NO"] = round(max(0.0, 1.0 - max(val.values())), 6)
        recs.append({"test_case": TCASE, "id": mid, "value": val})
    write(recs, f"task2_3_soft_{team_name}_{run_id}.json")

    # ── Task 2.3 HARD ─────────────────────────────────────────────────────────
    recs = []
    for mid in ids:
        active = [cat for cat in CATS if fused[mid].get(f"t23_{cat}_fused", 0.0) >= 0.5]
        recs.append({"test_case": TCASE, "id": mid,
                     "value": active if active else ["NO"]})
    write(recs, f"task2_3_hard_{team_name}_{run_id}.json")

    print(f"\n  All 6 files written to: {output_dir}/")

# ══════════════════════════════════════════════════════════════════════════════
# PART 5: QUICK COMPARISON REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(scores, fused, n=10):
    """Show before/after comparison for first n memes."""
    print("\n" + "="*70)
    print("BEFORE vs AFTER FUSION — first 10 memes")
    print("="*70)
    print(f"{'ID':<10} {'Old YES':>10} {'New YES':>10} {'Change':>10} {'Arousal':>10} {'Reasoning'}")
    print("-"*70)
    for mid in list(fused.keys())[:n]:
        old = scores[mid].get("t21_yes", 0.5)
        new = fused[mid].get("t21_yes_fused", old)
        chg = new - old
        rsn = fused[mid].get("llm_reasoning", "rule-based")[:30]
        arrow = "↑" if chg > 0.01 else ("↓" if chg < -0.01 else "→")
        print(f"  {mid:<8} {old:>10.4f} {new:>10.4f} {arrow}{abs(chg):>8.4f}   {rsn}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EXIST 2026 Sensor Fusion")
    parser.add_argument("--mode",  choices=["rule","llm"], default="rule",
                        help="rule=fast Python, llm=Ollama model")
    parser.add_argument("--model", default=CONFIG["ollama_model"],
                        help="Ollama model name (llama3.2, mistral, qwen2.5 ...)")
    parser.add_argument("--alpha", type=float, default=CONFIG["fusion_alpha"],
                        help="Fusion weight 0-1 (only for rule mode)")
    parser.add_argument("--test_json",  default=CONFIG["test_json"])
    parser.add_argument("--input_dir",  default=CONFIG["input_dir"])
    parser.add_argument("--output_dir", default=CONFIG["output_dir"])
    parser.add_argument("--run_id",     type=int, default=CONFIG["run_id"])
    args = parser.parse_args()

    print("\nEXIST 2026 Sensor Fusion Pipeline")
    print("="*50)
    print(f"  Mode:       {args.mode.upper()}")
    if args.mode == "llm":
        print(f"  Model:      {args.model}")
    print(f"  Alpha:      {args.alpha}")
    print(f"  Input:      {args.input_dir}")
    print(f"  Test JSON:  {args.test_json}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Run ID:     {args.run_id}")
    print()

    # Step 1: Load existing scores
    print("Step 1: Loading existing VLM predictions...")
    scores = load_existing_scores(args.input_dir, CONFIG["team_name"], 2)  # run2 = your current run
    if not scores:
        print(f"ERROR: No scores found in {args.input_dir}. Check CONFIG['input_dir'].")
        sys.exit(1)

    # Step 2: Load sensor data from test JSON
    print("\nStep 2: Loading sensor data from test JSON...")
    if not os.path.exists(args.test_json):
        print(f"  [WARN] Test JSON not found: {args.test_json}")
        print("  Running without sensor data — scores will be copied unchanged.")
        sensor_map = {}
    else:
        sensor_map = load_sensor_data(args.test_json)

    # Step 3: Fuse
    print(f"\nStep 3: Fusing scores ({args.mode} mode)...")
    if args.mode == "llm":
        CONFIG["ollama_model"] = args.model
        fused = llm_fuse_all(scores, sensor_map, CONFIG)
    else:
        fused = rule_fuse_all(scores, sensor_map, args.alpha)
        # Move fused keys into place (rule mode uses different key names)
        print(f"  Rule-based fusion complete for {len(fused)} memes")

    # Step 4: Write output
    print(f"\nStep 4: Writing output files...")
    write_output_files(fused, args.output_dir, CONFIG["team_name"], args.run_id)

    # Step 5: Comparison report
    print_comparison(scores, fused)

    print(f"\nDone! Submit the folder: {args.output_dir}/")

if __name__ == "__main__":
    main()