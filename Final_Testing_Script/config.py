"""
EXIST 2026 — Shared Configuration
All constants, model names, category labels used across all subtask scripts.
"""

# ── Ollama server ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Available models (pass model_key to scripts) ──────────────────────
MODELS = {
    "moondream":     "moondream:latest",
    "qwen":          "qwen2.5vl:latest",
    "llama":      "llama3.1:latest"
}

# ── Team name (used in output filenames) ──────────────────────────────
TEAM_NAME = "Aryan Somnath Banerjee"   # ← change this to your actual team name

# ── test_case field required by PyEvALL / EXIST format ────────────────
TEST_CASE = "Aryan Somnath Banerjee"

# ── Subtask 2.3 category labels (exact strings) ───────────────────────
CATEGORIES_2_3 = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
]

# ── All valid labels per subtask (for validation) ─────────────────────
VALID_LABELS_2_1 = ["YES", "NO"]
VALID_LABELS_2_2 = ["NO", "DIRECT", "JUDGEMENTAL"]
VALID_LABELS_2_3 = ["NO"] + CATEGORIES_2_3

# ── LLM generation settings ───────────────────────────────────────────
LLM_TEMPERATURE = 0.2
LLM_NUM_PREDICT = 512
