# 🧠 EXIST 2026 — Multimodal Meme Sexism Detection Pipeline

> **Author:** Aryan Somnath Banerjee
> **Task:** EXIST 2026 Shared Task — Learning with Disagreement (LeWiDi) — Task 2 (Memes)
> **Approach:** Multimodal Sensor Fusion — Vision-Language Model (VLM) + Physiological Signals (EEG · Eye-Tracking · Heart Rate) + LLM-based Reasoning

---

## 📌 Overview

This repository contains the submission for the **EXIST 2026 Shared Task** on sexism identification in memes. The task addresses three progressively complex subtasks under the **Learning with Disagreement (LeWiDi)** framework, where the goal is not only to predict hard labels but also to produce **soft (probabilistic) labels** that capture the inherent subjectivity and annotator disagreement in sexism perception.

The pipeline has evolved into a **multimodal sensor fusion architecture** that combines:
1. **VLM scores** — zero-shot soft predictions from a Vision-Language Model (Qwen2.5-VL via Ollama)
2. **Physiological arousal signals** — EEG brainwave power, eye-tracking (pupil dilation, fixations, saccades), and heart rate variability from human annotators who viewed the same memes
3. **LLM-based fusion** — a second LLM pass that interprets the sensor signals and adjusts VLM confidence scores accordingly

> **Key Insight:** When a human's body reacts strongly (high pupil dilation, elevated heart rate, theta-band EEG spikes) while viewing a meme, that physiological response is a strong signal the content is disturbing — even if the viewer themselves is unsure. The pipeline converts these biological reactions into label calibration signals.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXIST 2026 Sensor Fusion Pipeline                    │
│                                                                         │
│  ┌──────────────┐     ┌───────────────────────────────────────────────┐ │
│  │ Meme Image   │────▶│  VLM (Qwen2.5-VL / Ollama)                    │ │
│  │ + OCR Text   │     │  → Task 2.1 / 2.2 / 2.3 soft scores           │ │
│  └──────────────┘     └──────────────────┬────────────────────────────┘ │
│                                          │                               │
│  ┌──────────────────────────────────┐    │                               │
│  │   Physiological Sensor Data      │    │                               │
│  │  ┌──────┐  ┌──────┐  ┌───────┐  │    │                               │
│  │  │ EEG  │  │  ET  │  │  HR   │  │    │                               │
│  │  │16ch  │  │ Gaze │  │Garmin │  │    │                               │
│  │  └──┬───┘  └──┬───┘  └───┬───┘  │    │                               │
│  └─────┼─────────┼──────────┼──────┘    │                               │
│        └──────── Arousal Score ──────────┘                               │
│                       │                                                  │
│              ┌─────────▼──────────┐                                      │
│              │  Fusion Layer      │  Rule-based  OR  LLM-based (Ollama)  │
│              │  ± score adjustment│                                      │
│              └─────────┬──────────┘                                      │
│                        │                                                  │
│         ┌──────────────┼──────────────┐                                  │
│         ▼              ▼              ▼                                  │
│    Task 2.1       Task 2.2       Task 2.3                                │
│  (YES / NO)    (DIRECT / JDG)  (Multi-label)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Subtasks

### Subtask 2.1 — Sexism Identification (Binary)
- **Goal:** Determine if a meme is sexist (`YES`) or not (`NO`)
- **Output:** Hard label + Soft `{YES: float, NO: float}`

### Subtask 2.2 — Source Intention
- **Goal:** If sexist, classify intent as `DIRECT` (explicit) or `JUDGEMENTAL` (implicit/stereotype-based)
- **Output:** Hard label + Soft `{DIRECT: float, JUDGEMENTAL: float, NO: float}`

### Subtask 2.3 — Sexism Categorisation (Multi-Label)
- **Goal:** Assign one or more categories:
  - `IDEOLOGICAL-INEQUALITY`
  - `STEREOTYPING-DOMINANCE`
  - `OBJECTIFICATION`
  - `SEXUAL-VIOLENCE`
  - `MISOGYNY-NON-SEXUAL-VIOLENCE`
- **Output:** Hard label array + Independent category probabilities

---

## 🔬 Approach & Methodology

### 1. VLM Baseline (Text + Vision)
The first stage uses **Qwen2.5-VL** (multimodal) via Ollama to produce initial soft scores from the meme image and OCR text. Prompts use chain-of-thought reasoning with calibration anchoring.

### 2. Physiological Arousal Score Computation
For each meme, sensor readings from human annotators are aggregated into a single **arousal score [0–1]**:

| Sensor | Feature | High Arousal Signal |
|--------|---------|---------------------|
| EEG | Theta-band power (emotional processing) | > 0.5 |
| Eye-Tracking | Pupil diameter (cognitive load) | > 3.5 mm |
| Eye-Tracking | Reaction time (hesitation) | > 12,000 ms |
| Eye-Tracking | Fixation count (intensive study) | > 45 |
| Heart Rate | HR standard deviation (emotional turbulence) | > 1.5 bpm |

### 3. Fusion Layer
Two fusion strategies are supported:

#### A. Rule-Based Fusion
Deterministic score adjustments based on arousal thresholds:
- `arousal > 0.65` → confidence in YES increases by 0.05–0.15
- `arousal < 0.35` → confidence in YES decreases
- Bounded at **±0.20** from the original VLM score

#### B. LLM-Based Fusion (Ollama)
A second LLM (Qwen2.5 text model) receives the VLM scores + sensor features and generates adjusted scores in JSON format. This allows nuanced, context-aware calibration.

### 4. Cascaded Subtask Design
Task 2.1 predictions flow into Tasks 2.2 and 2.3:
- Non-sexist memes are **short-circuited** — no LLM call, deterministic zero outputs
- Avoids wasted compute and ensures logical consistency

### 5. Robust Error Handling
- Safe JSON parsing with markdown fence stripping
- Fallback defaults for all subtasks
- Probability clamping to `[0.0, 1.0]`
- Task 2.2 normalisation to sum to 1.0

---

## 📂 Repository Structure

```
.
├── README.md                          # This file
│
├── Final_Testing_Script/              # 🔹 MAIN INFERENCE PIPELINE
│   ├── config.py                      # Model names, constants, category labels
│   ├── prompts.py                     # All VLM prompt templates (2.1, 2.2, 2.3)
│   ├── utils.py                       # LLM builder, JSON parsing, output writers
│   ├── run_task2_1.py                 # Subtask 2.1 runner (binary sexism)
│   ├── run_task2_2.py                 # Subtask 2.2 runner (source intention)
│   ├── run_task2_3.py                 # Subtask 2.3 runner (multi-label categories)
│   ├── run_all.py                     # Master pipeline — runs all 3 subtasks
│   ├── sensor_fusion_final.py         # 🔥 Sensor fusion pipeline (VLM + EEG + ET + HR)
│   ├── generate_realistic_run2.py     # Run 2 generation script
│   ├── generate_run4.py               # Run 4 generation script
│   ├── run_validator.py               # Submission format validator
│   └── fix_and_rename.py             # Submission file fixer/renamer
│
├── EXIST_2026/                        # 🔹 TRAINING & EXPERIMENTATION
│   └── training/
│       ├── experiments.py             # Training pipeline with math calibration
│       ├── filter_memes.py            # Data filtering (EN/ES extraction)
│       └── requirements.txt           # Python dependencies
│
├── EXIST_2026_TESTING/                # 🔹 OFFICIAL TEST DATA & EVALUATION
│   └── evaluation/
│       └── exist2025_format_val_V0.2.py   # Official format validator
│
├── EXPERIMENT_ARCHITECTURES/          # 🔹 EXPERIMENTAL ARCHITECTURE SCRIPTS
│
├── EDA_code/                          # 🔹 EXPLORATORY DATA ANALYSIS
│   └── eda.py                         # EDA script (run on training JSON)
│
└── evaluation_metrices_2026/          # 🔹 EVALUATION METRIC SCRIPTS
```

> **Note:** Large data directories (`EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/`, all `.json` dataset files, meme images) are excluded via `.gitignore`. See **Data Setup** below.

---

## ⚙️ Setup & Installation

### Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running locally
- **conda** (recommended) or virtualenv

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/EXIST-2026-Meme-Sexism-Detection.git
cd EXIST-2026-Meme-Sexism-Detection
```

### 2. Create & Activate Environment
```bash
conda create -n my_ml_project python=3.10
conda activate my_ml_project
```

### 3. Install Dependencies
```bash
pip install langchain-core langchain-ollama requests pillow tqdm pandas numpy matplotlib seaborn
```

### 4. Pull Ollama Models
```bash
# VLM for image+text inference (Task 2.1 / 2.2 / 2.3)
ollama pull qwen2.5vl:latest

# Text LLM for sensor fusion (LLM-based fusion mode)
ollama pull qwen2.5:latest

# Alternative models
ollama pull llama3.1:latest
ollama pull moondream:latest
```

### 5. Start Ollama Server
```bash
ollama serve
```

### 6. Data Setup (Not in Git)
Download the EXIST 2026 dataset and place it as follows:
```
EXIST_2026_TESTING/
└── EXIST_2026_Memes_Dataset/
    ├── training/
    │   ├── EXIST2026_training.json   ← 54 MB
    │   └── memes/                    ← ~3984 meme images
    └── test/
        ├── EXIST2026_test.json
        └── memes/
```

---

## 🚀 Usage

### Run the Full Pipeline (All 3 Subtasks)
```bash
cd Final_Testing_Script

python run_all.py \
    --test_path  ../EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/test/EXIST2026_test.json \
    --output_dir ./submission \
    --model_key  qwen \
    --run_id     1
```

### Run Sensor Fusion Pipeline
```bash
cd Final_Testing_Script

python sensor_fusion_final.py \
    --json_path ../EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/training/EXIST2026_training.json \
    --memes_dir ../EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/training/memes \
    --output_dir ./submission \
    --fusion_mode rule \
    --run_id 1
```

Fusion modes:
- `--fusion_mode rule` — Fast deterministic fusion (recommended for full dataset)
- `--fusion_mode llm`  — LLM-based fusion (slower but more nuanced)

### Run EDA on Training Data
```bash
cd EDA_code
python eda.py --json_path ../EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/training/EXIST2026_training.json --split train
```

### Run Individual Subtasks (Text-Only)
```bash
# Task 2.1 — Binary Sexism Detection
python run_task2_1.py \
    --test_path ../EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/test/EXIST2026_test.json \
    --output_dir ./submission \
    --model_key qwen

# Task 2.2 — Source Intention (requires 2.1 output)
python run_task2_2.py \
    --test_path ../EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/test/EXIST2026_test.json \
    --pred_21_path ./submission/task2_1_hard_*.json \
    --output_dir ./submission \
    --model_key qwen

# Task 2.3 — Sexism Categorisation (requires 2.1 output)
python run_task2_3.py \
    --test_path ../EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/test/EXIST2026_test.json \
    --pred_21_path ./submission/task2_1_hard_*.json \
    --output_dir ./submission \
    --model_key qwen
```

### Available Models
| Key | Ollama Model | Notes |
|-----|-------------|-------|
| `qwen` | `qwen2.5vl:latest` | Recommended — multimodal, best accuracy |
| `llama` | `llama3.1:latest` | Good text-only fallback |
| `moondream` | `moondream:latest` | Lightweight, faster inference |

---

## 📊 Output Format

The pipeline produces **6 JSON files** per run, conforming to the [PyEvALL](https://github.com/UNEDLENAR/PyEvALL) evaluation format:

| File | Description |
|------|-------------|
| `task2_1_hard_*.json` | Hard binary labels (`YES`/`NO`) |
| `task2_1_soft_*.json` | Soft probabilities `{YES: p, NO: 1-p}` |
| `task2_2_hard_*.json` | Hard intent labels (`DIRECT`/`JUDGEMENTAL`/`NO`) |
| `task2_2_soft_*.json` | Soft intent probabilities |
| `task2_3_hard_*.json` | Hard category labels (array) |
| `task2_3_soft_*.json` | Independent category probabilities |

### Example Output — Task 2.1 (Soft)
```json
{
  "test_case": "Aryan Somnath Banerjee",
  "id": "100234",
  "value": {"YES": 0.87, "NO": 0.13}
}
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Multimodal VLM** | Captures both visual and textual cues in memes |
| **Physiological fusion** | Biological reactions provide ground-truth arousal signal independent of self-reported labels |
| **Dual fusion modes** | Rule-based is fast & deterministic; LLM-based allows nuanced context-aware adjustment |
| **Local inference (Ollama)** | Full privacy, no API costs, reproducible results, no rate limits |
| **Cascaded subtask design** | Logical consistency — non-sexist memes never analysed for intent or categories |
| **Bounded adjustment ±0.20** | Prevents over-correction; VLM priors are respected |
| **Zero-shot (no fine-tuning)** | Faster iteration, no GPU training infrastructure needed |

---

## 📈 Evaluation Metrics

The EXIST 2026 shared task uses:
- **ICM-Soft-Norm** — Information-Contrast Model for soft (probabilistic) evaluation
- **ICM-Norm** — Normalised Information-Contrast Model for hard labels
- **F1-Macro** — Standard macro-averaged F1 score

---

## 📊 Dataset Statistics (Training Set)

| Metric | Value |
|--------|-------|
| Total memes | 3,984 |
| English memes | 2,005 |
| Spanish memes | 1,979 |
| Annotators per meme | 6 (always) |
| Task 2.1 YES annotations | 13,286 (55.6%) |
| Task 2.1 NO annotations | 10,618 (44.4%) |
| Avg. text length | ~21 words |
| Sensor modalities | EEG (16ch) · Eye-Tracking · Heart Rate |

**Task 2.3 Category Distribution:**

| Category | Count |
|----------|-------|
| STEREOTYPING-DOMINANCE | 4,749 |
| OBJECTIFICATION | 4,549 |
| IDEOLOGICAL-INEQUALITY | 4,081 |
| SEXUAL-VIOLENCE | 2,225 |
| MISOGYNY-NON-SEXUAL-VIOLENCE | 2,047 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| VLM Inference | [Ollama](https://ollama.com) — Qwen2.5-VL (local) |
| LLM Orchestration | [LangChain](https://www.langchain.com/) |
| Sensor Fusion | Custom Python (rule-based + LLM) |
| Prompt Design | Custom `ChatPromptTemplate` with structured CoT |
| Output Parsing | `JsonOutputParser` with safe fallback recovery |
| EDA | Python · Pandas · Matplotlib · Seaborn |
| Language | Python 3.10+ |
| Evaluation | PyEvALL (official EXIST toolkit) |

---

## 📚 References

- [EXIST 2026 Shared Task](http://nlp.uned.es/exist2026/) — Official task page
- [Learning with Disagreement (LeWiDi)](https://le-wi-di.github.io/) — Soft label paradigm
- [Ollama](https://ollama.com) — Local LLM server
- [LangChain](https://www.langchain.com/) — LLM orchestration framework

---

## 📄 License

This project is developed for academic research as part of the EXIST 2026 shared task at CLEF 2026.

---

## 🙏 Acknowledgments

- **EXIST 2026 Organisers** for designing the shared task and providing the annotated multimodal dataset
- **Ollama** for making local LLM & VLM inference accessible
- **LangChain** for the robust prompt engineering and chain abstraction framework
