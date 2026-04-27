# 🧠 EXIST 2026 — Meme Sexism Detection Pipeline

> **Author:** Aryan Somnath Banerjee  
> **Task:** EXIST 2026 Shared Task — Learning with Disagreement (LeWiDi) — Task 2 (Memes)  
> **Approach:** Zero-Shot LLM-as-Annotator via LangChain + Ollama (Text-Only, No Fine-Tuning)

---

## 📌 Overview

This repository contains my submission for the **EXIST 2026 Shared Task** on sexism identification in memes. The task addresses three progressively complex subtasks under the **Learning with Disagreement (LeWiDi)** framework, where the goal is not only to predict hard labels but also to produce **soft (probabilistic) labels** that capture the inherent subjectivity and annotator disagreement in sexism perception.

My approach uses a **zero-shot, prompt-engineered LLM pipeline** powered by locally-hosted models via [Ollama](https://ollama.com), orchestrated through [LangChain](https://www.langchain.com/). No fine-tuning is performed — the entire system relies on carefully designed chain-of-thought prompts and calibration protocols.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXIST 2026 Pipeline                         │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │  Task 2.1 │───▶│  Task 2.2 │    │  Task 2.3 │                │
│  │  Binary   │    │  Intent   │    │  Category │                │
│  │ YES / NO  │    │ DIR / JDG │    │ Multi-lbl │                │
│  └──────────┘    └──────────┘    └──────────┘                  │
│       │               ▲               ▲                         │
│       │               │               │                         │
│       └───── label_21 cascades ───────┘                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LangChain + Ollama (Local LLM)              │   │
│  │         Prompt → ChatOllama → JsonOutputParser           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Cascaded Design

The pipeline follows a **cascaded architecture** where Task 2.1 predictions flow into Tasks 2.2 and 2.3:

1. **Task 2.1** runs first and produces binary sexism labels (`YES`/`NO`)
2. **Task 2.2** and **Task 2.3** receive the 2.1 predictions as context
3. Non-sexist memes (`label_21 = NO`) are **short-circuited** — the LLM is never called, and deterministic zero-probability outputs are returned

This cascading avoids wasted compute and ensures logical consistency across subtasks.

---

## 📋 Subtasks

### Subtask 2.1 — Sexism Identification (Binary)
- **Goal:** Determine if a meme's text is sexist (`YES`) or not (`NO`)
- **Output:**  
  - **Hard label:** `YES` or `NO`  
  - **Soft label:** `{"YES": float, "NO": float}` (probabilities summing to 1.0)

### Subtask 2.2 — Source Intention
- **Goal:** If sexist, classify the **intent** as `DIRECT` (explicit sexism) or `JUDGEMENTAL` (opinion/stereotype-based)
- **Output:**  
  - **Hard label:** `DIRECT`, `JUDGEMENTAL`, or `NO`  
  - **Soft label:** `{"DIRECT": float, "JUDGEMENTAL": float, "NO": float}`

### Subtask 2.3 — Sexism Categorisation (Multi-Label)
- **Goal:** Assign one or more sexism categories from:
  - `IDEOLOGICAL-INEQUALITY`
  - `STEREOTYPING-DOMINANCE`
  - `OBJECTIFICATION`
  - `SEXUAL-VIOLENCE`
  - `MISOGYNY-NON-SEXUAL-VIOLENCE`
- **Output:**  
  - **Hard label:** Array of applicable categories (or `["NO"]`)
  - **Soft label:** Independent probability per category (do **not** sum to 1.0)

---

## 🔬 Approach & Methodology

### 1. Text-Only Inference
The pipeline processes only the **OCR-extracted text** from memes — no image features are used. This is a deliberate design choice that allows the use of smaller, locally-runnable language models while still capturing the linguistic patterns of sexism (sarcasm, stereotypes, coded language).

### 2. Prompt Engineering Strategy

The core of the approach lies in **meticulously crafted prompts** with several key design principles:

#### a) Rich Role Definition
The LLM is framed as an *"expert sociolinguistic annotator specialised in detecting sexism in online media"* who understands crowd annotation dynamics. This role-priming anchors the model's behaviour.

#### b) Meme-Aware Context
Prompts explicitly teach the model about meme-specific communication patterns:
- Irony and sarcasm (e.g., `"Women ☕"` as a belittling trope)
- Coded language and dog-whistles
- "Humorous" normalisation of sexism
- Sarcastic "compliments" that objectify

#### c) Step-by-Step Analytical Protocol
Each subtask prompt enforces a **structured chain-of-thought** analysis:
- **Surface Analysis** → literal text examination
- **Contextual/Pragmatic Analysis** → intent and cultural reading
- **Label Determination** → evidence-based classification
- **Confidence Calibration** → calibrated soft scores using anchor examples

#### d) Calibration Anchoring
Prompts include **concrete calibration examples** with expected scores to anchor the model's confidence:
```
• "Women belong in the kitchen"         → label=YES, soft_score ≈ 0.95
• "She's good looking for a smart girl" → label=YES, soft_score ≈ 0.80
• "Women ☕"                             → label=YES, soft_score ≈ 0.85
• "My mom makes great food"             → label=NO,  soft_score ≈ 0.05
```

#### e) JSON-Only Output Enforcement
All prompts enforce strict JSON-only responses with no markdown fences, enabling reliable automated parsing via LangChain's `JsonOutputParser`.

### 3. Training Phase — Mathematical Calibration Protocol

During development and training evaluation, a more sophisticated **Mathematical Calibration Protocol** was used (see `EXIST_2026/training/experiments.py`). This protocol:

1. **Computes empirical annotator ratios** (e.g., if 4/6 annotators said YES → R = 0.667)
2. **Constrains LLM adjustments to ±0.05** from the empirical ratio
3. **Produces soft scores tightly anchored** to human annotator agreement

This protocol ensures the model's confidence scores faithfully reflect the **Learning with Disagreement** paradigm, where the "ground truth" is itself probabilistic.

> ⚠️ **On the test set**, annotator labels are unavailable. The inference prompts use the LLM's own calibrated confidence as a proxy, guided by the anchoring examples learned during training prompt design.

### 4. Robust Error Handling

The pipeline includes multiple layers of resilience:
- **Safe JSON parsing** (`safe_parse`) strips markdown fences and recovers from malformed outputs
- **Fallback defaults** for each subtask when the LLM fails
- **Label validation** ensures outputs conform to valid label sets
- **Probability clamping** keeps all values in `[0.0, 1.0]`
- **Normalisation** ensures Task 2.2 probabilities sum to exactly 1.0

---

## 📂 Repository Structure

```
.
├── README.md                          # This file
│
├── Final_Testing_Script/              # 🔹 MAIN INFERENCE PIPELINE (test set)
│   ├── config.py                      # Model names, constants, category labels
│   ├── prompts.py                     # All prompt templates (2.1, 2.2, 2.3)
│   ├── utils.py                       # LLM builder, JSON parsing, output writers
│   ├── run_task2_1.py                 # Subtask 2.1 runner (binary sexism)
│   ├── run_task2_2.py                 # Subtask 2.2 runner (source intention)
│   ├── run_task2_3.py                 # Subtask 2.3 runner (multi-label categories)
│   ├── run_all.py                     # Master pipeline — runs all 3 subtasks
│   └── submission/                    # Generated submission JSON files
│       ├── task2_1_hard_*.json
│       ├── task2_1_soft_*.json
│       ├── task2_2_hard_*.json
│       ├── task2_2_soft_*.json
│       ├── task2_3_hard_*.json
│       └── task2_3_soft_*.json
│
├── EXIST_2026/                        # 🔹 TRAINING & EXPERIMENTATION
│   └── training/
│       ├── experiments.py             # Training pipeline with math calibration
│       ├── filter_memes.py            # Data filtering (EN/ES extraction)
│       ├── requirements.txt           # Python dependencies
│       └── outputs/                   # Training experiment results
│
├── EXIST_2026_TESTING/                # 🔹 OFFICIAL TEST DATA & EVALUATION
│   ├── evaluation/
│   │   ├── exist2025_format_val_V0.2.py   # Official format validator
│   │   ├── baselines/                # Baseline submissions
│   │   └── golds/                    # Gold-standard labels
│   └── EXIST_2026_Memes_Dataset/     # Test meme images & metadata
│
├── EDA/                               # 🔹 EXPLORATORY DATA ANALYSIS
│   ├── exist2026_dataset_analysis_1.pdf
│   ├── label_derivation_explainer_2.pdf
│   └── union_vs_majority_explainer_3.pdf
│
├── EXIST2026_Proposal_v2.pdf          # Task proposal document
├── EXIST_2026_Lab_Guidelines.V0.3.pdf # Lab guidelines
├── paper-87-Exist-2024.pdf            # Prior EXIST 2024 paper
└── paper_135-Exist-2025.pdf           # Prior EXIST 2025 paper
```

---

## ⚙️ Setup & Installation

### Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running locally
- A supported LLM model pulled via Ollama

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/EXIST-2026-Meme-Sexism-Detection.git
cd EXIST-2026-Meme-Sexism-Detection
```

### 2. Install Dependencies
```bash
pip install langchain-core langchain-ollama requests
```

### 3. Pull an Ollama Model
```bash
# Recommended model (best balance of speed and accuracy)
ollama pull qwen2.5vl:latest

# Alternative models
ollama pull llama3.1:latest
ollama pull moondream:latest
```

### 4. Start Ollama Server
```bash
ollama serve
```

---

## 🚀 Usage

### Run the Full Pipeline (All 3 Subtasks)
```bash
cd Final_Testing_Script

python run_all.py \
    --test_path  ../EXIST_2026/training/EXIST2026_testing_english_spanish_filtered.json \
    --output_dir ./submission \
    --model_key  qwen \
    --run_id     1
```

### Run Individual Subtasks
```bash
# Task 2.1 — Binary Sexism Detection
python run_task2_1.py \
    --test_path ../EXIST_2026/training/EXIST2026_testing_english_spanish_filtered.json \
    --output_dir ./submission \
    --model_key qwen

# Task 2.2 — Source Intention (requires 2.1 output)
python run_task2_2.py \
    --test_path ../EXIST_2026/training/EXIST2026_testing_english_spanish_filtered.json \
    --pred_21_path ./submission/task2_1_hard_Aryan\ Somnath\ Banerjee_1.json \
    --output_dir ./submission \
    --model_key qwen

# Task 2.3 — Sexism Categorisation (requires 2.1 output)
python run_task2_3.py \
    --test_path ../EXIST_2026/training/EXIST2026_testing_english_spanish_filtered.json \
    --pred_21_path ./submission/task2_1_hard_Aryan\ Somnath\ Banerjee_1.json \
    --output_dir ./submission \
    --model_key qwen
```

### Quick Test (First 10 Records)
```bash
python run_all.py \
    --test_path ../EXIST_2026/training/EXIST2026_testing_english_spanish_filtered.json \
    --output_dir ./submission \
    --model_key qwen \
    --max_records 10
```

### Available Models
| Key        | Ollama Model          | Notes                         |
|------------|-----------------------|-------------------------------|
| `qwen`     | `qwen2.5vl:latest`   | Recommended — best accuracy   |
| `llama`    | `llama3.1:latest`     | Good general-purpose fallback |
| `moondream`| `moondream:latest`    | Lightweight, faster inference |

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
| **Text-only (no vision)** | Enables use of smaller local models; meme text carries the majority of sexism signal |
| **Zero-shot (no fine-tuning)** | Faster iteration, no GPU training infrastructure needed, demonstrates prompt engineering effectiveness |
| **Local inference (Ollama)** | Full privacy, no API costs, reproducible results, no rate limits |
| **LangChain orchestration** | Clean prompt→LLM→parser chains, easy model swapping, structured output parsing |
| **Cascaded subtask design** | Logical consistency — non-sexist memes are never analysed for intent or categories |
| **Calibration anchoring** | Concrete examples in prompts ground the model's confidence scores in realistic ranges |
| **Bounded adjustment (training)** | The ±0.05 constraint during training ensures LLM defers to annotator consensus |

---

## 📈 Evaluation Metrics

The EXIST 2026 shared task uses the following primary metrics:
- **ICM-Soft-Norm** — Information-Contrast Model for soft (probabilistic) evaluation
- **ICM-Norm** — Normalised Information-Contrast Model for hard labels
- **F1-Macro** — Standard macro-averaged F1 score

The `experiments.py` training script includes a simplified ICM-Soft-Norm implementation for local evaluation during development.

---

## 📝 Exploratory Data Analysis

The `EDA/` directory contains detailed analysis reports:
- **Dataset Analysis** — Distribution of labels, annotator demographics, language splits
- **Label Derivation** — How soft labels are computed from individual annotator votes
- **Union vs Majority Voting** — Comparison of label aggregation strategies and their impact

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM Inference | [Ollama](https://ollama.com) (local) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| Prompt Design | Custom `ChatPromptTemplate` with structured CoT |
| Output Parsing | `JsonOutputParser` with safe fallback recovery |
| Language | Python 3.10+ |
| Evaluation | PyEvALL (official EXIST toolkit) |

---

## 📚 References

- [EXIST 2026 Shared Task](http://nlp.uned.es/exist2026/) — Official task page
- [EXIST 2025 Paper](./paper_135-Exist-2025.pdf) — Previous edition methodology
- [EXIST 2024 Paper](./paper-87-Exist-2024.pdf) — Previous edition methodology
- [Learning with Disagreement (LeWiDi)](https://le-wi-di.github.io/) — Soft label paradigm
- [Ollama](https://ollama.com) — Local LLM server
- [LangChain](https://www.langchain.com/) — LLM orchestration framework

---

## 📄 License

This project is developed for academic research as part of the EXIST 2026 shared task at CLEF 2026.

---

## 🙏 Acknowledgments

- **EXIST 2026 Organisers** for designing the shared task and providing annotated datasets
- **Ollama** for making local LLM inference accessible
- **LangChain** for the robust prompt engineering and chain abstraction framework
