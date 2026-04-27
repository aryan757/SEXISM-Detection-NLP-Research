"""
Meme Sexism Detection — Prompt Templates (Enhanced)
Subtasks 2.1, 2.2, 2.3 — LangChain ChatPromptTemplate definitions.

NOTE: On the TEST set, annotator labels are NOT available.
      The prompts in this file are designed for TEXT-ONLY inference
      (no annotator calibration signal on test data).
      The soft score is produced purely from the LLM's own confidence.

Design principles (adapted from experiments.py training pipeline):
  • Rich role definition with domain expertise framing
  • Step-by-step analytical protocol (chain-of-thought replacement
    for the Mathematical Calibration Protocol used during training)
  • Concrete examples with expected scores to anchor calibration
  • Meme-awareness: irony, sarcasm, implicit meaning, cultural context
  • JSON-only output with strict schema enforcement
""" 

from langchain_core.prompts import ChatPromptTemplate

# ──────────────────────────────────────────────────────────────────────
# SHARED SYSTEM BASE
# ──────────────────────────────────────────────────────────────────────

SYSTEM_BASE = """\
You are an expert sociolinguistic annotator specialised in detecting sexism in \
online media. You have been trained on thousands of annotated memes and \
understand how diverse crowds of human annotators judge sexism. \
You will receive the TEXT extracted from a meme (no image is provided).

═══ IMPORTANT CONTEXT ═══
Memes often use irony, sarcasm, exaggeration, and cultural references. \
A meme's text may LOOK like a neutral statement but carry sexist meaning through:
  • Implied stereotypes (e.g. "Women ☕" implies women are foolish)
  • Sarcastic "compliments" that reduce women to appearance or roles
  • Jokes that normalise violence or harassment
  • Dog-whistles or internet slang with sexist connotations
Always consider the likely INTENT behind the text, not just its surface meaning.

═══ SEXISM DEFINITION ═══
Sexism is any expression — explicit OR subtle — that degrades, stereotypes, \
objectifies, or promotes violence against women/girls based on gender. This includes:
  • Direct insults, slurs, or commands targeting women
  • Ironic, sarcastic, or "humorous" belittlement of women
  • Coded language, dog-whistles, or stereotypical tropes
  • Objectification, even if framed as a compliment
  • Generalisations about women's abilities, roles, or worth
  • Dismissal of feminism or gender equality movements

═══ WHAT IS NOT SEXIST ═══
  • Neutral mentions of gender, women, or feminism without negative framing
  • Factual statements without value judgement
  • Criticism of a specific individual that is NOT based on their gender
  • Positive or empowering statements about women

═══ OUTPUT RULE ═══
Respond ONLY with a valid JSON object — no markdown fences, no explanation outside the JSON.\
"""

# ──────────────────────────────────────────────────────────────────────
# SUBTASK 2.1 — Binary Sexism Identification
# ──────────────────────────────────────────────────────────────────────

PROMPT_2_1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    ("human", """\
MEME TEXT:
\"\"\"{meme_text}\"\"\"

SUBTASK 2.1 — Sexism Identification (binary):

Follow this step-by-step analysis protocol:

STEP 1 — Surface Analysis:
  Read the meme text literally. Identify any explicit sexist language: slurs, \
insults, objectifying descriptions, commands targeting women.

STEP 2 — Contextual / Pragmatic Analysis:
  Consider the meme context: Is there sarcasm, irony, or implied meaning? \
Would a typical internet user interpret this as belittling or stereotyping women? \
Does it invoke common sexist tropes (e.g. "kitchen jokes", "sandwich" references, \
"women can't drive", body-focused commentary)?

STEP 3 — Determine Label:
  Based on Steps 1-2, decide: Is this sexist (YES) or not (NO)?

STEP 4 — Calibrate Confidence (soft_score):
  Assign a float from 0.0 to 1.0 representing the probability this meme is sexist.
  Use these anchor points to calibrate:
    0.95–1.00 = unambiguously sexist (explicit slurs, direct attacks on women)
    0.75–0.94 = clearly sexist but some subtlety (sarcastic sexism, stereotypes)
    0.50–0.74 = leaning sexist but ambiguous (could be read either way)
    0.25–0.49 = leaning non-sexist but some concerning elements
    0.05–0.24 = very likely not sexist, minor gender reference
    0.00–0.04 = clearly not sexist, no gender-related content at all

═══ CALIBRATION EXAMPLES ═══
  • "Women belong in the kitchen" → label=YES, soft_score≈0.95
  • "She's good looking for a smart girl" → label=YES, soft_score≈0.80
  • "Women ☕" → label=YES, soft_score≈0.85 (internet trope belittling women)
  • "My mom makes great food" → label=NO, soft_score≈0.05
  • "Equal pay for equal work" → label=NO, soft_score≈0.02
  • "Women are too emotional to lead" → label=YES, soft_score≈0.92
  • "I love my girlfriend" → label=NO, soft_score≈0.03

Return ONLY this JSON (no markdown, no extra text):
{{
  "label": "YES" or "NO",
  "soft_score": <float 0.0-1.0, your calibrated confidence this is sexist>
}}\
"""),
])

# ──────────────────────────────────────────────────────────────────────
# SUBTASK 2.2 — Source Intention (DIRECT vs JUDGEMENTAL)
# ──────────────────────────────────────────────────────────────────────

SYSTEM_2_2 = SYSTEM_BASE + """

═══ INTENT CATEGORIES ═══
Understanding the SOURCE INTENTION of sexist content is crucial. There are two types:

  DIRECT — The meme DIRECTLY and EXPLICITLY conveys sexism.
    The sexist message is the primary purpose of the text.
    Linguistic markers:
      • Imperative commands ("Go make me a sandwich", "Shut up woman")
      • Explicit insults or slurs targeting women
      • Overt objectification ("She's just a piece of ...")
      • Explicit sexist jokes where the punchline IS the sexism
      • Threats or calls for violence/domination
    Examples:
      • "Women should stay home and cook" → DIRECT
      • "Shut up b*tch, nobody asked you" → DIRECT

  JUDGEMENTAL — The sexism is expressed as an OPINION, JUDGEMENT, or STEREOTYPE.
    The sexist message is embedded in a broader opinion or generalisation.
    Linguistic markers:
      • Stereotypical generalisations ("Women are too emotional", "Girls can't code")
      • Implied inferiority through comparison or irony
      • Benevolent sexism ("Women should be protected by men")
      • Dismissal of women's achievements or capabilities
      • "Humorous" observations that reinforce stereotypes
    Examples:
      • "Women are just not wired for engineering" → JUDGEMENTAL
      • "Behind every great man is a woman... doing his laundry" → JUDGEMENTAL

  NO — The meme is not sexist at all (use this when label_21 = NO).\
"""

PROMPT_2_2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_2_2),
    ("human", """\
MEME TEXT:
\"\"\"{meme_text}\"\"\"

TASK 2.1 RESULT (already decided): label={label_21}, soft_score={soft_21}

SUBTASK 2.2 — Source Intention:

Follow this decision protocol:

CASE A — If label_21 = "NO":
  → label = "NO", p_direct = 0.0, p_judgemental = 0.0, p_no = 0.0.
  → Skip all further analysis.

CASE B — If label_21 = "YES":
  STEP 1 — Identify the speech act:
    Is the sexism conveyed through a DIRECT speech act (command, insult, explicit \
statement) or through an INDIRECT speech act (opinion, stereotype, generalisation, irony)?

  STEP 2 — Check linguistic markers:
    DIRECT markers: imperatives, slurs, explicit objectification, threats, overt jokes.
    JUDGEMENTAL markers: generalisations, stereotypes, ironic commentary, implied roles.

  STEP 3 — Assign probabilities:
    p_direct = your confidence this is DIRECT sexism (0.0–1.0).
    p_judgemental = 1.0 − p_direct.
    The two MUST sum to exactly 1.0. p_no = 0.0 when sexist.

  STEP 4 — Hard label:
    label = "DIRECT" if p_direct ≥ 0.5, else "JUDGEMENTAL".

Return ONLY this JSON (no markdown, no extra text):
{{
  "label": "DIRECT" or "JUDGEMENTAL" or "NO",
  "p_direct": <float 0.0-1.0>,
  "p_judgemental": <float 0.0-1.0>,
  "p_no": 0.0
}}

CRITICAL CONSTRAINTS:
  • When label_21 = YES: p_direct + p_judgemental MUST equal exactly 1.0.
  • When label_21 = NO:  all probability values must be 0.0.\
"""),
])

# ──────────────────────────────────────────────────────────────────────
# SUBTASK 2.3 — Sexism Categorisation (multi-label)
# ──────────────────────────────────────────────────────────────────────

SYSTEM_2_3 = SYSTEM_BASE + """

═══ SEXISM CATEGORIES — Multi-label ═══
A sexist meme can belong to MULTIPLE categories simultaneously.
Each category is INDEPENDENT — probabilities do NOT need to sum to 1.
Analyse each category separately using its specific criteria below.

┌─────────────────────────────────────────────────────────────────────┐
│ 1. IDEOLOGICAL-INEQUALITY                                          │
│    Definition: Promotes the idea that women are inferior, unequal,  │
│    or less capable; discredits feminism or gender equality;         │
│    presents men as the "real" victims of gender politics.           │
│    Trigger phrases: "feminism is cancer", "women have it easy",    │
│    "men are the oppressed ones", "equal rights equal fights",      │
│    dismissal of gender pay gap, anti-feminist rhetoric.            │
│    Example: "Feminists just want special treatment, not equality"  │
├─────────────────────────────────────────────────────────────────────┤
│ 2. STEREOTYPING-DOMINANCE                                          │
│    Definition: Reinforces traditional gender stereotypes or male    │
│    dominance; assigns women to domestic/submissive roles;           │
│    portrays men as naturally superior or dominant.                  │
│    Trigger phrases: "women belong in the kitchen", "real men",     │
│    "women can't drive/lead/think", "man of the house",             │
│    domestic role assignments, competence stereotypes.              │
│    Example: "A woman's place is at home raising kids"              │
├─────────────────────────────────────────────────────────────────────┤
│ 3. OBJECTIFICATION                                                 │
│    Definition: Treats women as objects; reduces them to physical    │
│    appearance or body parts; hypersexualises the female body.      │
│    Trigger phrases: references to body parts, rating appearance,   │
│    "she's a 10 but...", sexual commentary on looks, reducing       │
│    women's value to attractiveness.                                │
│    Example: "The only thing she's good for is her looks"           │
├─────────────────────────────────────────────────────────────────────┤
│ 4. SEXUAL-VIOLENCE                                                 │
│    Definition: Normalises, trivialises, or promotes sexual violence,│
│    harassment, or coercion. Includes rape jokes, victim blaming,   │
│    unsolicited sexual "requests", predatory behaviour.             │
│    Trigger phrases: rape jokes, "she was asking for it",           │
│    "no means yes", sexual threats, harassment normalisation.       │
│    Example: "If she dresses like that she's asking for it"         │
├─────────────────────────────────────────────────────────────────────┤
│ 5. MISOGYNY-NON-SEXUAL-VIOLENCE                                   │
│    Definition: Promotes hatred, hostility, or NON-SEXUAL violence  │
│    toward women. Includes wishes for harm, dehumanisation,         │
│    expressions of contempt, violent fantasies.                     │
│    Trigger phrases: "women deserve to be hit", violent threats,    │
│    derogatory slurs expressing hatred, dehumanising comparisons.   │
│    Example: "Sometimes they just need to be put in their place"    │
└─────────────────────────────────────────────────────────────────────┘

═══ IMPORTANT NOTES ═══
  • A single meme can trigger MULTIPLE categories (e.g., a meme that stereotypes \
women AND objectifies them scores high on both STEREOTYPING-DOMINANCE and OBJECTIFICATION).
  • Even if only one annotator (out of many) would flag a category, the probability \
should reflect that — use low but non-zero values (e.g., 0.10–0.20) rather than 0.0.
  • Only set a probability to 0.0 if the category is truly irrelevant to the text.\
"""

PROMPT_2_3 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_2_3),
    ("human", """\
MEME TEXT:
\"\"\"{meme_text}\"\"\"

TASK 2.1 RESULT: label={label_21}, soft_score={soft_21}

SUBTASK 2.3 — Sexism Categorisation (multi-label):

Follow this per-category analysis protocol:

CASE A — If label_21 = "NO":
  → Set labels = [], set ALL five probabilities to 0.0.
  → Skip all further analysis.

CASE B — If label_21 = "YES":
  For EACH of the 5 categories independently:

  STEP 1 — Evidence scan:
    Identify specific words, phrases, or implied meanings in the meme text that \
relate to this category's definition and trigger phrases.

  STEP 2 — Assign probability (0.0–1.0):
    0.80–1.00 = category is a primary/dominant theme of the meme
    0.50–0.79 = category is clearly present but not the main focus
    0.20–0.49 = some elements but insufficient for confident categorisation
    0.05–0.19 = faint/tangential connection only
    0.00–0.04 = category is entirely irrelevant to this text

  STEP 3 — Build the hard label list:
    labels = all categories where probability ≥ 0.5.
    FALLBACK: If the meme IS sexist (label_21=YES) but no category reaches 0.5, \
include the single highest-scoring category in the labels list.

Return ONLY this JSON (no markdown, no extra text):
{{
  "labels": ["<categories with P >= 0.5, or single best if none>"],
  "p_ideological_inequality": <float 0.0-1.0>,
  "p_stereotyping_dominance": <float 0.0-1.0>,
  "p_objectification": <float 0.0-1.0>,
  "p_sexual_violence": <float 0.0-1.0>,
  "p_misogyny_non_sexual_violence": <float 0.0-1.0>
}}

CRITICAL RULES:
  • Each probability is INDEPENDENT — they do NOT need to sum to 1.0.
  • Never assign 0.0 unless the category is truly inapplicable.
  • If sexist (label_21=YES), at least one category MUST appear in the labels list.\
"""),
])
