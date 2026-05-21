import nbformat as nbf
import os

notebook_path = '/Users/aryan/Desktop/aryan_Exist_2026_dataset/Final_Testing_Script/LIME_Interpretability.ipynb'

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("# LIME Interpretability for Sexism Detection\n\nThis notebook demonstrates how to use **LIME** (Local Interpretable Model-Agnostic Explanations) to highlight the specific words in the meme text that cause the model to classify it as sexist (`YES`).\n\nSince we don't have the original live LLM endpoint to query for perturbations, we train a **surrogate linear model** (TF-IDF + Logistic Regression) on the predicted test labels. We then use LIME on this surrogate model to extract the keyword attributions."))

code_cells = [
    """\
# Install required libraries if not present
!pip install -q lime scikit-learn pandas
""",
    """\
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

# Fix for LIME in newer IPython environments
import IPython.core.display
import IPython.display
IPython.core.display.display = IPython.display.display
IPython.core.display.HTML = IPython.display.HTML
""",
    """\
# Define file paths
TEST_DATA_PATH = '/Users/aryan/Desktop/aryan_Exist_2026_dataset/EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/test/EXIST2026_test_clean.json'
PREDICTIONS_PATH = '/Users/aryan/Desktop/aryan_Exist_2026_dataset/Final_Testing_Script/exist2026_AryanSomnathBanerjee_run2/task2_1_hard_AryanSomnathBanerjee_2.json'
""",
    """\
# Load test dataset
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

texts = []
ids = []
for k, v in test_data.items():
    ids.append(str(k))
    texts.append(v.get('text', ''))

df_test = pd.DataFrame({'id': ids, 'text': texts})
display(df_test.head())
""",
    """\
# Load predictions
with open(PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
    preds = json.load(f)

pred_ids = [str(p['id']) for p in preds]
pred_labels = [p['value'] for p in preds]

df_preds = pd.DataFrame({'id': pred_ids, 'label': pred_labels})
df_preds['target'] = df_preds['label'].map({'YES': 1, 'NO': 0})
display(df_preds.head())
""",
    """\
# Merge data
df = pd.merge(df_test, df_preds, on='id', how='inner')
print(f"Total instances: {len(df)}")
display(df.head())
""",
    """\
# We will train a simple surrogate model (TF-IDF + Logistic Regression) on the predicted labels.
# This allows us to use LIME to understand which words are highly correlated with the 'YES' predictions.
# Since the dataset contains both Spanish and English, we use a regex tokenizer that works for both.
vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r'(?u)\\b\\w+\\b')
classifier = LogisticRegression(random_state=42, class_weight='balanced')

# Pipeline
pipeline = make_pipeline(vectorizer, classifier)

# Train the surrogate model on the predictions
pipeline.fit(df['text'], df['target'])
print(f"Surrogate model accuracy against predictions: {pipeline.score(df['text'], df['target']):.4f}")
""",
    """\
# Initialize LIME Text Explainer
class_names = ['NO (Not Sexist)', 'YES (Sexist)']
explainer = LimeTextExplainer(class_names=class_names)
""",
    """\
# Let's find some examples that were predicted as YES (Sexist)
sexist_examples = df[df['label'] == 'YES'].sample(5, random_state=42)

for idx, row in sexist_examples.iterrows():
    print("="*80)
    print(f"ID: {row['id']}")
    print(f"Text: {row['text']}")
    print(f"Predicted Label: {row['label']}")
    
    # Generate explanation
    exp = explainer.explain_instance(
        row['text'], 
        pipeline.predict_proba, 
        num_features=6
    )
    
    # Show explanation in notebook
    exp.show_in_notebook(text=True)
"""
]

for cell_code in code_cells:
    nb.cells.append(nbf.v4.new_code_cell(cell_code))

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created successfully at {notebook_path}")
