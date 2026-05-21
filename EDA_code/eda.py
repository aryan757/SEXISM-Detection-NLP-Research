import json
import pandas as pd
import numpy as np
from collections import Counter
import sys

def load_json_data(filepath):
    """Load JSON dataset"""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def analyze_dataset(data):
    """Comprehensive EDA of the EXIST 2026 dataset"""

    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - EXIST 2026 Training Dataset")
    print("="*80)

    # Convert to list of records for easier analysis
    records = list(data.values())

    print(f"\n1. DATASET OVERVIEW")
    print("-" * 80)
    print(f"Total records: {len(records)}")
    print(f"Data type: Dictionary with {len(data)} items")

    # Basic structure
    if records:
        print(f"\nFirst record structure:")
        first_record = records[0]
        print(f"  Keys: {list(first_record.keys())}")

    # 2. BASIC FIELDS ANALYSIS
    print(f"\n2. BASIC FIELDS ANALYSIS")
    print("-" * 80)

    # Text information
    texts = [r.get('text', '') for r in records if isinstance(r, dict)]
    text_lengths = [len(str(t).split()) for t in texts]
    print(f"\nText Field:")
    print(f"  Total records with text: {len([t for t in texts if t])}")
    print(f"  Text length (words) - Mean: {np.mean(text_lengths):.2f}")
    print(f"  Text length (words) - Median: {np.median(text_lengths):.2f}")
    print(f"  Text length (words) - Min: {min(text_lengths)}")
    print(f"  Text length (words) - Max: {max(text_lengths)}")
    print(f"  Text length (chars) - Mean: {np.mean([len(str(t)) for t in texts]):.2f}")

    # Language distribution
    langs = [r.get('lang', 'unknown') for r in records if isinstance(r, dict)]
    lang_dist = Counter(langs)
    print(f"\nLanguage Distribution:")
    for lang, count in lang_dist.most_common():
        percentage = (count / len(langs)) * 100
        print(f"  {lang}: {count} ({percentage:.2f}%)")

    # 3. ANNOTATOR INFORMATION
    print(f"\n3. ANNOTATOR INFORMATION")
    print("-" * 80)

    num_annotators = [r.get('number_annotators', 0) for r in records if isinstance(r, dict)]
    print(f"\nNumber of Annotators per Record:")
    print(f"  Mean: {np.mean(num_annotators):.2f}")
    print(f"  Median: {np.median(num_annotators):.2f}")
    print(f"  Min: {min(num_annotators)}")
    print(f"  Max: {max(num_annotators)}")

    # Annotator demographics
    genders = []
    ages = []
    ethnicities = []
    for r in records:
        if isinstance(r, dict):
            genders.extend(r.get('gender_annotators', []))
            ages.extend(r.get('age_annotators', []))
            ethnicities.extend(r.get('ethnicities_annotators', []))

    print(f"\nAnnotator Gender Distribution (total: {len(genders)}):")
    for gender, count in Counter(genders).most_common():
        percentage = (count / len(genders)) * 100
        print(f"  {gender}: {count} ({percentage:.2f}%)")

    print(f"\nAnnotator Age Distribution (total: {len(ages)}):")
    for age, count in Counter(ages).most_common():
        percentage = (count / len(ages)) * 100
        print(f"  {age}: {count} ({percentage:.2f}%)")

    print(f"\nAnnotator Ethnicity Distribution (top 10, total: {len(ethnicities)}):")
    for ethnicity, count in Counter(ethnicities).most_common(10):
        percentage = (count / len(ethnicities)) * 100
        print(f"  {ethnicity}: {count} ({percentage:.2f}%)")

    # 4. TASK LABELS ANALYSIS
    print(f"\n4. TASK LABELS ANALYSIS")
    print("-" * 80)

    # Task 2.1 - Binary classification (YES/NO)
    task2_1_labels = []
    for r in records:
        if isinstance(r, dict):
            task2_1_labels.extend(r.get('labels_task2_1', []))

    print(f"\nTask 2.1 - Sexism Classification (binary - total: {len(task2_1_labels)}):")
    for label, count in Counter(task2_1_labels).most_common():
        percentage = (count / len(task2_1_labels)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")

    # Task 2.2 - Sexism type
    task2_2_labels = []
    for r in records:
        if isinstance(r, dict):
            task2_2_labels.extend(r.get('labels_task2_2', []))

    print(f"\nTask 2.2 - Sexism Type (total: {len(task2_2_labels)}):")
    for label, count in Counter(task2_2_labels).most_common():
        percentage = (count / len(task2_2_labels)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")

    # Task 2.3 - Target identification (can have multiple targets)
    task2_3_labels = []
    for r in records:
        if isinstance(r, dict):
            targets = r.get('labels_task2_3', [])
            for target_list in targets:
                if isinstance(target_list, list):
                    task2_3_labels.extend(target_list)
                else:
                    task2_3_labels.append(target_list)

    print(f"\nTask 2.3 - Target Groups (total: {len(task2_3_labels)}):")
    for label, count in Counter(task2_3_labels).most_common(10):
        percentage = (count / len(task2_3_labels)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")

    # 5. SENSORIAL DATA AVAILABILITY
    print(f"\n5. SENSORIAL DATA AVAILABILITY")
    print("-" * 80)

    sensorial_available = {
        'ET (Eye Tracking)': 0,
        'HR (Heart Rate)': 0,
        'EEG': 0
    }

    for r in records:
        if isinstance(r, dict) and 'sensorial' in r:
            sensorial = r['sensorial']
            modalities = sensorial.get('modalities', {})
            if 'ET' in modalities:
                sensorial_available['ET (Eye Tracking)'] += 1
            if 'HR' in modalities:
                sensorial_available['HR (Heart Rate)'] += 1
            if 'EEG' in modalities:
                sensorial_available['EEG'] += 1

    print(f"Records with sensorial data (total: {len(records)}):")
    for modality, count in sensorial_available.items():
        percentage = (count / len(records)) * 100
        print(f"  {modality}: {count} ({percentage:.2f}%)")

    # 6. DATASET SPLIT
    print(f"\n6. DATASET SPLIT")
    print("-" * 80)

    splits = [r.get('split', 'unknown') for r in records if isinstance(r, dict)]
    split_dist = Counter(splits)
    print(f"Dataset split distribution:")
    for split, count in split_dist.most_common():
        percentage = (count / len(splits)) * 100
        print(f"  {split}: {count} ({percentage:.2f}%)")

    # 7. MEME FILES
    print(f"\n7. MEME FILES")
    print("-" * 80)

    meme_files = [r.get('meme') for r in records if isinstance(r, dict)]
    print(f"Total meme files: {len([m for m in meme_files if m])}")
    print(f"Sample meme files: {meme_files[:5]}")

    # 8. DATA QUALITY CHECK
    print(f"\n8. DATA QUALITY CHECK")
    print("-" * 80)

    missing_counts = {
        'text': 0,
        'meme': 0,
        'labels_task2_1': 0,
        'labels_task2_2': 0,
        'labels_task2_3': 0,
        'sensorial': 0,
    }

    for r in records:
        if isinstance(r, dict):
            if not r.get('text'):
                missing_counts['text'] += 1
            if not r.get('meme'):
                missing_counts['meme'] += 1
            if not r.get('labels_task2_1'):
                missing_counts['labels_task2_1'] += 1
            if not r.get('labels_task2_2'):
                missing_counts['labels_task2_2'] += 1
            if not r.get('labels_task2_3'):
                missing_counts['labels_task2_3'] += 1
            if 'sensorial' not in r:
                missing_counts['sensorial'] += 1

    print(f"Missing values in key fields:")
    for field, count in missing_counts.items():
        percentage = (count / len(records)) * 100
        print(f"  {field}: {count} ({percentage:.2f}%)")

    # 9. LABEL CONSISTENCY
    print(f"\n9. LABEL CONSISTENCY")
    print("-" * 80)

    # Check annotation agreement
    print(f"\nAnnotation array sizes (should all have same length for each record):")
    for i, r in enumerate(records[:3]):
        if isinstance(r, dict):
            t2_1_len = len(r.get('labels_task2_1', []))
            t2_2_len = len(r.get('labels_task2_2', []))
            t2_3_len = len(r.get('labels_task2_3', []))
            n_annotators = r.get('number_annotators', 0)
            print(f"  Record {i}: task2_1={t2_1_len}, task2_2={t2_2_len}, task2_3={t2_3_len}, num_annotators={n_annotators}")

    # 10. SUMMARY STATISTICS
    print(f"\n10. SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total records: {len(records)}")
    print(f"Memory usage: {sys.getsizeof(data) / (1024**2):.2f} MB")
    print(f"Unique languages: {len(lang_dist)}")
    print(f"Unique splits: {len(split_dist)}")
    print(f"Average annotators per record: {np.mean(num_annotators):.2f}")
    print(f"Average text length: {np.mean(text_lengths):.0f} words")

if __name__ == "__main__":
    filepath = "/Users/aryan/Desktop/aryan_Exist_2026_dataset/EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/training/EXIST2026_training.json"

    try:
        data = load_json_data(filepath)
        analyze_dataset(data)
        print("\n" + "="*80)
        print("EDA Complete!")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
