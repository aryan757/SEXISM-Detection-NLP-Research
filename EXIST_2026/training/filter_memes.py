import json

def process_file(input_file, output_file):
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")

    keys_to_keep = [
        "meme", "path_memes", "text", "number_annotators", "annotators", 
        "gender_annotators", "age_annotators", "ethnicities_annotators", 
        "study_levels_annotators", "countries_annotators", 
        "labels_task2_1", "labels_task2_2", "labels_task2_3"
    ]

    filtered_data = {}
    for key, item in data.items():
        if item.get("lang") == "en":
            filtered_item = {}
            for k in keys_to_keep:
                if k in item:
                    filtered_item[k] = item[k]
            filtered_data[key] = filtered_item
        if item.get("lang") == "es":
            filtered_item = {}
            for k in keys_to_keep:
                if k in item:
                    filtered_item[k] = item[k]
            filtered_data[key] = filtered_item

        else:
            continue

    print(f"Filtered down to {len(filtered_data)} English items.")

    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    process_file('/Users/aryan/Desktop/aryan_Exist_2026_dataset/EXIST_2026_TESTING/EXIST_2026_Memes_Dataset/test/EXIST2026_test_clean.json', 'EXIST2026_testing_english_spanish_filtered.json')
