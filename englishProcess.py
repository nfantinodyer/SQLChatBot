import os
import json
import re
import glob
import requests
import time
from bs4 import BeautifulSoup

def main():
    json_file = 'trainingData.json'
    # Process Wikipedia Dumps
    process_wikipedia_dump('TrainingMaterials/extracted_wikipedia', json_file)
    # Process Project Gutenberg Texts
    process_gutenberg_texts('TrainingMaterials/gutenberg_texts', json_file)
    # Process OpenWebText Dataset
    process_openwebtext('TrainingMaterials/openwebtext', json_file)
    # Optionally, process other datasets similarly

def process_wikipedia_dump(dump_dir, json_file_path):
    data = []
    for root, dirs, files in os.walk(dump_dir):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    articles = text.split('\n\n')
                    for article in articles:
                        cleaned_text = clean_text(article)
                        if cleaned_text:
                            prompt = "Provide detailed information on the following topic:"
                            data.append({"prompt": prompt, "response": cleaned_text})
    append_data_to_json(data, json_file_path)

def process_gutenberg_texts(texts_dir, json_file_path):
    data = []
    for file_path in glob.glob(f"{texts_dir}/*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            cleaned_text = clean_gutenberg_text(text)
            if cleaned_text:
                prompt = "Summarize the following text:"
                data.append({"prompt": prompt, "response": cleaned_text})
    append_data_to_json(data, json_file_path)

def process_openwebtext(data_dir, json_file_path):
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        cleaned_text = clean_text(line)
                        if cleaned_text:
                            prompt = "Continue the following passage:"
                            data.append({"prompt": prompt, "response": cleaned_text})
    append_data_to_json(data, json_file_path)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 50:
        return None
    return text

def clean_gutenberg_text(text):
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1 and end != -1:
        text = text[start:end]
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 50:
        return None
    return text

def append_data_to_json(new_data, json_file_path):
    try:
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        else:
            data = []
        data.extend(new_data)
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"Appended {len(new_data)} records to {json_file_path}")
    except Exception as e:
        print(f"Error appending data to JSON file: {str(e)}")

if __name__ == "__main__":
    main()
