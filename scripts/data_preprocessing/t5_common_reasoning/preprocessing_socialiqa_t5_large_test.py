# check_socialiqa.py

import logging
from datasets import load_dataset
import yaml
import os
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config/config.yaml'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def check_socialiqa_dataset(dataset):
    issues_found = False
    logging.info("Checking dataset for issues...")

    for i, example in enumerate(dataset):
        # Check for missing fields
        required_fields = ["context", "question", "answerA", "answerB", "answerC", "label"]
        for field in required_fields:
            if example.get(field) is None:
                logging.warning(f"Missing field '{field}' in example {i}")
                issues_found = True
                continue
        
        # Check for NaN values
        if any(pd.isna(example.get(field, "")) for field in required_fields[:-1]):
            logging.warning(f"NaN values found in example {i}: {example}")
            issues_found = True

        # Check label validity (should be 1, 2, or 3)
        try:
            label = int(example["label"])
            if label not in [1, 2, 3]:
                logging.warning(f"Invalid label '{label}' in example {i}. Expected values: 1, 2, or 3.")
                issues_found = True
        except ValueError:
            logging.warning(f"Non-integer label '{example['label']}' in example {i}")
            issues_found = True

        # Check if the length of input string is reasonable
        input_str = (
            f"Context: {example['context']}\n"
            f"Question: {example['question']}\n"
            f"Choose the correct option:\n"
            f"1: {example['answerA']}\n2: {example['answerB']}\n3: {example['answerC']}\nAnswer:"
        )
        if len(input_str) < 5:
            logging.warning(f"Very short input detected in example {i}: '{input_str}'")
            issues_found = True

    if not issues_found:
        logging.info("No issues found in the dataset!")
    else:
        logging.info("Dataset check completed with issues found.")

    # Print 200 sample questions
    print("\nSample Questions:")
    for i, example in enumerate(dataset.select(range(200))):
        print(f"Example {i + 1}:")
        print(f"Context: {example['context']}")
        print(f"Question: {example['question']}")
        print(f"Answer 1: {example['answerA']}")
        print(f"Answer 2: {example['answerB']}")
        print(f"Answer 3: {example['answerC']}")
        print(f"Label: {example['label']}\n")
        print("-" * 50)

def main():
    config = load_config()
    dataset_name = config["datasets"]["socialiqa"]["name"]

    logging.info("Loading SocialIQA dataset...")
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        logging.error(f"Error loading SocialIQA dataset: {e}")
        return

    check_socialiqa_dataset(dataset)

if __name__ == "__main__":
    main()
