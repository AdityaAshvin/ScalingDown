# preprocessing.py

import torch
from transformers import T5Tokenizer
from datasets import load_dataset
import yaml
import os
import logging
from itertools import zip_longest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config/config.yaml'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_tokenizer(config):
    tokenizer_path = os.path.join(config["tokenizer"]["save_dir"], config["tokenizer"]["name"])
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer directory {tokenizer_path} does not exist.")
        raise FileNotFoundError(f"Tokenizer directory {tokenizer_path} does not exist.")
    logging.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def preprocess_dataset(examples, tokenizer, config, stage):
    inputs, targets = [], []

    if stage == 1:  # Winograd (pronoun disambiguation)
        for sentence, option1, option2, answer in zip_longest(
            examples.get("sentence", []),
            examples.get("option1", []),
            examples.get("option2", []),
            examples.get("answer", []),
            fillvalue=None
        ):
            if sentence and option1 and option2 and answer is not None:
                input_str = f"Sentence: {sentence}\nOption1: {option1}\nOption2: {option2}\nAnswer:"
                inputs.append(input_str)
                targets.append(str(answer))

    elif stage == 2:  # PIQA
        for goal, sol1, sol2, answer in zip_longest(
            examples.get("goal", []),
            examples.get("sol1", []),
            examples.get("sol2", []),
            examples.get("label", []),
            fillvalue=None
        ):
            if goal and sol1 and sol2 and answer is not None:
                input_str = f"Goal: {goal}\nOption1: {sol1}\nOption2: {sol2}\nAnswer:"
                inputs.append(input_str)
                targets.append(str(answer + 1))

    elif stage == 3:  # aNLI
        for context, hypothesis1, hypothesis2, label in zip_longest(
            examples.get("context", []),
            examples.get("hypothesis1", []),
            examples.get("hypothesis2", []),
            examples.get("label", []),
            fillvalue=None
        ):
            # Ensure no empty fields for aNLI examples
            if context and hypothesis1 and hypothesis2 and label is not None:
                input_str = f"Context: {context}\nOption1: {hypothesis1}\nOption2: {hypothesis2}\nAnswer:"
                inputs.append(input_str)
                targets.append(str(label + 1))

    # Tokenize the inputs and labels as before
    model_inputs = tokenizer(
        inputs,
        padding=config["preprocessing"]["padding"],
        truncation=config["preprocessing"]["truncation"],
        max_length=config["preprocessing"][f"max_length_stage{stage}"],
        return_tensors="pt"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding=config["preprocessing"]["padding"],
            truncation=config["preprocessing"]["truncation"],
            max_length=config["preprocessing"][f"max_length_labels_stage{stage}"],
            return_tensors="pt"
        )
    
    if labels["input_ids"].nelement() > 0:
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def save_tokenized_dataset(dataset, filename, config):
    dataset_dir = os.path.dirname(config["datasets"]["winograd"]["path"])
    os.makedirs(dataset_dir, exist_ok=True)
    
    filepath = os.path.join(dataset_dir, filename)
    try:
        torch.save(dataset, filepath)
        logging.info(f"Successfully saved tokenized dataset to {filepath}.")
    except (OSError, IOError) as e:
        logging.error(f"Error saving the dataset to {filepath}: {e}")

def main():
    config = load_config()
    os.makedirs(os.path.dirname(config["datasets"]["winograd"]["path"]), exist_ok=True)
    os.makedirs(config["tokenizer"]["save_dir"], exist_ok=True)
    
    try:
        tokenizer = load_tokenizer(config)
    except FileNotFoundError as e:
        logging.error(str(e))
        return
    
    # Define dataset names, paths, configurations, and splits
    dataset_names = ["winograd", "piqa", "anli"]
    dataset_paths = {
        "winograd": config["datasets"]["winograd"]["path"],
        "piqa": config["datasets"]["piqa"]["path"],
        "anli": config["datasets"]["anli"]["path"]
    }
    dataset_configs = {
        "winograd": config["datasets"]["winograd"]["config"],
        "piqa": None,
        "anli": None
    }
    dataset_splits = {
        "winograd": "train",
        "piqa": "train",
        "anli": "train_r1"  # Use 'train_r1' as the split name for aNLI
    }

    # Map datasets to stages
    stage_mapping = {
        "winograd": 1,
        "piqa": 2,
        "anli": 3
    }

    # Loop through datasets and process each one according to its stage
    for dataset_name, stage in stage_mapping.items():
        config_name = dataset_configs[dataset_name]
        split_name = dataset_splits[dataset_name]
        logging.info(f"Loading {dataset_name} dataset for Stage {stage}...")

        try:
            if config_name:
                dataset = load_dataset(config["datasets"][dataset_name]["name"], config_name, split=split_name)
            else:
                dataset = load_dataset(config["datasets"][dataset_name]["name"], split=split_name)
        except Exception as e:
            logging.error(f"Error loading {dataset_name} dataset: {e}")
            continue

        # Preprocess the dataset with the wrapper function
        def preprocess_wrapper(examples):
            return preprocess_dataset(examples, tokenizer, config, stage)

        logging.info(f"Preprocessing {dataset_name} dataset for Stage {stage}...")
        try:
            tokenized_dataset = dataset.map(
                preprocess_wrapper,
                batched=True,
                num_proc=config["preprocessing"].get("num_proc", 1),
                remove_columns=dataset.column_names
            )
        except Exception as e:
            logging.error(f"Error during {dataset_name} preprocessing: {e}")
            continue

        tokenized_dataset.set_format(type='torch')
        save_tokenized_dataset(tokenized_dataset, os.path.basename(dataset_paths[dataset_name]), config)

        # Print a sample for verification
        if len(tokenized_dataset) > 0:
            sample = tokenized_dataset[0]
            input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            label_text = tokenizer.decode(
                torch.where(sample['labels'] == -100, tokenizer.pad_token_id, sample['labels']),
                skip_special_tokens=True
            ).strip()
            print(f"\nSample {dataset_name.capitalize()} Input: {input_text}")
            print(f"Sample {dataset_name.capitalize()} Label: {label_text}\n")

if __name__ == "__main__":
    main()