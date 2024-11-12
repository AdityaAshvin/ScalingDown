# preprocess_winograd.py

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

def preprocess_winograd(examples, tokenizer, config):
    inputs, targets = [], []

    for sentence, option1, option2, answer in zip_longest(
        examples.get("sentence", []),
        examples.get("option1", []),
        examples.get("option2", []),
        examples.get("answer", []),
        fillvalue=None
    ):
        if sentence and option1 and option2 and answer is not None:
            input_str = f"Sentence: {sentence}\nChoose the correct option:\n1: {option1}\n2: {option2}\nAnswer:"
            inputs.append(input_str)
            targets.append(str(answer))

    model_inputs = tokenizer(
        inputs,
        padding=config["preprocessing"]["padding"],
        truncation=config["preprocessing"]["truncation"],
        max_length=config["preprocessing"]["max_length_stage1"],
        return_tensors="pt"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding=config["preprocessing"]["padding"],
            truncation=config["preprocessing"]["truncation"],
            max_length=config["preprocessing"]["max_length_labels_stage1"],
            return_tensors="pt"
        )
    
    if labels["input_ids"].nelement() > 0:
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def save_tokenized_dataset(dataset, filename, config):
    dataset_dir = os.path.dirname(filename)
    os.makedirs(dataset_dir, exist_ok=True)
    
    try:
        torch.save(dataset, filename)
        logging.info(f"Successfully saved tokenized Winograd dataset to {filename}.")
    except (OSError, IOError) as e:
        logging.error(f"Error saving the dataset to {filename}: {e}")

def main():
    config = load_config()
    os.makedirs(config["tokenizer"]["save_dir"], exist_ok=True)
    
    try:
        tokenizer = load_tokenizer(config)
    except FileNotFoundError as e:
        logging.error(str(e))
        return

    dataset_name = "winogrande"
    config_name = config["datasets"]["winograd"]["config"]
    dataset_path = config["datasets"]["winograd"]["path"]

    logging.info("Loading Winograd dataset...")
    try:
        dataset = load_dataset(dataset_name, config_name, split="train")
    except Exception as e:
        logging.error(f"Error loading Winograd dataset: {e}")
        return

    logging.info("Preprocessing Winograd dataset...")
    try:
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_winograd(examples, tokenizer, config),
            batched=True,
            num_proc=config["preprocessing"].get("num_proc", 1),
            remove_columns=dataset.column_names
        )
    except Exception as e:
        logging.error(f"Error during Winograd preprocessing: {e}")
        return

    tokenized_dataset.set_format(type='torch')
    save_tokenized_dataset(tokenized_dataset, dataset_path, config)

    # Print a sample for verification
    if len(tokenized_dataset) > 0:
        sample = tokenized_dataset[0]
        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        label_text = tokenizer.decode(
            torch.where(sample['labels'] == -100, tokenizer.pad_token_id, sample['labels']),
            skip_special_tokens=True
        ).strip()
        print(f"\nSample Winograd Input: {input_text}")
        print(f"Sample Winograd Label: {label_text}\n")

if __name__ == "__main__":
    main()
