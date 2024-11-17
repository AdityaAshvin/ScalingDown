# preprocess_socialiqa.py

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
    tokenizer_path = os.path.join(config["tokenizer"]["save_dir"])
    if not os.path.exists(tokenizer_path):
        logging.info(f"Tokenizer directory {tokenizer_path} does not exist. Creating it.")
        os.makedirs(config["tokenizer"]["save_dir"], exist_ok=True)
    logging.info(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    except:
        logging.info("Tokenizer not found locally, downloading from model hub...")
        tokenizer = T5Tokenizer.from_pretrained(config["tokenizer"]["name"])
        tokenizer.save_pretrained(tokenizer_path)  # Ensure tokenizer is saved locally after download
    return tokenizer

def preprocess_socialiqa(examples, tokenizer, config):
    inputs, targets = [], []

    for context, question, answerA, answerB, answerC, label in zip_longest(
        examples.get("context", []),
        examples.get("question", []),
        examples.get("answerA", []),
        examples.get("answerB", []),
        examples.get("answerC", []),        
        examples.get("label", []),
        fillvalue=None
    ):
        # Convert label to an integer, if possible
        try:
            label = int(label)
        except ValueError:
            logging.error(f"Label {label} could not be converted to an integer.")
            continue
        
        # Check if label is 1 or 2 or 3
        if context and question and answerA and answerB and answerC and label in [1, 2, 3]:
            input_str = (
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Choose the correct option:\n"
                f"1: {answerA}\n2: {answerB}\n3: {answerC}\nAnswer:"
            )
            inputs.append(input_str)
            targets.append(str(label))  # Use label directly as a string for the target
        else:
            logging.warning(f"Skipping invalid example with context: {context}, question: {question}, label: {label}")

    if not inputs:
        logging.error("No valid inputs found. Check the dataset structure.")
        return {"input_ids": [], "labels": []}  # Provide default empty keys to avoid KeyError

    # Tokenize inputs and targets
    model_inputs = tokenizer(
        inputs,
        padding=config["preprocessing"]["padding"],
        truncation=config["preprocessing"]["truncation"],
        max_length=config["preprocessing"]["max_length"],
        return_tensors="pt"
    )

    labels = tokenizer(
        text_target=targets,
        padding=config["preprocessing"]["padding"],
        truncation=config["preprocessing"]["truncation"],
        max_length=config["preprocessing"]["max_length_labels"],
        return_tensors="pt",
    )
    
    if labels["input_ids"].nelement() > 0:
        labels["input_ids"] = torch.nan_to_num(labels["input_ids"], nan=tokenizer.pad_token_id)
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
    
    model_inputs["labels"] = labels["input_ids"]

    logging.info(f"Sample input_ids: {model_inputs.get('input_ids', 'Not Found')}")
    logging.info(f"Sample labels: {labels.get('input_ids', 'Not Found')}")

    return model_inputs



def save_tokenized_dataset(dataset, filename, config):
    try:
        dataset.save_to_disk(filename)
        logging.info(f"Successfully saved tokenized SocialIQA dataset to {filename}.")
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

    dataset_name = config["datasets"]["socialiqa"]["name"]
    dataset_path = config["datasets"]["socialiqa"]["path"]

    logging.info("Loading SocialIQA dataset...")
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        logging.error(f"Error loading SocialIQA dataset: {e}")
        return

    logging.info("Preprocessing SocialIQA dataset...")
    try:
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_socialiqa(examples, tokenizer, config),
            batched=True,
            num_proc=config["preprocessing"].get("num_proc", 1),
            remove_columns=dataset.column_names
        )
        # Check if 'input_ids' is present in the processed dataset
        if "input_ids" not in tokenized_dataset.column_names:
            logging.error("input_ids not found in tokenized dataset. Check preprocessing function.")
            return
        
    except ValueError as e:
        logging.error(f"ValueError during SocialIQA preprocessing: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during SocialIQA preprocessing: {e}")
        return

    print("Sample processed data:")
    for i in range(10, 15):
        sample = tokenized_dataset[i]
        decoded_input = tokenizer.decode(
            sample['input_ids'],
            skip_special_tokens=True
        )
        labels_tensor = sample['labels']
        if not isinstance(labels_tensor, torch.Tensor):
            labels_tensor = torch.tensor(labels_tensor)

        decoded_label = tokenizer.decode(
            torch.where(labels_tensor == -100, torch.tensor(tokenizer.pad_token_id), labels_tensor),
            skip_special_tokens=True
        ).strip()


        print(f"Sample {i + 1}:")
        print("Input Text:", decoded_input)
        print("Label Text:", decoded_label)
        print("-" * 50)

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
        print(f"\nSample SocialIQA Input: {input_text}")
        print(f"Sample SocialIQA Label: {label_text}\n")

if __name__ == "__main__":
    main()