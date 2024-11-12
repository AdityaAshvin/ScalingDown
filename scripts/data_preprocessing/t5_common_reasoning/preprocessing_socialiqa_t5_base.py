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
    tokenizer_path = os.path.join(config["tokenizer"]["save_dir"], config["tokenizer"]["name"])
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer directory {tokenizer_path} does not exist.")
        raise FileNotFoundError(f"Tokenizer directory {tokenizer_path} does not exist.")
    logging.info(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    except:
        logging.info("Tokenizer not found locally, downloading from model hub...")
        tokenizer = T5Tokenizer.from_pretrained(config["tokenizer"]["name"])
    return tokenizer

def preprocess_socialiqa(examples, tokenizer, config):
    inputs, targets = [], []

    for context, question, answerA, answerB, label in zip_longest(
        examples.get("context", []),
        examples.get("question", []),
        examples.get("answerA", []),
        examples.get("answerB", []),
        examples.get("label", []),
        fillvalue=None
    ):
        # Convert label to an integer, if possible
        try:
            label = int(label)
        except ValueError:
            logging.error(f"Label {label} could not be converted to an integer.")
            continue
        
        # Check if label is 1 or 2
        if context and question and answerA and answerB and label in [1, 2]:
            input_str = f"Context: {context}\nQuestion: {question}\nChoose the correct option:\n1: {answerA}\n2: {answerB}\nAnswer:"
            inputs.append(input_str)
            targets.append(str(label))  # Use label directly as a string for the target

    if not inputs:
        logging.error("No inputs were processed. Check dataset structure and conditions.")
        return {}  # Return empty dict if inputs is empty to avoid tokenization error

    # Tokenize inputs and targets
    model_inputs = tokenizer(
        inputs,
        padding=config["preprocessing"]["padding"],
        truncation=config["preprocessing"]["truncation"],
        max_length=config["preprocessing"]["max_length"],
        return_tensors="pt"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding=config["preprocessing"]["padding"],
            truncation=config["preprocessing"]["truncation"],
            max_length=config["preprocessing"]["max_length_labels"],
            return_tensors="pt"
        )
    
    if labels["input_ids"].nelement() > 0:
        labels["input_ids"] = torch.nan_to_num(labels["input_ids"], nan=tokenizer.pad_token_id)
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



def save_tokenized_dataset(dataset, filename, config):
    dataset_dir = os.path.dirname(filename)
    os.makedirs(dataset_dir, exist_ok=True)
    
    try:
        torch.save(dataset, filename)
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

    dataset_name = "allenai/social_i_qa"
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
    except ValueError as e:
        logging.error(f"ValueError during SocialIQA preprocessing: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during SocialIQA preprocessing: {e}")
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
        print(f"\nSample SocialIQA Input: {input_text}")
        print(f"Sample SocialIQA Label: {label_text}\n")

if __name__ == "__main__":
    main()