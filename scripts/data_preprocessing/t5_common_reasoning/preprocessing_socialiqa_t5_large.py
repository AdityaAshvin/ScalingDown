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
        print(f"Tokenizer with special tokens saved to {tokenizer_path}")

    vocab_size = len(tokenizer)
    print(f"Tokenizer Vocabulary Size: {vocab_size}")
    print("Token IDs for '1':", tokenizer.encode('<1>'))
    print("Token IDs for '2':", tokenizer.encode('<2>'))
    print("Token IDs for '3':", tokenizer.encode('<3>'))

    # Add additional special tokens for labels
    special_tokens = {'additional_special_tokens': ['<1>', '<2>', '<3>']}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    if num_added_tokens > 0:
        print(f"Added {num_added_tokens} special tokens: {special_tokens['additional_special_tokens']}")
    else:
        print("No new special tokens were added.")
    
    tokenizer.save_pretrained(tokenizer_path)  # Ensure tokenizer is saved locally after download
    return tokenizer

def preprocess_socialiqa(examples, tokenizer, config):
    inputs, targets = [], []
    input_texts, label_texts = [], []  # New lists to store original texts

    num_examples = len(examples["context"])

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

        # Adjust label to be in [1, 2, 3]
        #label += 1  # Original labels are 0-based; adjust to 1-based
        
        # Check if label is 1 or 2 or 3
        if context and question and answerA and answerB and answerC and label in [1, 2, 3]:
            label_token = f"<{label}>"
            input_str = (
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Choose the correct option:\n"
                f"1: {answerA}\n2: {answerB}\n3: {answerC}\nAnswer:"
            )
            inputs.append(input_str)
            targets.append(label_token)  # Use label directly as a string for the target

            input_texts.append(input_str)
            label_texts.append(label_token)
        else:
            logging.warning(f"Skipping invalid example with context: {context}, question: {question}, label: {label}")

    if not inputs:
        logging.error("No valid inputs found. Check the dataset structure.")
        return {"input_ids": [], "labels": [], "input_text": [], "label_text": []}  # Provide default empty keys to avoid KeyError

    # Tokenize inputs and targets
    model_inputs = tokenizer(
        inputs,
        padding=config["preprocessing"]["padding"],
        truncation=config["preprocessing"]["truncation"],
        max_length=config["preprocessing"]["max_length"],
    )

    labels = tokenizer(
        text_target=targets,
        padding=config["preprocessing"]["padding"],
        truncation=config["preprocessing"]["truncation"],
        max_length=config["preprocessing"]["max_length_labels"],
    )
    
    if labels["input_ids"]:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["input_text"] = input_texts
    model_inputs["label_text"] = label_texts

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
        dataset = load_dataset(dataset_name)
    except Exception as e:
        logging.error(f"Error loading SocialIQA dataset: {e}")
        return

    splits = dataset.keys()
    logging.info(f"Available splits in SocialIQA: {splits}")

    for split in splits:
        logging.info(f"Preprocessing {split} split...")
        try:
            split_dataset = dataset[split]
            tokenized_dataset = split_dataset.map(
                lambda examples: preprocess_socialiqa(examples, tokenizer, config),
                batched=True,
                num_proc=config["preprocessing"].get("num_proc", 1),
                remove_columns=split_dataset.column_names
            )
            # Check if 'input_ids' is present in the processed dataset
            if "input_ids" not in tokenized_dataset.column_names:
                logging.error(f"input_ids not found in tokenized {split} dataset. Check preprocessing function.")
                continue
            
            # Set format for PyTorch
            tokenized_dataset.set_format(type='torch')
            
            # Define the path for the split
            split_path = os.path.join(dataset_path, split)
            os.makedirs(split_path, exist_ok=True)  # Ensure the directory exists
            save_tokenized_dataset(tokenized_dataset, os.path.join(split_path, "dataset.pt"), config)

        
        except ValueError as e:
            logging.error(f"ValueError during SocialIQA preprocessing for split {split}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during SocialIQA preprocessing for split {split}: {e}")
            continue

        print(f"Sample processed data for {split} split:")
        for i in range(10, 15):
            try:
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
            except IndexError:
                print(f"Sample index {i + 1} out of range for {split} split.")
                break

    logging.info("Preprocessing completed for all splits.")

if __name__ == "__main__":
    main()
