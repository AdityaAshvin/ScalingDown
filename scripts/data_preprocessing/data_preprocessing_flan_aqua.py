import os
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def get_preprocessed_data(save_dir=''):
    train_file = os.path.join(save_dir, 'train_flan_aqua.pkl')
    val_file = os.path.join(save_dir, 'val_flan_aqua.pkl')

    if os.path.exists(train_file) and os.path.exists(val_file):
        print("Loading preprocessed data...")
        with open(train_file, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(val_file, 'rb') as f:
            val_dataset = pickle.load(f)
        return train_dataset, val_dataset
    else:
        print("Preprocessing data...")
        train_dataset, val_dataset = preprocess_data(save_dir)
        return train_dataset, val_dataset

# data_preprocessing_flan.py

def preprocess_data(save_dir):
    # Load the AQuA dataset
    ds = load_dataset("deepmind/aqua_rat", "raw")

    # Split into training and validation datasets
    train_data_raw = ds['train']
    val_data_raw = ds['validation']

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    # Preprocess function
    def preprocess_function(examples):
        inputs = []
        labels = []
        for question, options, correct in zip(examples['question'], examples['options'], examples['correct']):
            # Prepare options string
            if options[0].startswith(('A)', 'B)', 'C)', 'D)', 'E)')):
                options_str = '\n'.join(options)
            else:
                option_labels = ['A', 'B', 'C', 'D', 'E']
                labeled_options = [f"{label}) {option}" for label, option in zip(option_labels, options)]
                options_str = '\n'.join(labeled_options)

            input_text = (
                f"Question: {question}\n"
                f"Options:\n{options_str}\n\n"
                "Please select the correct option."
            )
            inputs.append(input_text)
            labels.append(correct)

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        # Tokenize labels
        tokenized_labels = tokenizer(
            labels,
            max_length=5,
            truncation=True,
            padding='max_length'
        )['input_ids']

        # Replace padding token id's with -100 to ignore in loss computation
        tokenized_labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in tokenized_labels]

        model_inputs['labels'] = tokenized_labels
        return model_inputs

    # Apply preprocessing to the datasets
    print("Tokenizing training data...")
    train_dataset = train_data_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=[]  # Do not remove any columns to retain 'correct' and 'rationale'
    )

    print("Tokenizing validation data...")
    val_dataset = val_data_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=[]  # Do not remove any columns to retain 'correct' and 'rationale'
    )

    # Save preprocessed data to files
    train_file = os.path.join(save_dir, 'train_flan_aqua.pkl')
    val_file = os.path.join(save_dir, 'val_flan_aqua.pkl')
    with open(train_file, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(val_file, 'wb') as f:
        pickle.dump(val_dataset, f)

    print(f"Preprocessed data saved to {train_file} and {val_file}")
    return train_dataset, val_dataset