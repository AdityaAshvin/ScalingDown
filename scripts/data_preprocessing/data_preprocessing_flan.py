# data_preprocessing_flan.py

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
        for question, options in zip(examples['question'], examples['options']):
            # Check if options are already labeled by inspecting the first option
            if options[0].startswith(('A)', 'B)', 'C)', 'D)', 'E)')):
                # Options are already labeled; join them with newline
                options_str = '\n'.join(options)
            else:
                # Label the options with A, B, C, D, E
                option_labels = ['A', 'B', 'C', 'D', 'E']
                labeled_options = [f"{label}) {option}" for label, option in zip(option_labels, options)]
                options_str = '\n'.join(labeled_options)

            input_text = (
                f"Question: {question}\n"
                f"Options:\n{options_str}\n\n"
                "Please solve the problem and provide the correct option along with a detailed step-by-step rationale."
                "\n\nRationale:\nAnswer:"
            )
            inputs.append(input_text)

        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        # Keep the 'correct' and 'rationale' fields
        model_inputs['correct'] = examples['correct']
        model_inputs['rationale'] = examples['rationale']
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