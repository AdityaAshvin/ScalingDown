# data_preprocessing_bert_phrasebank.py

import os
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split


def get_preprocessed_data(save_dir=''):
    train_file = os.path.join(save_dir, 'train_bert_phrasebank.pkl')
    val_file = os.path.join(save_dir, 'val_bert_phrasebank.pkl')
    test_file = os.path.join(save_dir, 'test_bert_phrasebank.pkl')

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Loading preprocessed data...")
        with open(train_file, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(val_file, 'rb') as f:
            val_dataset = pickle.load(f)
        with open(test_file, 'rb') as f:
            test_dataset = pickle.load(f)
        return train_dataset, val_dataset, test_dataset
    else:
        print("Preprocessing data...")
        train_dataset, val_dataset, test_dataset = preprocess_data(save_dir)
        return train_dataset, val_dataset, test_dataset


def preprocess_data(save_dir):
    # Load the enhanced Financial PhraseBank dataset
    ds = load_dataset("descartes100/enhanced-financial-phrasebank")

    # Extract the data from the nested 'train' column
    data = ds['train']['train']  # This is a list of dictionaries

    # Now, extract sentences and labels
    sentences = [item['sentence'] for item in data]
    labels = [item['label'] for item in data]

    # Map numerical labels to integers (0, 1, 2)
    label_mapping = {0: 1, 1: 2, 2: 0}  # negative -> 1, neutral -> 2, positive -> 0
    labels = [label_mapping[label] for label in labels]

    # Split data into train, validation, and test sets
    # 20% test, 20% of remaining for validation
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Prepare and tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            max_length=256,
            truncation=True,
            padding='max_length'
        )

    datasets = {}
    for split_name, (X, y) in zip(['train', 'validation', 'test'],
                                  [(X_train, y_train), (X_val, y_val), (X_test, y_test)]):
        examples = {'text': X, 'labels': y}
        dataset = Dataset.from_dict(examples)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['text'])
        datasets[split_name] = tokenized_dataset

    # Save preprocessed data to files
    train_file = os.path.join(save_dir, 'train_bert_phrasebank.pkl')
    val_file = os.path.join(save_dir, 'val_bert_phrasebank.pkl')
    test_file = os.path.join(save_dir, 'test_bert_phrasebank.pkl')
    with open(train_file, 'wb') as f:
        pickle.dump(datasets['train'], f)
    with open(val_file, 'wb') as f:
        pickle.dump(datasets['validation'], f)
    with open(test_file, 'wb') as f:
        pickle.dump(datasets['test'], f)

    print(f"Preprocessed data saved to {train_file}, {val_file}, and {test_file}")
    return datasets['train'], datasets['validation'], datasets['test']
