# data_preprocessing_bert_phrasebank.py

import os
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

def get_preprocessed_data(save_dir='', teacher_model_names=None):
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
        train_dataset, val_dataset, test_dataset = preprocess_data(save_dir,teacher_model_names)
        return train_dataset, val_dataset, test_dataset

def preprocess_data(save_dir, teacher_model_names):
    # Load the enhanced Financial PhraseBank dataset
    ds = load_dataset("descartes100/enhanced-financial-phrasebank")

    # Extract the data from the nested 'train' column
    data = ds['train']['train']

    # Extract sentences and labels
    sentences = [item['sentence'] for item in data]
    labels = [item['label'] for item in data]

    # Map labels to match model expectations
    label_mapping = {0: 1, 1: 2, 2: 0}  # negative -> 1, neutral -> 2, positive -> 0
    labels = [label_mapping[label] for label in labels]

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=566, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=566, stratify=y_train_val
    )

    # Initialize student tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Initialize teacher tokenizers
    teacher_tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in teacher_model_names}

    # Prepare and tokenize datasets
    def tokenize_function(examples):
        tokenized_inputs = student_tokenizer(
            examples['text'],
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        tokenized_inputs['labels'] = examples['labels']

        # Tokenize with each teacher's tokenizer
        for name, tokenizer in teacher_tokenizers.items():
            teacher_tokenized = tokenizer(
                examples['text'],
                max_length=256,
                truncation=True,
                padding='max_length'
            )
            # Prefix keys to avoid collision
            tokenized_inputs[f'{name}_input_ids'] = teacher_tokenized['input_ids']
            tokenized_inputs[f'{name}_attention_mask'] = teacher_tokenized['attention_mask']

        return tokenized_inputs

    datasets = {}
    for split_name, (X, y) in zip(['train', 'validation', 'test'],
                                  [(X_train, y_train), (X_val, y_val), (X_test, y_test)]):
        examples = {'text': X, 'labels': y}
        dataset = Dataset.from_dict(examples)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        print(f"Columns in {split_name} dataset:", tokenized_dataset.column_names)
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