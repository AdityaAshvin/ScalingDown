from datasets import load_dataset
from transformers import T5Tokenizer, AutoTokenizer
import pandas as pd
import os
import torch

# Load tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')  # Correct model name
# Add pad_token if not present
if t5_tokenizer.pad_token is None:
    t5_tokenizer.add_special_tokens({'pad_token': '<pad>'})
t5_tokenizer.save_pretrained('tokenizers/t5_tokenizer/')

def load_aqua_data():
    # using hugging face
    print("Loading AQuA-Rat dataset...")
    dataset = load_dataset("deepmind/aqua_rat", "raw")

    # ref: https://huggingface.co/datasets/deepmind/aqua_rat
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    val_df = pd.DataFrame(dataset['validation'])

    return train_df, test_df, val_df

# Function to preprocess data
def preprocess_data_for_models(df, max_length=512):
    student_inputs = []
    student_labels = []

    for index, row in df.iterrows():
        # Prepare the input text
        input_text = (
            f"Question: {row['question']}\n"
            f"Options: {', '.join(row['options'])}\n"
            f"Please provide your answer and rationale."
        )

        # Prepare the target text
        target_text = (
            f"Answer: {row['correct']}\n"
            f"Rationale: {row['rationale']}"
        )

        # Tokenize the input
        input_encoding = t5_tokenizer(
            input_text, 
            return_tensors='pt', 
            padding='max_length',
            truncation=True, 
            max_length=max_length
        )

        # Tokenize the target
        target_encoding = t5_tokenizer(
            target_text, 
            return_tensors='pt', 
            padding='max_length',
            truncation=True, 
            max_length=max_length
        )

        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels = target_encoding['input_ids']
        labels[labels == t5_tokenizer.pad_token_id] = -100

        # Append input_ids and attention_mask to student_inputs
        student_inputs.append({
            'input_ids': input_encoding['input_ids'].squeeze(),         # Shape: (max_length,)
            'attention_mask': input_encoding['attention_mask'].squeeze()  # Shape: (max_length,)
        })

        # Append labels to student_labels
        student_labels.append({
            'labels': labels.squeeze()  # Shape: (max_length,)
        })

    return student_inputs, student_labels

def get_preprocessed_data(save_path='dataset_trainer_preprocessed_data.pkl'):    
    if os.path.exists(save_path):
        print(f"Loading preprocessed data from {save_path}...")
        with open(save_path, 'rb') as f:
            preprocessed_data = torch.load(f)
        return preprocessed_data
    else:
        train_df, test_df, val_df = load_aqua_data()

        print("Tokenizing training data...")
        train_student_inputs, train_student_labels = preprocess_data_for_models(train_df)
        print("Tokenizing testing data...")
        test_student_inputs, test_student_labels = preprocess_data_for_models(test_df)
        print("Tokenizing validating data...")
        val_student_inputs, val_student_labels = preprocess_data_for_models(val_df)

        preprocessed_data = {
            'train': {
                'input_ids': torch.stack([item['input_ids'] for item in train_student_inputs]),
                'attention_mask': torch.stack([item['attention_mask'] for item in train_student_inputs]),
                'labels': torch.stack([item['labels'] for item in train_student_labels]),
            },
            'test': {
                'input_ids': torch.stack([item['input_ids'] for item in test_student_inputs]),
                'attention_mask': torch.stack([item['attention_mask'] for item in test_student_inputs]),
                'labels': torch.stack([item['labels'] for item in test_student_labels]),
            },
            'validation': {
                'input_ids': torch.stack([item['input_ids'] for item in val_student_inputs]),
                'attention_mask': torch.stack([item['attention_mask'] for item in val_student_inputs]),
                'labels': torch.stack([item['labels'] for item in val_student_labels]),
            }
        }

        # Save preprocessed data to disk
        torch.save(preprocessed_data, save_path)
        print(f"Preprocessed data saved to {save_path}")

        return preprocessed_data

if __name__ == '__main__':
    preprocessed_data = get_preprocessed_data()
