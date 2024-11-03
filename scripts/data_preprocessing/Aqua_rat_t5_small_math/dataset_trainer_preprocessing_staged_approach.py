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
def preprocess_data_for_models(df, max_length=512, include_rationale=True):
    student_inputs = []
    student_labels = []

    for index, row in df.iterrows():
        # Prepare the input text
        input_text = (
            f"Question: {row['question']}\n"
            f"Options: {', '.join(row['options'])}\n"
        )
        
        # Adjust the prompt based on whether we want rationale
        if include_rationale:
            input_text += "Please provide your answer and rationale."
            target_text = (
                f"Answer: {row['correct']}\n"
                f"Rationale: {row['rationale']}"
            )
        else:
            input_text += "Please provide your answer."
            target_text = f"Answer: {row['correct']}"

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


def get_preprocessed_data(save_path_stage1='dataset_stage1.pkl', save_path_stage2='dataset_stage2.pkl'):
    if os.path.exists(save_path_stage1) and os.path.exists(save_path_stage2):
        print(f"Loading preprocessed data from {save_path_stage1} and {save_path_stage2}...")
        with open(save_path_stage1, 'rb') as f1, open(save_path_stage2, 'rb') as f2:
            preprocessed_data_stage1 = torch.load(f1)
            preprocessed_data_stage2 = torch.load(f2)
        return preprocessed_data_stage1, preprocessed_data_stage2
    else:
        train_df, test_df, val_df = load_aqua_data()

        print("Tokenizing training data for Stage 1...")
        train_inputs_stage1, train_labels_stage1 = preprocess_data_for_models(train_df, include_rationale=False)
        print("Tokenizing training data for Stage 2...")
        train_inputs_stage2, train_labels_stage2 = preprocess_data_for_models(train_df, include_rationale=True)

        # Do the same for test and validation data if needed
        # For simplicity, we'll focus on the training data here

        # Prepare and save Stage 1 data
        preprocessed_data_stage1 = {
            'train': {
                'input_ids': torch.stack([item['input_ids'] for item in train_inputs_stage1]),
                'attention_mask': torch.stack([item['attention_mask'] for item in train_inputs_stage1]),
                'labels': torch.stack([item['labels'] for item in train_labels_stage1]),
            }
            # You can add 'test' and 'validation' datasets similarly if needed
        }
        torch.save(preprocessed_data_stage1, save_path_stage1)
        print(f"Stage 1 preprocessed data saved to {save_path_stage1}")

        # Prepare and save Stage 2 data
        preprocessed_data_stage2 = {
            'train': {
                'input_ids': torch.stack([item['input_ids'] for item in train_inputs_stage2]),
                'attention_mask': torch.stack([item['attention_mask'] for item in train_inputs_stage2]),
                'labels': torch.stack([item['labels'] for item in train_labels_stage2]),
            }
            # You can add 'test' and 'validation' datasets similarly if needed
        }
        torch.save(preprocessed_data_stage2, save_path_stage2)
        print(f"Stage 2 preprocessed data saved to {save_path_stage2}")

        return preprocessed_data_stage1, preprocessed_data_stage2


if __name__ == '__main__':
    preprocessed_data_stage1, preprocessed_data_stage2 = get_preprocessed_data()