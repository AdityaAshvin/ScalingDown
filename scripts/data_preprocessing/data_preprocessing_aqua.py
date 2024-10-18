from datasets import load_dataset
from transformers import T5Tokenizer, AutoTokenizer
import pandas as pd
import os
import pickle
import torch

special_tokens = {'additional_special_tokens': ['<ANSWER>', '<RATIONALE>']}

# Load tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
t5_tokenizer.add_special_tokens(special_tokens)


llemma_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")
llemma_tokenizer.add_special_tokens(special_tokens)

gptneo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
gptneo_tokenizer.add_special_tokens(special_tokens)

teacher_tokenizers = {
    'llemma': llemma_tokenizer,
    'gptneo': gptneo_tokenizer,
}

# After adding special tokens
t5_tokenizer.save_pretrained('tokenizers/t5_tokenizer/')
llemma_tokenizer.save_pretrained('tokenizers/llemma_tokenizer/')
gptneo_tokenizer.save_pretrained('tokenizers/gptneo_tokenizer/')

if llemma_tokenizer.pad_token is None:
    llemma_tokenizer.pad_token = llemma_tokenizer.eos_token

if gptneo_tokenizer.pad_token is None:
    gptneo_tokenizer.pad_token = gptneo_tokenizer.eos_token


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
    teacher_inputs = {key: [] for key in teacher_tokenizers.keys()}
    student_inputs = []

    for index, row in df.iterrows():
        question_text = (
            f"Question: {row['question']} " 
            f"Options: {row['options']} "
            "Please provide the answer and explain your reasoning in the following format: "
            "<ANSWER> [Your answer here] <RATIONALE> [Your rationale here]"
        )

        student_question_encoding = t5_tokenizer(
            question_text, 
            return_tensors='pt', 
            padding='longest',
            truncation=True, 
            max_length=max_length
        )

        # Student tokens
        student_inputs.append({
            'input_ids': student_question_encoding['input_ids'],
            'attention_mask': student_question_encoding['attention_mask'],
        })

        for teacher_name, tokenizer in teacher_tokenizers.items():
            # if tokenizer.pad_token is None:
            #     tokenizer.pad_token = tokenizer.eos_token
            question_encoding = tokenizer(
                question_text, 
                return_tensors='pt', 
                padding='longest', 
                truncation=True,
                max_length=max_length
            )
            
            teacher_inputs[teacher_name].append({
                'input_ids': question_encoding['input_ids'],
                'attention_mask': question_encoding['attention_mask'],
            })

    return teacher_inputs, student_inputs


def get_preprocessed_data(save_path='preprocessed_data.pkl'):    
    if os.path.exists(save_path):
        print(f"Loading preprocessed data from {save_path}...")
        with open(save_path, 'rb') as f:
            preprocessed_data = torch.load(f)
        return preprocessed_data
    else:
    
        train_df, test_df, val_df = load_aqua_data()

        print(f"Teacher list: {', '.join(teacher_tokenizers.keys())}")
        print("Tokenizing training data...")
        train_teacher_inputs, train_student_inputs = preprocess_data_for_models(train_df)
        print("Tokenizing testing data...")
        test_teacher_inputs, test_student_inputs = preprocess_data_for_models(test_df)
        print("Tokenizing validating data...")
        val_teacher_inputs, val_student_inputs = preprocess_data_for_models(val_df)

        preprocessed_data = {
            'train': {'teacher_inputs': train_teacher_inputs, 'student_inputs': train_student_inputs},
            'test': {'teacher_inputs': test_teacher_inputs, 'student_inputs': test_student_inputs},
            'val': {'teacher_inputs': val_teacher_inputs, 'student_inputs': val_student_inputs}
        }

        # Save preprocessed data to disk
        with open(save_path, 'wb') as f:
            torch.save(preprocessed_data, f)
        print(f"Preprocessed data saved to {save_path}")

        return preprocessed_data

if __name__ == '__main__':
    preprocessed_data = get_preprocessed_data()
    print("Train Teacher Input Example:", preprocessed_data['train']['teacher_inputs']['llemma'][0])