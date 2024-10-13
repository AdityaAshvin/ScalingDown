from datasets import load_dataset
from transformers import T5Tokenizer, AutoTokenizer
import pandas as pd
import torch
import os

# Load tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
llemma_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")
gptneo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

teacher_tokenizers = {
    'llemma': llemma_tokenizer,
    'gptneo': gptneo_tokenizer,
}


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
def preprocess_data_for_models(df, max_length=256):
    teacher_inputs = {key: [] for key in teacher_tokenizers.keys()}
    student_inputs = []

    for index, row in df.iterrows():
        question_text = f"Question: {row['question']} Options: {row['options']}. Explain your rationale, and limit your total response to 200 words max."

        student_question_encoding = t5_tokenizer(question_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

        # Student tokens
        student_inputs.append({
            'input_ids': student_question_encoding['input_ids'],
            'attention_mask': student_question_encoding['attention_mask'],
        })

        for teacher_name, tokenizer in teacher_tokenizers.items():
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            question_encoding = tokenizer(question_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
            assert question_encoding['input_ids'].shape == question_encoding['attention_mask'].shape
            
            teacher_inputs[teacher_name].append({'input_ids': question_encoding['input_ids'], 'attention_mask': yquestion_encoding['attention_mask'],
            })

    return teacher_inputs, student_inputs


def get_preprocessed_data():
    train_df, test_df, val_df = load_aqua_data()

    print(f"Teacher list: {', '.join(teacher_tokenizers.keys())}")
    print("Tokenizing training data...")
    train_teacher_inputs, train_student_inputs = preprocess_data_for_models(train_df)
    print("Tokenizing testing data...")
    test_teacher_inputs, test_student_inputs = preprocess_data_for_models(test_df)
    print("Tokenizing validating data...")
    val_teacher_inputs, val_student_inputs = preprocess_data_for_models(val_df)

    return {
        'train': {'teacher_inputs': train_teacher_inputs, 'student_inputs': train_student_inputs},
        'test': {'teacher_inputs': test_teacher_inputs, 'student_inputs': test_student_inputs},
        'val': {'teacher_inputs': val_teacher_inputs, 'student_inputs': val_student_inputs}
    }


def save_preprocessed_data(preprocessed_data, filename):
    directory = 'data/preprocessed/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    torch.save(preprocessed_data, filepath)
    print(f"Preprocessed data saved to {filename}")


if __name__ == '__main__':
    preprocessed_data = get_preprocessed_data()
    save_preprocessed_data(preprocessed_data, 'preprocessed_aqua_data.pt')
    print("Train Teacher Input Example:", preprocessed_data['train']['teacher_inputs']['llemma'][0])