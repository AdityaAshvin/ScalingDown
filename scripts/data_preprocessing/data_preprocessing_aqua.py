from datasets import load_dataset
from transformers import T5Tokenizer, AutoTokenizer
import pandas as pd


def load_aqua_data():
    # using hugging face
    print("Loading AQuA-Rat dataset...")
    dataset = load_dataset("deepmind/aqua_rat", "raw")

    # ref: https://huggingface.co/datasets/deepmind/aqua_rat
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    val_df = pd.DataFrame(dataset['validation'])

    return train_df, test_df, val_df


# Load tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
llemma_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")
gptneo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

teacher_tokenizers = {
    'llemma': llemma_tokenizer,
    'gptneo': gptneo_tokenizer,
}


# Function to preprocess data
def preprocess_data_for_models(df, max_length=256):
    teacher_inputs = {key: [] for key in teacher_tokenizers.keys()}
    student_inputs = []

    for index, row in df.iterrows():
        question_text = f"Question: {row['question']} Options: {row['options']}"
        rationale_text = f"Rationale: {row['rationale']}"
        answer_text = row['correct']
        student_question_encoding = t5_tokenizer(question_text, return_tensors='pt', padding='max_length',
                                                 truncation=True, max_length=max_length)
        student_rationale_encoding = t5_tokenizer(rationale_text, return_tensors='pt', padding='max_length',
                                                  truncation=True, max_length=max_length)
        decoder_input_encoding = t5_tokenizer(answer_text, return_tensors='pt', padding='max_length',
                                              truncation=True, max_length=max_length)

        # Student tokens
        student_inputs.append({
            'input_ids': student_question_encoding['input_ids'],
            'attention_mask': student_question_encoding['attention_mask'],
            'rationale_ids': student_rationale_encoding['input_ids'],
            'rationale_attention_mask': student_rationale_encoding['attention_mask'],
            'decoder_input_ids': decoder_input_encoding['input_ids'],
            'correct_index': ord(row['correct']) - ord('A')  # Assumes answers are in A, B, C, etc.
        })

        for teacher_name, tokenizer in teacher_tokenizers.items():
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            question_encoding = tokenizer(question_text, return_tensors='pt', padding='max_length', truncation=True,
                                          max_length=max_length)
            rationale_encoding = tokenizer(rationale_text, return_tensors='pt', padding='max_length', truncation=True,
                                           max_length=max_length)
            assert question_encoding['input_ids'].shape == question_encoding['attention_mask'].shape
            assert rationale_encoding['input_ids'].shape == rationale_encoding['attention_mask'].shape
            teacher_inputs[teacher_name].append({
                'input_ids': question_encoding['input_ids'],
                'attention_mask': question_encoding['attention_mask'],
                'rationale_ids': rationale_encoding['input_ids'],
                'rationale_attention_mask': rationale_encoding['attention_mask'],
                'decoder_input_ids': decoder_input_encoding['input_ids'],
                'correct_index': ord(row['correct']) - ord('A')
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


if __name__ == '__main__':
    preprocessed_data = get_preprocessed_data()
    print("Train Teacher Input Example:", preprocessed_data['train']['teacher_inputs']['llemma'][0])