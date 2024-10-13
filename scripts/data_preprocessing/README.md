## Usage

How to Use get_preprocessed_data()

The get_preprocessed_data() function loads and tokenizes the AQuA dataset. It prepares both teacher and student input formats, which can then be used in the training process.

Function: get_preprocessed_data()

```python
from data_preprocessing import get_preprocessed_data

# Call the function to get the preprocessed data
preprocessed_data = get_preprocessed_data()

# Accessing the data
train_data = preprocessed_data['train']
test_data = preprocessed_data['test']
val_data = preprocessed_data['val']
```

### Description of the Output:

The function returns a dictionary with three keys: train, test, and val. Each key contains two sets of preprocessed inputs:
- student_inputs: Preprocessed data for the student model (e.g., T5-small).
- teacher_inputs: Preprocessed data for the teacher models (e.g., GPT-Neo, LLaMA).

##### Example
```python
{
    'train': {
        'student_inputs': [ 
            { 
              'input_ids': tensor([...]),
              'attention_mask': tensor([...]),
              'decoder_input_ids': tensor([...]),
              'rationale_ids': tensor([...]),
              'rationale_attention_mask': tensor([...]),
              'correct_index': int(2)
            },
            ...
        ],
        'teacher_inputs': {
            'llemma': [ 
                { 
                  'input_ids': tensor([...]),
                  'attention_mask': tensor([...]),
                  'rationale_ids': tensor([...]),
                  'rationale_attention_mask': tensor([...]),
                  'correct_index': int(2)
                },
                ...
            ],
            'gptneo': [
                { 
                  'input_ids': tensor([...]),
                  'attention_mask': tensor([...]),
                  'rationale_ids': tensor([...]),
                  'rationale_attention_mask': tensor([...]),
                  'correct_index': int(2)
                },
                ...
            ]
        }
    },
    'test': {
        'student_inputs': [...],
        'teacher_inputs': {...}
    },
    'val': {
        'student_inputs': [...],
        'teacher_inputs': {...}
    }
}
```

### How Output Will Be Used

1. student_inputs:
- Contains tokenized question text and rationale text for the student model.
- input_ids and attention_mask correspond to the tokenized input.
- decoder_input_ids is used for the student’s output.
- rationale_ids and rationale_attention_mask are used for reasoning-related tasks.
2. teacher_inputs:
- Contains tokenized data for multiple teacher models.
- Organized by teacher model names (e.g., llemma, gptneo).
- Each teacher model has input_ids, attention_mask, and reasoning-related fields (rationale_ids and rationale_attention_mask).
