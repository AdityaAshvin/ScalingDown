# Preprocessed Data for Knowledge Distillation

This folder contains the preprocessed data for the student-teacher knowledge distillation project. The data has been tokenized and structured for both the **student model (T5-small)** and **teacher models (e.g., Llemma, GPT-Neo)**. The preprocessed data is saved in `.pt` format for efficient loading and use during training.

## Files

- **`preprocessed_data.pt`**: This file contains the tokenized inputs for both student and teacher models. The data is stored as PyTorch tensors.

## Data Format

The preprocessed data is structured as follows:

### **Student Inputs**

Each entry in `student_inputs` contains tokenized data for the student model (T5-small). The structure is:

```python
[
    {
        'input_ids': tensor([...]),  # Token IDs for the question and options
        'attention_mask': tensor([...]),  # Attention mask for question tokens
        'rationale_ids': tensor([...]),  # Token IDs for the rationale text
        'rationale_attention_mask': tensor([...]),  # Attention mask for rationale tokens
        'correct_index': int  # Correct answer index (0 for 'A', 1 for 'B', etc.)
    },
    ...
]
```
### **Teacher Inputs**
The teacher inputs are stored in a dictionary where each key corresponds to a teacher model (e.g., llemma, gptneo). Each teacher has a list of tokenized inputs similar to the student:
```python
{
    'llemma': [
        {
            'input_ids': tensor([...]),  # Token IDs for the question and options
            'attention_mask': tensor([...]),  # Attention mask for question tokens
            'rationale_ids': tensor([...]),  # Token IDs for the rationale text
            'rationale_attention_mask': tensor([...]),  # Attention mask for rationale tokens
            'correct_index': int  # Correct answer index
        },
        ...
    ],
    'gptneo': [
        {
            'input_ids': tensor([...]),
            'attention_mask': tensor([...]),
            'rationale_ids': tensor([...]),
            'rationale_attention_mask': tensor([...]),
            'correct_index': int
        },
        ...
    ]
}
```

### **Usage**
To load the preprocessed data into your training script, use the following:
```python
import torch

# Load the preprocessed data
preprocessed_data = torch.load('preprocessed_data.pt')

# Access student and teacher inputs
student_inputs = preprocessed_data['student_inputs']
teacher_inputs = preprocessed_data['teacher_inputs']
```

### **Note**
- The .pt file is in PyTorch format and should be loaded into a PyTorch environment for training.
-  Tokenization includes both question+options and rationale explanations to support the knowledge distillation process between teacher and student models.
