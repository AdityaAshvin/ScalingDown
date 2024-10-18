import sys
import torch
import random
import os
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scripts.data_preprocessing.data_preprocessing_aqua import get_preprocessed_data

# Load tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained('tokenizers/t5_tokenizer/')
teacher_tokenizers = {
    'llemma': AutoTokenizer.from_pretrained('tokenizers/llemma_tokenizer/'),
    'gptneo': AutoTokenizer.from_pretrained('tokenizers/gptneo_tokenizer/')
}

# Ensure pad_token is set for teacher tokenizers
for tokenizer in teacher_tokenizers.values():
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

class TeacherStudentDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for teacher-student knowledge distillation.

    Args:
    - student_inputs: Preprocessed inputs for the student model.
    - teacher_inputs: Preprocessed inputs for the teacher models.

    Returns:
    - __getitem__: Returns a dictionary with student inputs and corresponding teacher inputs.
    """

    def __init__(self, student_inputs, teacher_inputs):
        self.student_inputs = student_inputs
        self.teacher_inputs = teacher_inputs

    def __len__(self):
        return len(self.student_inputs)

    def __getitem__(self, idx):
        student_input = self.student_inputs[idx]
        teacher_input = {name: self.teacher_inputs[name][idx] for name in self.teacher_inputs}
        return student_input, teacher_input


def setup(batch_size=6, use_gpu=False, subset_ratio=1.0):
    
    # Preprocess the dataset
    print("Preprocessing AQuA Dataset...")
    pre_processed_data = get_preprocessed_data()
    train_student_inputs = pre_processed_data['train']['student_inputs']
    train_teacher_inputs = pre_processed_data['train']['teacher_inputs']
    print(f"Finished Preprocessing. Dataset size: {len(train_student_inputs)}")

    # Reduce dataset size for testing purposes
    dataset_size = int(len(train_student_inputs) * subset_ratio)
    indices = random.sample(range(len(train_student_inputs)), dataset_size)
    print(f"Using {dataset_size} samples (subset ratio: {subset_ratio})")

    # Create DataLoader from the subset dataset
    train_dataset = Subset(TeacherStudentDataset(train_student_inputs, train_teacher_inputs), indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Configure device (GPU or CPU)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"GPU mode is {'enabled' if use_gpu else 'disabled'}. Using device: {device}")

    # Load student model
    print(f"Loading student model...")
    try:
        student_model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-small').to(device)
    except Exception as e:
        print(f"Error loading student model: {e}")
        exit(1)
    # student_model.resize_token_embeddings(len(t5_tokenizer))

    teacher_model_names = {
        'llemma': "EleutherAI/llemma_7b",
        'gptneo': "EleutherAI/gpt-neo-2.7B"
    }

    # Optimizer for student model
    optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)

    # Resize model embeddings after adding special tokens
    student_model.resize_token_embeddings(len(t5_tokenizer))
    
    return student_model, teacher_model_names, train_loader, device, optimizer

# def train_with_teacher(epoch_num, teacher_name, teacher_model_path, student_model, train_loader, device, answer_loss_fn, rationale_loss_fn, optimizer):    
def train_with_teacher(epoch_num, teacher_name, teacher_model_path, student_model, train_loader, device, optimizer, start_epoch=0):
    """
    Training function to perform knowledge distillation by training a student model using teacher models' guidance.

    Args:
    - epoch_num: Number of training epochs.
    - batch_size: Number of samples per batch.
    - use_gpu: Boolean indicating whether to use GPU or CPU.
    - subset_ratio: Ratio of dataset to be used during training (for testing with smaller datasets).

    The function pre-processes the AQuA dataset, sets up the training loop, computes losses, and performs
    knowledge distillation using student and teacher models.
    """

    print(f"Loading {teacher_name} teacher model...")
    try:
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(device)
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        exit(1)        
    teacher_tokenizer = teacher_tokenizers[teacher_name]
    teacher_model.resize_token_embeddings(len(teacher_tokenizer))

    for epoch in range(start_epoch, epoch_num):
        epoch_loss = 0
        for batch_idx, (student_input, teacher_input) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epoch_num}")):

            # Move student inputs to the device (CUDA/CPU)
            student_input_ids = student_input['input_ids'].squeeze(1).to(device)
            student_attention_mask = student_input['attention_mask'].squeeze(1).to(device)

            teacher_input_ids = teacher_input[teacher_name]['input_ids'].squeeze(1).to(device)
            teacher_attention_mask = teacher_input[teacher_name]['attention_mask'].squeeze(1).to(device)

            # Generate outputs from teacher model
            with torch.no_grad():
                teacher_outputs_ids = teacher_model.generate(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    max_new_tokens=128,
                    eos_token_id=teacher_tokenizer.eos_token_id,
                )

            # Decode teacher outputs to text
            teacher_outputs_text = teacher_tokenizer.batch_decode(
                teacher_outputs_ids, skip_special_tokens=False)

            teacher_outputs_encodings = t5_tokenizer(
                teacher_outputs_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=512
            ).input_ids.to(device)
            teacher_outputs_encodings[teacher_outputs_encodings == t5_tokenizer.pad_token_id] = -100

            outputs = student_model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask,
                labels=teacher_outputs_encodings
            )

            loss = outputs.loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader)}")

        # Save model checkpoint after each epoch
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, f'student_model_epoch_{epoch + 1}.pt')        
        torch.save(student_model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")

        del teacher_model
        del teacher_tokenizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    epoch_num = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    use_gpu = sys.argv[3].lower() == 'true'
    subset_ratio = float(sys.argv[4])

    student_model, teacher_model_names, train_loader, device, optimizer = setup(batch_size, use_gpu, subset_ratio)
    
    # Check for a saved model checkpoint
    start_epoch = 0
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    latest_checkpoint = None
    for file in os.listdir(checkpoint_dir):
        if file.startswith('student_model_epoch_') and file.endswith('.pt'):
            latest_checkpoint = file
    if latest_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Loading student model from {checkpoint_path}")
        student_model.load_state_dict(torch.load(checkpoint_path))
        start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
        if start_epoch >= epoch_num:
            print("Training already completed.")
            exit()
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    for teacher_name, teacher_model_path in teacher_model_names.items():
        train_with_teacher(epoch_num, teacher_name, teacher_model_path, student_model, train_loader, device, optimizer, start_epoch)

# main(2, 10, False, 0.005)