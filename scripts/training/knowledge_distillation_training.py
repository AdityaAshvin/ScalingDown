import datetime
import sys
import torch
import random
import os
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    T5Tokenizer
)
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
    """

    def __init__(self, student_inputs, teacher_inputs):
        """
        Initializes the dataset with student and teacher inputs.

        Args:
            student_inputs (list): List of student input encodings.
            teacher_inputs (dict): Dictionary mapping teacher names to their input encodings.
        """
        self.student_inputs = student_inputs
        self.teacher_inputs = teacher_inputs

    def __len__(self):
        return len(self.student_inputs)

    def __getitem__(self, idx):
        student_input = self.student_inputs[idx]
        teacher_input = {name: self.teacher_inputs[name][idx] for name in self.teacher_inputs}
        return student_input, teacher_input


def custom_collate_fn(batch):
    student_batch = [item[0] for item in batch]
    teacher_batch = [item[1] for item in batch]

    # Student inputs
    student_input_ids = [item['input_ids'].squeeze(0) for item in student_batch]
    student_attention_masks = [item['attention_mask'].squeeze(0) for item in student_batch]

    student_input_ids = torch.nn.utils.rnn.pad_sequence(
        student_input_ids, batch_first=True, padding_value=t5_tokenizer.pad_token_id)
    student_attention_masks = torch.nn.utils.rnn.pad_sequence(
        student_attention_masks, batch_first=True, padding_value=0)

    student_inputs = {
        'input_ids': student_input_ids,
        'attention_mask': student_attention_masks
    }

    # Teacher inputs
    teacher_inputs = {}
    for teacher_name in teacher_tokenizers.keys():
        teacher_input_ids = [item[teacher_name]['input_ids'].squeeze(0) for item in teacher_batch]
        teacher_attention_masks = [item[teacher_name]['attention_mask'].squeeze(0) for item in teacher_batch]

        teacher_input_ids = torch.nn.utils.rnn.pad_sequence(
            teacher_input_ids, batch_first=True, padding_value=teacher_tokenizers[teacher_name].pad_token_id)
        teacher_attention_masks = torch.nn.utils.rnn.pad_sequence(
            teacher_attention_masks, batch_first=True, padding_value=0)

        teacher_inputs[teacher_name] = {
            'input_ids': teacher_input_ids,
            'attention_mask': teacher_attention_masks
        }

    return student_inputs, teacher_inputs


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

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

    teacher_model_names = {
        'llemma': "EleutherAI/llemma_7b",
        'gptneo': "EleutherAI/gpt-neo-2.7B"
    }

    # Optimizer for student model
    optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)

    # Resize model embeddings after adding special tokens
    student_model.resize_token_embeddings(len(t5_tokenizer))
    
    return student_model, teacher_model_names, train_loader, device, optimizer

def train_with_teacher(epoch_num, teacher_name, teacher_model_path, student_model, train_loader, device, optimizer, start_epoch=0, save_interval=500):
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
    # teacher_model.resize_token_embeddings(len(teacher_tokenizer))

    # Initialize variables to track training progress
    total_batches = len(train_loader)
    global_step = 0

    for epoch in range(start_epoch, epoch_num):
        epoch_loss = 0
        for batch_idx, (student_input, teacher_input) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epoch_num}")):

            # Update global step
            global_step += 1            

            # Move student inputs to the device (CUDA/CPU)
            student_input_ids = student_input['input_ids'].to(device)
            student_attention_mask = student_input['attention_mask'].to(device)

            teacher_input_ids = teacher_input[teacher_name]['input_ids'].to(device)
            teacher_attention_mask = teacher_input[teacher_name]['attention_mask'].to(device)

            print(f"Asking teacher a question at {datetime.datetime.now().strftime('%H:%M:%S')}")

            # Generate outputs from teacher model
            with torch.no_grad():
                teacher_outputs_ids = teacher_model.generate(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    max_new_tokens=150,
                    min_length=50,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=False
                )

            print(f"Received teacher's answer at {datetime.datetime.now().strftime('%H:%M:%S')}")

            # Decode teacher outputs to text
            teacher_outputs_text = teacher_tokenizer.batch_decode(
                teacher_outputs_ids, skip_special_tokens=True)
            
            print(f"Teacher's output: {teacher_outputs_text}")

            teacher_outputs_encodings = t5_tokenizer(
                teacher_outputs_text,
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=512
            ).input_ids.to(device)
            
            teacher_outputs_encodings[teacher_outputs_encodings == t5_tokenizer.pad_token_id] = -100

            print(f"Asking student a question at {datetime.datetime.now().strftime('%H:%M:%S')}")

            outputs = student_model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask,
                labels=teacher_outputs_encodings
            )

            logits = outputs.logits
            predicted_ids = logits.argmax(dim=-1)
            decoded_student_output = t5_tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            print(f"Student's output: {decoded_student_output}")

            print(f"Received student's answer at {datetime.datetime.now().strftime('%H:%M:%S')}")

            print(f"Beginning training of student at {datetime.datetime.now().strftime('%H:%M:%S')}")
            loss = outputs.loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"Completed training of student at {datetime.datetime.now().strftime('%H:%M:%S')}")

            # Check if it's time to save a checkpoint
            if global_step % save_interval == 0:
                print(f"Saving checkpoint at epoch {epoch + 1}, batch {batch_idx + 1} (global step {global_step})")
                save_checkpoint(student_model, optimizer, epoch + 1, global_step)


        print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader)}")

        # Save a checkpoint at the end of each epoch
        print(f"Saving checkpoint at the end of epoch {epoch + 1}")
        save_checkpoint(student_model, optimizer, epoch + 1, global_step)
    
    # Save final checkpoint after training
    print("Saving final checkpoint after training completion.")
    save_checkpoint(student_model, optimizer, epoch_num, global_step)


    # Clear teacher model from memory
    print(f"Removing {teacher_name} teacher model from memory...")
    del teacher_model
    del teacher_tokenizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def save_checkpoint(model, optimizer, epoch, global_step):
    """
    Saves the model and optimizer states along with training progress.

    Args:
    - model: The student model.
    - optimizer: The optimizer.
    - epoch: Current epoch number.
    - global_step: Current global step (batch number).
    - teacher_name: Name of the teacher model.
    - final: Boolean indicating if this is the final checkpoint.
    """

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
    }

    checkpoint_filename = 'student_model_latest.pt'

    save_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # Save the checkpoint
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Loads the model and optimizer states from a checkpoint.

    Args:
    - model: The student model.
    - optimizer: The optimizer.
    - checkpoint_path: Path to the checkpoint file.
    - device: Device to map the model and optimizer states.

    Returns:
    - epoch: The epoch to resume from.
    - global_step: The global step to resume from.
    """

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    print(f"Loaded checkpoint: Epoch {epoch}, Global Step {global_step}")
    return epoch, global_step        


if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 5:
        print("Usage: python training_script.py <epoch_num> <batch_size> <use_gpu> <subset_ratio> [save_interval]")
        print("Example: python training_script.py 10 32 True 1.0 500")
        sys.exit(1)

    epoch_num = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    use_gpu = sys.argv[3].lower() == 'true'
    subset_ratio = float(sys.argv[4])
    save_interval = int(sys.argv[5]) if len(sys.argv) > 5 else 500  # Default to saving every 500 batches

    student_model, teacher_model_names, train_loader, device, optimizer = setup(batch_size, use_gpu, subset_ratio)
    
    # Define the checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists
    
    # Initialize training progress variables
    start_epoch = 0
    global_step = 0
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'student_model_latest.pt')

    # Check for a saved '_latest' checkpoint
    if os.path.exists(latest_checkpoint_path):
        load_epoch, load_step = load_checkpoint(student_model, optimizer, latest_checkpoint_path, device)
        start_epoch = load_epoch
        global_step = load_step
        print(f"Resuming training from epoch {start_epoch}, global step {global_step}")
    else:
        print("No '_latest' checkpoint found. Starting training from scratch.")


    # Iterate over each teacher model for knowledge distillation
    for teacher_name, teacher_model_path in teacher_model_names.items():
        train_with_teacher(
            epoch_num=epoch_num,
            teacher_name=teacher_name,
            teacher_model_path=teacher_model_path,
            student_model=student_model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            start_epoch=start_epoch,
            save_interval=save_interval
        )

# main(2, 10, False, 0.005)