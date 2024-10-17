import sys
import torch
import random
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scripts.data_preprocessing.data_preprocessing_aqua import get_preprocessed_data


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
    print(f"Finished Preprocessing. Teacher size: {len(train_teacher_inputs)}")

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
    student_model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-small').to(device)
    
    teacher_model_names = {
        'llemma': "EleutherAI/llemma_7b",
        'gptneo': "EleutherAI/gpt-neo-2.7B"
    }

    # # Load teacher models
    # print(f"Loading teacher models...")
    # teacher_models = {
    #     'llemma': AutoModelForCausalLM.from_pretrained("EleutherAI/llemma_7b").to(device),
    #     'gptneo': AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
    # }

    # Loss functions
    answer_loss_fn = nn.CrossEntropyLoss()
    rationale_loss_fn = nn.CrossEntropyLoss()

    # Optimizer for student model
    optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)

    return student_model,teacher_model_names, train_loader, device, answer_loss_fn, rationale_loss_fn, optimizer


def train_with_teacher(epoch_num, teacher_name, teacher_model_path, student_model, train_loader, device, answer_loss_fn, rationale_loss_fn, optimizer):    
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
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(device)

    for epoch in range(epoch_num):
        epoch_loss = 0
        for batch_idx, (student_input, teacher_input) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epoch_num}")):

            # Move student inputs to the device (CUDA/CPU)
            student_input_ids = student_input['input_ids'].squeeze(1).to(device)
            student_attention_mask = student_input['attention_mask'].squeeze(1).to(device)

            # Forward pass through teacher models
            teacher_outputs = {}

            teacher_input_ids = teacher_input[teacher_name]['input_ids'].squeeze(1).to(device)
            teacher_attention_mask = teacher_input[teacher_name]['attention_mask'].squeeze(1).to(device)
            with torch.no_grad():  # No gradients needed for teacher models
                teacher_logits = teacher_model(teacher_input_ids, attention_mask=teacher_attention_mask).logits
                teacher_labels = torch.argmax(teacher_logits, dim=-1)  # Get predicted token ids from teacher logits

            # Forward pass through student model (including decoder input)
            student_output = student_model(student_input_ids, attention_mask=student_attention_mask)
            student_answer_logits = student_output.logits

            # Calculate Answer Loss (CrossEntropy)
            logits = student_answer_logits.view(-1, student_answer_logits.size(-1))
            labels = teacher_labels.view(-1)
            answer_loss = answer_loss_fn(logits, labels)

            # Rationale distillation loss (student mimics teacher rationale logits)
            rationale_losses = []
            for teacher_output in teacher_outputs.items():
                student_rationale_output = student_model(student_rationale_ids,
                                                         attention_mask=student_rationale_attention_mask,
                                                         decoder_input_ids=student_decoder_input_ids).logits

                # Equalize sequence and vocab sizes
                min_seq_length = min(student_rationale_output.size(1), teacher_output.size(1))
                common_vocab_size = min(student_rationale_output.size(-1), teacher_output.size(-1))

                # Align dimensions
                aligned_student_output = student_rationale_output[:, :min_seq_length, :common_vocab_size]
                aligned_teacher_output = teacher_output[:, :min_seq_length, :common_vocab_size]

                # Flatten for loss calculation
                flattened_student_output = aligned_student_output.reshape(-1)
                flattened_teacher_output = aligned_teacher_output.reshape(-1)

                rationale_loss = rationale_loss_fn(flattened_student_output, flattened_teacher_output)
                rationale_losses.append(rationale_loss)

            # Combine rationale losses (averaging over multiple teachers)
            total_rationale_loss = torch.stack(rationale_losses).mean()

            # Combine losses (40:60 ratio for answer:rationale)
            total_loss = 0.4 * answer_loss + 0.6 * total_rationale_loss
            epoch_loss += total_loss.item()

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader)}")


if __name__ == "__main__":
    epoch_num = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    use_gpu = sys.argv[3].lower() == 'true'
    subset_ratio = float(sys.argv[4])
    student_model, teacher_model_names, train_loader, device, answer_loss_fn, rationale_loss_fn, optimizer = setup(batch_size, use_gpu, subset_ratio)
    for teacher_name, teacher_model_path in teacher_model_names.items():
        train_with_teacher(epoch_num, student_model, teacher_name, teacher_model_path, student_model, train_loader, device, answer_loss_fn, rationale_loss_fn, optimizer)


# main(2, 10, False, 0.005)
