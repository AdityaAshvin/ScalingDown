import sys
import torch
import random
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scripts.data_preprocessing.data_preprocessing_aqua import get_preprocessed_data


def load_train_data(train_data_path):
    preprocessed_data = torch.load(train_data_path, weights_only=True)
    train_student_inputs = preprocessed_data['student_inputs']
    train_teacher_inputs = preprocessed_data['teacher_inputs']

    return train_student_inputs, train_teacher_inputs


class TeacherStudentDataset(torch.utils.data.Dataset):
    def __init__(self, student_inputs, teacher_inputs):
        self.student_inputs = student_inputs
        self.teacher_inputs = teacher_inputs

    def __len__(self):
        return len(self.student_inputs)

    def __getitem__(self, idx):
        student_input = self.student_inputs[idx]
        teacher_input = {name: self.teacher_inputs[name][idx] for name in self.teacher_inputs}
        return student_input, teacher_input


def main(epoch_num, batch_size=6, use_gpu=False, subset_ratio=1.0):
    print("Preprocessing AQuA Dataset...")
    pre_processed_data = get_preprocessed_data()
    train_student_inputs = pre_processed_data['train']['student_inputs']
    train_teacher_inputs = pre_processed_data['train']['teacher_inputs']
    print(f"Finished Preprocessing. Teacher size: {len(train_teacher_inputs)}")

    # Reduce dataset size for testing purposes (optional, don't need it if your pc is fire)
    dataset_size = int(len(train_student_inputs) * subset_ratio)
    indices = random.sample(range(len(train_student_inputs)), dataset_size)
    print(f"Using {dataset_size} samples (subset ratio: {subset_ratio})")

    # Create DataLoader from the subset dataset
    train_dataset = Subset(TeacherStudentDataset(train_student_inputs, train_teacher_inputs), indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Device configuration
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"GPU mode is {'enabled' if use_gpu else 'disabled'}. Using device: {device}")

    print(f"Loading student model...")
    student_model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-small').to(device)
    print(f"Loading teacher model...")
    teacher_models = {
        'llemma': AutoModelForCausalLM.from_pretrained("EleutherAI/llemma_7b").to(device),
        'gptneo': AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
    }

    # Loss functions
    cross_entropy_loss = nn.CrossEntropyLoss()
    rationale_loss_fn = nn.MSELoss()

    # Optimizer for student model
    optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)

    # Training loop
    for epoch in range(epoch_num):
        epoch_loss = 0

        for batch_idx, (student_input, teacher_input) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epoch_num}")):
            # Move student inputs to the device (CUDA/CPU)
            student_input_ids = student_input['input_ids'].squeeze(1).to(device)
            student_attention_mask = student_input['attention_mask'].squeeze(1).to(device)
            student_rationale_ids = student_input['rationale_ids'].squeeze(1).to(device)
            student_rationale_attention_mask = student_input['rationale_attention_mask'].squeeze(1).to(device)
            student_decoder_input_ids = student_input['decoder_input_ids'].squeeze(1).to(device)

            # Forward pass through teacher models
            teacher_outputs = {}
            for teacher_name, teacher_model in teacher_models.items():
                teacher_input_ids = teacher_input[teacher_name]['input_ids'].squeeze(1).to(device)
                teacher_attention_mask = teacher_input[teacher_name]['attention_mask'].squeeze(1).to(device)
                with torch.no_grad():
                    teacher_outputs[teacher_name] = teacher_model(teacher_input_ids,
                                                                  attention_mask=teacher_attention_mask).logits

            # Forward pass through student model (including decoder input)
            student_output = student_model(student_input_ids, attention_mask=student_attention_mask,
                                           decoder_input_ids=student_decoder_input_ids)
            student_answer_logits = student_output.logits

            # reshape student answer to 2D tensor
            logits = student_answer_logits.view(-1, student_answer_logits.size(-1))
            # reshape actual answer to 2D tensor
            labels = student_decoder_input_ids.view(-1)

            # Calculate CrossEntropyLoss
            answer_loss = cross_entropy_loss(logits, labels)

            # Rationale distillation loss (student mimics teacher rationale logits)
            rationale_losses = []
            for teacher_name, teacher_output in teacher_outputs.items():
                # Forward pass through student model for rationale
                student_rationale_output = student_model(student_rationale_ids,
                                                         attention_mask=student_rationale_attention_mask,
                                                         decoder_input_ids=student_decoder_input_ids).logits

                # equalize sequence size
                min_seq_length = min(student_rationale_output.size(1), teacher_output.size(1))
                truncated_student_rationale_output = student_rationale_output[:, :min_seq_length, :]
                truncated_teacher_output = teacher_output[:, :min_seq_length, :]
                # equalize vocab size
                # Note that the "vocab" of teacher and student might not be the same space (need truncation)
                common_vocab_size = min(truncated_student_rationale_output.size(-1), truncated_teacher_output.size(-1))
                aligned_student_output = truncated_student_rationale_output[:, :, :common_vocab_size]
                aligned_teacher_output = truncated_teacher_output[:, :, :common_vocab_size]
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

        # Epoch completion
        print(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(train_loader)}")


# if __name__ == "__main__":
#     epoch_num = int(sys.argv[1])
#     use_gpu = sys.argv[2].lower() == 'true'  # Set True or False for GPU usage
#     subset_ratio = float(sys.argv[3])  # Set the subset ratio (e.g., 0.5 = 50% of the dataset)
#     main(epoch_num, batch_size=2, use_gpu=use_gpu, subset_ratio=subset_ratio)

main(2, 10, False, 0.005)
