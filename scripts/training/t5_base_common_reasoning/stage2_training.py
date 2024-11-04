# training_stage2.py

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # Updated import for mixed precision
from torch.utils.data import DataLoader, random_split, Subset
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup, T5Config
from torch.optim import AdamW
import yaml
import argparse
import logging
import os
import random
from tqdm import tqdm
from torch.nn.functional import mse_loss
from torch.nn.utils.rnn import pad_sequence


# Loss function for Knowledge Distillation
class DistillationLoss(nn.Module):
    def __init__(self, ignore_index=-100, alpha=0.5, pad_token_id=0, repeat_penalty=1.2, length_penalty_weight=0.5, blank_penalty=10.0):
        super(DistillationLoss, self).__init__()
        self.seq2seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.alpha = alpha  # Weighting factor between teacher-student and label loss
        self.pad_token_id = pad_token_id
        self.repeat_penalty = repeat_penalty
        self.length_penalty_weight = length_penalty_weight
        self.blank_penalty = blank_penalty

    def forward(self, student_logits, teacher_logits, labels, model_output_ids):
        # Cross-entropy loss with actual labels
        ce_loss_per_token = self.seq2seq_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        ).view(labels.size())  # Shape: (batch_size, seq_length)
        
        # Mask out the padding tokens for accurate loss computation
        valid_token_mask = (labels != self.seq2seq_loss.ignore_index)
        valid_token_counts = valid_token_mask.sum(dim=1).float()
        ce_loss_per_sample = (ce_loss_per_token * valid_token_mask.float()).sum(dim=1) / valid_token_counts.clamp(min=1.0)  # Shape: (batch_size,)

        # Distillation loss between teacher and student logits (Mean Squared Error)
        distillation_loss = mse_loss(student_logits, teacher_logits)
        
        # Initialize total loss with the weighted combination of CE loss and distillation loss
        total_loss = (1 - self.alpha) * ce_loss_per_sample + self.alpha * distillation_loss  # Shape: (batch_size,)
        
        # Penalty for generating empty or blank responses
        gen_lengths = (model_output_ids != self.pad_token_id).sum(dim=1).float()  # Shape: (batch_size,)
        blank_mask = gen_lengths.eq(0)  # Shape: (batch_size,)

        # Apply the blank penalty for each sample that generated an empty response
        total_loss += self.blank_penalty * blank_mask.float()  # Shape: (batch_size,)
        
        # Penalty for overly long responses
        length_penalty = self.length_penalty_weight * torch.clamp(gen_lengths - 1, min=0.0)  # Shape: (batch_size,)
        total_loss += length_penalty  # Shape: (batch_size,)
        
        # Average total loss and cross-entropy loss over the batch
        total_loss = total_loss.mean()  # Scalar
        cross_entropy_loss = ce_loss_per_sample.mean()  # Scalar

        return total_loss, cross_entropy_loss


def apply_repetition_penalty(logits, previous_answers, tokenizer, penalty=1.2):
    batch_size, seq_length, vocab_size = logits.size()

    # Track the count of the most recent answer
    if len(previous_answers) >= 3:
        # Check the last three answers to see if they are the same
        if previous_answers[-1] == previous_answers[-2] == previous_answers[-3]:
            # Get the first token ID of the repeated answer
            answer_ids = tokenizer.encode(previous_answers[-1], add_special_tokens=False)
            if len(answer_ids) == 0:
                return logits
            penalized_token_id = answer_ids[0]
            
            if 0 <= penalized_token_id < vocab_size:
                # Penalize the logits for the first token position in each sample if repeated 3 times in a row
                logits[:, 0, penalized_token_id] /= penalty  # Vectorized operation

    return logits

def create_dataloaders(dataset, config, tokenizer):
    """
    Splits the dataset into training and validation sets and creates DataLoader objects for both.

    Args:
        dataset (Dataset): The processed dataset containing input_ids and labels.
        config (dict): Configuration dictionary with batch size and other parameters.
        tokenizer (Tokenizer): The tokenizer used for padding.

    Returns:
        train_loader, val_loader (DataLoader, DataLoader): DataLoader objects for training and validation.
    """

    # Split the dataset into training and validation sets
    total_samples = len(dataset)
    val_size = total_samples // (config["training"]["validation_frequency"] + 1)
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    def collate_fn(batch):
        # Collate function to handle padding within each batch
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad sequences to the maximum length in the batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored label positions
        
        return {'input_ids': input_ids, 'labels': labels}

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count()
    )

    return train_loader, val_loader



def parse_args():
    parser = argparse.ArgumentParser(description="Train T5 model with knowledge distillation.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--dataset_percentage", type=float, default=1.0, help="Percentage of the dataset to use for training.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Directory path to save/load checkpoints.")
    parser.add_argument("--log_file", type=str, default=None, help="File path for logging.")
    return parser.parse_args()

def setup_logging(log_file):
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_tokenizer(config):
    tokenizer_path = os.path.join(config["tokenizer"]["save_dir"], config["tokenizer"]["name"])
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_model(config, model_name, device):
    t5config = T5Config.from_pretrained(model_name)
    t5config.dropout_rate = 0.3  # Adjust as needed
    t5config.attention_dropout_rate = 0.3  # Adjust as needed
    
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        config=t5config
    ).to(device)
    return model

def validate(student_model, teacher_model, tokenizer, val_loader, device, pad_token_id):
    student_model.eval()
    teacher_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # Randomly select one batch from val_loader
        val_batch = random.choice(list(val_loader))
        
        input_ids = val_batch['input_ids'].to(device)
        labels = val_batch['labels'].to(device)
        
        # Compute loss for the student model
        student_outputs = student_model(input_ids=input_ids, labels=labels)
        loss = student_outputs.loss
        val_loss = loss.item()  # Single sample loss

        # Generate teacher's answer
        teacher_generated_ids = teacher_model.generate(input_ids.to("cpu"), max_length=2)
        teacher_answer = tokenizer.decode(teacher_generated_ids[0], skip_special_tokens=True).strip()

        # Generate student's answer
        student_generated_ids = student_model.generate(input_ids, max_length=2)
        student_answer = tokenizer.decode(student_generated_ids[0], skip_special_tokens=True).strip()

        # Decode input to get question and options for display
        question_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        correct_answer = tokenizer.decode(labels[0], skip_special_tokens=True).strip()

        # Display question, correct answer, teacher's answer, and student's answer
        print("\nValidation Sample:")
        print(f"Question and Options: {question_text}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Teacher's Answer: {teacher_answer}")
        print(f"Student's Answer: {student_answer}")

        # Check if the student's answer is correct
        correct = (student_answer == correct_answer)
        total = 1

    student_model.train()
    teacher_model.train()
    accuracy = correct / total if total > 0 else 0
    return val_loss, accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, batch, checkpoint_dir, is_epoch_end=False, previous_answers=None):
    # Define the checkpoint filename based on type
    filename = f"epoch_checkpoint_{epoch + 1}.pth" if is_epoch_end else "checkpoint.pth"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'previous_answers': previous_answers  # Add this line
    }
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    """
    Loads the latest checkpoint from the specified directory if available.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load state into.
        checkpoint_dir (str): Directory containing checkpoint files.

    Returns:
        dict: A dictionary with the latest epoch, batch, best_val_loss, best_val_accuracy, and previous_answers, or defaults if no checkpoint found.
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        logging.info("No checkpoint found. Starting from scratch.")
        return {'epoch': 0, 'batch': 0, 'best_val_loss': float('inf'), 'best_val_accuracy': 0.0, 'previous_answers': []}

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info("Scheduler state loaded from checkpoint.")

    logging.info(f"Loaded checkpoint from {checkpoint_path}.")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'batch': checkpoint.get('batch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
        'previous_answers': checkpoint.get('previous_answers', [])
    }

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    validation_frequency = config["training"]["validation_frequency"]

    # Set up logging
    checkpoint_dir = args.checkpoint_path if args.checkpoint_path else config["checkpointing"]["save_dir"]
    log_file = args.log_file if args.log_file else config["logging"]["log_file"]
    setup_logging(log_file)

    # Load tokenizer
    tokenizer = load_tokenizer(config)
    pad_token_id = tokenizer.pad_token_id
    
    # Load teacher and student models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load teacher model onto CPU to save GPU memory if needed
    teacher_model = load_model(config, "google/flan-t5-xl", "cpu")  # Change device to 'cpu' for teacher model
    student_model = load_model(config, "t5-base", device)  # Keep student model on GPU
    
    # Freeze the teacher model (we don’t want to update its weights)
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Define optimizer and scheduler for student model
    optimizer = AdamW(student_model.parameters(), lr=config["training"].get("learning_rate", 5e-5))
    distillation_loss_fn = DistillationLoss(
        ignore_index=pad_token_id, 
        alpha=config["training"].get("alpha", 0.5),
        pad_token_id=pad_token_id,
        repeat_penalty=config["training"].get("repeat_penalty", 1.2),
        length_penalty_weight=config["training"].get("length_penalty_weight", 0.5),
        blank_penalty=config["training"].get("blank_penalty", 10.0)
    )

    # Gradient accumulation settings
    accumulation_steps = config["training"].get("accumulation_steps", 1)

    # Load dataset
    dataset = torch.load(config["datasets"]["socialiqa"]["path"])  # Path to the preprocessed SocialIQA dataset

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(dataset, config, tokenizer)

    # Define scheduler for student model
    total_steps = len(train_loader) * config["training"]["num_epochs_stage2"] // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # Initialize the GradScaler for mixed precision
    scaler = GradScaler()

    # Load checkpoint if exists and retrieve training state
    checkpoint = load_checkpoint(student_model, optimizer, scheduler, checkpoint_dir)
    start_epoch = checkpoint.get('epoch', 0)
    start_batch = checkpoint.get('batch', 0)
    previous_answers = checkpoint.get('previous_answers', [])  # Initialize from checkpoint

    # Initialize tracking variables
    global_batch_count = checkpoint.get('batch', 0)  # Initialize from checkpoint

    # Training loop
    for epoch in range(start_epoch, config["training"]["num_epochs_stage2"]):
        student_model.train()
        epoch_loss = 0.0
        batch_count = 0
        batch_idx = -1
        
        # Wrap train_loader in tqdm for progress bar display
        train_loader_iter = iter(tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{config['training']['num_epochs_stage2']}", 
            leave=False
        ))

        # If resuming mid-epoch, continue from the last batch
        if epoch == start_epoch and start_batch > 0:
            for _ in range(start_batch):
                try:
                    next(train_loader_iter)
                except StopIteration:
                    break

        for batch_idx, batch in enumerate(train_loader_iter, start=start_batch):            
            global_batch_count += 1  # Increment global batch count
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):                # Forward pass through teacher and student
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=input_ids.to("cpu"), labels=labels.to("cpu"))
                student_outputs = student_model(input_ids=input_ids, labels=labels)
                
                # Apply the repetition penalty to the logits based on previous answers
                penalized_logits = apply_repetition_penalty(
                    student_outputs.logits, 
                    previous_answers, 
                    tokenizer, 
                    penalty=config['training'].get('repeat_penalty', 1.2)
                )

                # Generate sequences to update previous_answers
                generated_ids = student_model.generate(input_ids, max_length=labels.size(1))
                generated_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                previous_answers.extend(generated_answers)

                # Limit size of previous_answers to prevent it from growing indefinitely
                if len(previous_answers) > 1000:
                    previous_answers = previous_answers[-1000:]

                # Compute loss (distillation + label loss)
                loss, ce_loss = distillation_loss_fn(
                    penalized_logits, 
                    teacher_outputs.logits.to(device), 
                    labels, 
                    generated_ids  # Pass generated_ids instead of logits
                )

            # Scale the loss for mixed precision
            scaler.scale(loss).backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            batch_count += 1

            # Logging and validation at specified intervals
            if (batch_idx + 1) % validation_frequency == 0:
                val_loss, accuracy = validate(student_model, teacher_model, tokenizer, val_loader, device, pad_token_id)
                logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

            # Save checkpoints periodically based on configuration
            if global_batch_count % config["checkpointing"]["checkpoint_frequency_batches"] == 0:
                save_checkpoint(
                    student_model, optimizer, scheduler, epoch, global_batch_count, checkpoint_dir, False, previous_answers
                )

        # Save checkpoint at the end of each epoch
        save_checkpoint(
            student_model, optimizer, scheduler, epoch, batch_idx, checkpoint_dir, True, previous_answers
        )

        # Log the average loss for this epoch
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
        else:
            avg_epoch_loss = 0.0
        logging.info(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_epoch_loss:.4f}")

    logging.info("Training complete.")

if __name__ == "__main__":
    main()
