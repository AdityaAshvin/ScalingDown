# training_stage2.py

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # Updated import for mixed precision
from torch.utils.data import DataLoader, random_split
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
# Loss function for Knowledge Distillation
class DistillationLoss(nn.Module):
    def __init__(self, ignore_index=-100, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.seq2seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Cross-entropy loss with actual labels
        ce_loss_per_token = self.seq2seq_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        ).view(labels.size())

        valid_token_mask = (labels != self.seq2seq_loss.ignore_index)
        valid_token_counts = valid_token_mask.sum(dim=1).float().clamp(min=1.0)
        ce_loss_per_sample = (ce_loss_per_token * valid_token_mask.float()).sum(dim=1) / valid_token_counts

        # Apply temperature scaling
        T = self.temperature
        student_logits_T = student_logits / T
        teacher_logits_T = teacher_logits / T

        # Compute soft targets without adding epsilon
        student_prob = nn.functional.log_softmax(student_logits_T, dim=-1)
        teacher_prob = nn.functional.softmax(teacher_logits_T, dim=-1)

        # Distillation loss with KL Divergence
        distillation_loss = self.kl_loss(student_prob, teacher_prob) * (self.temperature ** 2)

        # Combine losses
        total_loss = (1 - self.alpha) * ce_loss_per_sample.mean() + self.alpha * distillation_loss

        return total_loss, ce_loss_per_sample.mean()



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

def save_checkpoint(model, optimizer, scheduler, epoch, batch, checkpoint_dir, is_epoch_end=False, custom_path=None):
    # Define the checkpoint filename based on type
    filename = f"epoch_checkpoint_{epoch + 1}.pth" if is_epoch_end else "checkpoint.pth"
    
    if custom_path:
        checkpoint_path = custom_path  # Use provided custom path for milestone checkpoints
    else:
        filename = f"epoch_checkpoint_{epoch + 1}.pth" if is_epoch_end else "checkpoint.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'is_epoch_end': is_epoch_end,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    if batch % 200 == 0:  # Milestone saving every 200 batches
        milestone_path = os.path.join(checkpoint_dir, f"checkpoint-epoch{epoch + 1}-batch{batch}.pth")
        torch.save(checkpoint, milestone_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    """
    Loads the latest checkpoint from the specified directory if available.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load state into.
        checkpoint_dir (str): Directory containing checkpoint files.

    Returns:
        dict: A dictionary with the latest epoch, batch, best_val_loss, best_val_accuracy, or defaults if no checkpoint found.
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        logging.info("No checkpoint found. Starting from scratch.")
        return {
            'epoch': 0, 
            'batch': 0, 
            'best_val_loss': float('inf'), 
            'best_val_accuracy': 0.0, 
        }

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info("Scheduler state loaded from checkpoint.")

    logging.info(f"Loaded checkpoint from {checkpoint_path}.")

    # Determine if the checkpoint was saved at epoch end
    is_epoch_end = checkpoint.get('is_epoch_end', False)

    if is_epoch_end:
        # If the checkpoint was saved at the end of an epoch, start from the next epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        start_batch = 0
    else:
        # If the checkpoint was saved during an epoch, continue from the same epoch and batch
        start_epoch = checkpoint.get('epoch', 0)
        start_batch = checkpoint.get('batch', 0)

    return {
        'epoch': start_epoch,
        'batch': start_batch,
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
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
    
    # Initialize the student model from scratch
    # Initialize the student model from scratch
    t5config = T5Config.from_pretrained("t5-base")  # Define T5 configuration
    student_model = T5ForConditionalGeneration(config=t5config).to(device)
    student_model.apply(student_model._init_weights)  # Reinitialize entire model weights

    # Freeze the teacher model (we don’t want to update its weights)
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Load dataset
    dataset = torch.load(config["datasets"]["socialiqa"]["path"])  # Path to the preprocessed SocialIQA dataset

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(dataset, config, tokenizer)

    accumulation_steps = config["training"]["accumulation_steps"]

    total_steps = (len(train_loader) // accumulation_steps) * config["training"]["num_epochs_stage2"]

    # Define optimizer and scheduler for student model
    optimizer = AdamW(student_model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    distillation_loss_fn = DistillationLoss(
        ignore_index=pad_token_id, 
        alpha=config["training"]["alpha"])
    scaler = GradScaler()

    # Check dataset tensor values for NaNs or inf
    for idx, sample in enumerate(dataset):
        for name, tensor in sample.items():
            if torch.is_tensor(tensor):
                if torch.isnan(tensor).any():
                    logging.warning(f"NaN detected in dataset field '{name}' at sample index {idx}")
                if torch.isinf(tensor).any():
                    logging.warning(f"Inf detected in dataset field '{name}' at sample index {idx}")

    # Initialize checkpoint loading
    checkpoint = load_checkpoint(student_model, optimizer, scheduler, checkpoint_dir)
    start_epoch = checkpoint.get('epoch', 0)
    start_batch = checkpoint.get('batch', 0)
    global_batch_count = checkpoint.get('batch', 0)

    # Training loop
    for epoch in range(start_epoch, config["training"]["num_epochs_stage2"]):
        student_model.train()
        epoch_loss = 0.0
        batch_count = 0
        current_batch = 0
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
            current_batch = start_batch  # Set the current batch to start_batch

        for batch_idx, batch in enumerate(train_loader_iter, start=1):
            # Log if NaNs are in input data or labels
            if not torch.isfinite(batch['input_ids']).all():
                logging.warning(f"NaN found in input_ids at Batch {batch_idx}")
            if not torch.isfinite(batch['labels']).all():
                logging.warning(f"NaN found in labels at Batch {batch_idx}")
            if epoch == start_epoch and current_batch < start_batch:
                current_batch += 1
                continue


            global_batch_count += 1  # Increment global batch count
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                with torch.no_grad():
                    teacher_input_ids = input_ids.to("cpu")
                    teacher_labels = labels.to("cpu")
                    teacher_outputs = teacher_model(input_ids=teacher_input_ids, labels=teacher_labels)

                student_outputs = student_model(input_ids=input_ids, labels=labels)

                # Compute loss (distillation + label loss)
                loss, ce_loss = distillation_loss_fn(
                    student_outputs.logits, 
                    teacher_outputs.logits.to(device), 
                    labels, 
                )

            # After computing loss
            if torch.isnan(loss):
                logging.warning(f"NaN loss at Epoch {epoch + 1}, Batch {batch_idx + 1}. Skipping batch.")
                continue

            # Scale the loss for mixed precision
            scaler.scale(loss).backward()

            # After backward pass and before optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

            # Check for NaNs in gradients
            skip_update = False
            for name, param in student_model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    logging.warning(f"NaN detected in gradients of {name}. Skipping update.")
                    skip_update = True
                    break

            if not skip_update:

            # Proceed with optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            batch_count += 1
            current_batch += 1
        
            # Logging and validation at specified intervals
            if (batch_idx) % validation_frequency == 0:
                val_loss, accuracy = validate(student_model, teacher_model, tokenizer, val_loader, device, pad_token_id)
                logging.info(f"Epoch {epoch + 1}, Global Batch {global_batch_count}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

            # Checkpointing logic
            if global_batch_count % config["checkpointing"]["checkpoint_frequency_batches"] == 0:
                save_checkpoint(student_model, optimizer, scheduler, epoch, global_batch_count, checkpoint_dir, False)
            if global_batch_count % config["checkpointing"]["checkpoint_frequency_milestone"] == 0:  # Milestone checkpoint
                checkpoint_milestone = os.path.join(
                    checkpoint_dir, f"checkpoint-epoch{epoch + 1}-batch{global_batch_count}.pth"
                )

                save_checkpoint(student_model, optimizer, scheduler, epoch, batch_idx, checkpoint_dir, is_epoch_end=False, custom_path=checkpoint_milestone)                
                logging.info(f"Milestone checkpoint saved to {checkpoint_milestone}")

        # End of epoch checkpoint
        save_checkpoint(student_model, optimizer, scheduler, epoch, batch_idx if batch_idx >=0 else 0, checkpoint_dir, True)

        # Log the average loss for this epoch
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
        else:
            avg_epoch_loss = 0.0
        logging.info(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_epoch_loss:.4f}")

    logging.info("Training complete.")

if __name__ == "__main__":
    main()
