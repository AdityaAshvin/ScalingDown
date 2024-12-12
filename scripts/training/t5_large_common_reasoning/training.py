# training_stage2.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup, T5Config
from torch.optim import AdamW
import yaml
import argparse
import logging
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk

# Loss function for Knowledge Distillation
class DistillationLoss(nn.Module):
    def __init__(self, ignore_index=-100, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.seq2seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        vocab_size = student_logits.size(-1)  # Get vocab_size from logits
        valid_labels = labels[labels != self.seq2seq_loss.ignore_index]
        
        if valid_labels.numel() > 0:
            min_label = valid_labels.min().item()
            max_label = valid_labels.max().item()
            if min_label < 0 or max_label >= vocab_size:
                raise ValueError(f"Label values out of range: min={min_label}, max={max_label}, vocab_size={vocab_size}")


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

def create_dataloaders(train_dataset, val_dataset, config, tokenizer):
    """
    Creates DataLoader objects for training and validation sets.

    Args:
        train_dataset (Dataset): The preprocessed training dataset.
        val_dataset (Dataset): The preprocessed validation dataset.
        config (dict): Configuration dictionary with batch size and other parameters.
        tokenizer (Tokenizer): The tokenizer used for padding.

    Returns:
        train_loader, val_loader (DataLoader, DataLoader): DataLoader objects for training and validation.
    """

    def collate_fn(batch):
        # Extract texts
        input_texts = [item['input_text'] for item in batch]
        label_texts = [item['label_text'] for item in batch]

        # Existing code
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).long()

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_texts': input_texts,
            'label_texts': label_texts
        }
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
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count()
    )

    return train_loader, val_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Train Flan-T5-large model with knowledge distillation.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--dataset_percentage", type=float, default=1.0, help="Percentage of the dataset to use for training.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory path to save/load checkpoints.")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load (optional).")
    parser.add_argument("--global_batch_count", type=int, default=None, help="Global batch count to resume from (if not in checkpoint).")
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
    tokenizer_path = os.path.join(config["tokenizer"]["save_dir"])
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    print(f"Tokenizer Vocabulary Size: {vocab_size}")
    print("Token IDs for '<1>':", tokenizer.encode('<1>'))
    print("Token IDs for '<2>':", tokenizer.encode('<2>'))
    print("Token IDs for '<3>':", tokenizer.encode('<3>'))
    return tokenizer

def load_model(config, model_name, device, tokenizer):
    t5config = T5Config.from_pretrained(model_name)
    t5config.dropout_rate = config["training"]["dropout_rate"]       # Adjust as per config
    t5config.attention_dropout_rate = config["training"]["dropout_rate"]  # Adjust as per config
    
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        config=t5config
    )
    
    # Resize token embeddings to accommodate new special tokens
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    return model

def validate(student_model, tokenizer, val_loader, device, pad_token_id, sample_fraction=1.0):
    student_model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    samples_printed = 0  # Counter for printed samples

    # Determine the number of samples to use
    total_samples_in_dataset = len(val_loader.dataset)
    num_samples = max(1, int(sample_fraction * total_samples_in_dataset))

    # Create a random subset of indices
    subset_indices = np.random.choice(total_samples_in_dataset, num_samples, replace=False)

    # Create a SubsetRandomSampler
    sampler = torch.utils.data.SubsetRandomSampler(subset_indices)

    # Create a new DataLoader with the sampler
    val_loader_subset = DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size,
        sampler=sampler,
        collate_fn=val_loader.collate_fn,
        num_workers=val_loader.num_workers
    )

    # Normalization function
    import string
    import re

    def normalize_text(text):
        # Remove special tokens like <pad>, </s>, <s> from text
        tokens_to_remove = ['<pad>', '</s>', '<s>']
        for token in tokens_to_remove:
            text = text.replace(token, '')
        text = text.strip()
        
        # If text is in format '<number>', extract the number
        match = re.match(r'<(\d)>', text)
        if match:
            return match.group(1)
        else:
            # Try to find any digit in text
            match = re.search(r'(\d)', text)
            if match:
                return match.group(1)
            else:
                # Additional normalization if needed
                text = text.lower()
                text = ' '.join(text.split())
                return text

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader_subset, desc="Validation", leave=False)):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            input_texts = batch['input_texts']  # Original questions with options
            label_texts = batch['label_texts']  # Original correct answers

            # Compute loss
            student_outputs = student_model(input_ids=input_ids, labels=labels)
            loss = student_outputs.loss
            val_loss += loss.item()

            # Generate predictions
            generated_ids = student_model.generate(
                input_ids=input_ids,
                max_length=labels.size(1),
                num_beams=5,
                early_stopping=True
            )

            # Decode the generated sequences and labels
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            label_ids = labels.cpu().numpy()
            # Replace -100 with pad_token_id in labels
            label_ids = np.where(label_ids == -100, pad_token_id, label_ids)
            labels_decoded = tokenizer.batch_decode(label_ids, skip_special_tokens=False)

            # Compute accuracy and print sample data
            for pred, label_text, input_text in zip(preds, label_texts, input_texts):
                pred_norm = normalize_text(pred)
                label_norm = normalize_text(label_text)
                if pred_norm == label_norm:
                    total_correct += 1
                total_samples += 1

                # Print the first sample
                if samples_printed < 1:
                    print("\nValidation Sample:")
                    print(f"Question: {input_text}")
                    print(f"Correct Answer: {label_text}")
                    print(f"Student's Response: {pred}")
                    print(f"Normalized Student's Response: {pred_norm}")
                    print(f"Normalized Correct Answer: {label_norm}")
                    samples_printed += 1

    # Compute average loss
    avg_val_loss = val_loss / len(val_loader_subset)
    # Compute accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_val_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, batch, global_batch_count, checkpoint_dir, is_epoch_end=False, custom_path=None,config=None):
    # Define the checkpoint filename based on type
    if custom_path:
        checkpoint_path = custom_path  # Use provided custom path for milestone checkpoints
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    else:
        if is_epoch_end:
            filename = f"epoch_checkpoint_{epoch + 1}.pth"
        else:
            filename = "checkpoint.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'global_batch_count': global_batch_count,
        'is_epoch_end': is_epoch_end,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")

    # Milestone saving every checkpoint_frequency_milestone batches
    if not custom_path and batch % config["checkpointing"]["checkpoint_frequency_milestone"] == 0:
        milestone_path = os.path.join(checkpoint_dir, f"checkpoint-epoch{epoch + 1}-batch{batch}.pth")
        os.makedirs(os.path.dirname(milestone_path), exist_ok=True)
        torch.save(checkpoint, milestone_path)
        logging.info(f"Milestone checkpoint saved to {milestone_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, checkpoint_file=None, manual_global_batch_count=None):
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
    if checkpoint_file:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    if not os.path.exists(checkpoint_path):
        logging.info("No checkpoint found. Starting from scratch.")
        return {
            'epoch': 0, 
            'batch': 0, 
            'global_batch_count': manual_global_batch_count or 0,            
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

    # Retrieve global_batch_count
    if 'global_batch_count' in checkpoint:
        global_batch_count = checkpoint['global_batch_count']
    else:
        if manual_global_batch_count is not None:
            global_batch_count = manual_global_batch_count
            logging.warning(f"'global_batch_count' not found in checkpoint. Using provided value: {global_batch_count}")
        else:
            logging.error("Checkpoint does not contain 'global_batch_count' and no value was provided. Cannot proceed.")
            exit(1)  # Or handle as appropriate

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
        'global_batch_count': global_batch_count,  # Return it here
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    validation_frequency = config["training"]["validation_frequency"]

    # Set up logging
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else config["checkpointing"]["save_dir"]
    log_file = args.log_file if args.log_file else config["logging"]["log_file"]
    setup_logging(log_file)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.dirname(log_file)) if log_file else SummaryWriter()

    # Set seeds for reproducibility
    set_seed(config["random_seed"])

    # Load tokenizer
    tokenizer = load_tokenizer(config)
    pad_token_id = tokenizer.pad_token_id

    # Load teacher and student models
    device = torch.device("cpu")
    
    # Load tokenizer first
    tokenizer = load_tokenizer(config)
    pad_token_id = tokenizer.pad_token_id

    # Load teacher model onto CPU to save GPU memory
    teacher_model = load_model(config, "google/flan-t5-xl", "cpu", tokenizer)  # Change device to 'cpu' for teacher model

    # Initialize the student model from scratch
    student_model = load_model(config, "google/flan-t5-large", device, tokenizer)  # Pass tokenizer
    # student_model.apply(student_model._init_weights)  # Reinitialize entire model weights

    # Freeze the teacher model (we don’t want to update its weights)
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Load preprocessed datasets
    try:
        train_split_path = os.path.join(config["datasets"]["socialiqa"]["path"], config["datasets"]["socialiqa"]["splits"]["train"], "dataset.pt")
        val_split_path = os.path.join(config["datasets"]["socialiqa"]["path"], config["datasets"]["socialiqa"]["splits"]["validation"], "dataset.pt")        
    
        train_dataset = load_from_disk(train_split_path)
        val_dataset = load_from_disk(val_split_path)
    except Exception as e:
        logging.error(f"Error loading preprocessed datasets: {e}")
        return

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config, tokenizer)

    accumulation_steps = config["training"]["accumulation_steps"]

    total_steps = (len(train_loader) // accumulation_steps) * config["training"]["num_train_epochs_stage2"]

    # Define optimizer and scheduler for student model
    optimizer = AdamW(student_model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps
    )
    distillation_loss_fn = DistillationLoss(
        ignore_index=-100, 
        alpha=config["training"]["alpha"],
        temperature=2.0,
    )

    # Initialize checkpoint loading
    checkpoint = load_checkpoint(
        student_model, 
        optimizer, 
        scheduler, 
        checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        manual_global_batch_count=args.global_batch_count
    )
    start_epoch = checkpoint.get('epoch', 0)
    start_batch = checkpoint.get('batch', 0)
    global_batch_count = checkpoint.get('global_batch_count', 0)

    logging.info(f"Resuming training from epoch {start_epoch}, batch {start_batch}, global batch count {global_batch_count}.")
    try:
        # Training loop
        for epoch in range(start_epoch, config["training"]["num_train_epochs_stage2"]):
            student_model.train()
            epoch_loss = 0.0
            batch_count = 0
            current_batch = 0
            batch_idx = -1
            
            # Wrap train_loader in tqdm for progress bar display
            train_loader_iter = iter(tqdm(
                train_loader, 
                desc=f"Epoch {epoch + 1}/{config['training']['num_train_epochs_stage2']}", 
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
                    logging.warning(f"NaN found in input_ids at Epoch {epoch + 1}, Batch {batch_idx}")
                if not torch.isfinite(batch['labels']).all():
                    logging.warning(f"NaN found in labels at Epoch {epoch + 1}, Batch {batch_idx}")
                if epoch == start_epoch and current_batch < start_batch:
                    current_batch += 1
                    continue

                
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                # Debug: Inspect label values
                labels_cpu = labels.cpu()
                min_label = labels_cpu[labels_cpu != -100].min().item()
                max_label = labels_cpu[labels_cpu != -100].max().item()
                student_outputs = student_model(input_ids=input_ids, labels=labels)
                student_logits = student_outputs.logits
                vocab_size = student_logits.size(-1)  # Use the size from logits directly

                # logging.info(f"Label value range: min={min_label}, max={max_label}, vocab_size={vocab_size}")

                if min_label < -100 or max_label >= vocab_size:
                    logging.error(f"Invalid label values detected: min={min_label}, max={max_label}, vocab_size={vocab_size}")                
                    continue  # Skip this batch

                optimizer.zero_grad()

                # Forward pass through teacher model
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=input_ids.to("cpu"), labels=labels.to("cpu"))
                    teacher_logits = teacher_outputs.logits.to(device)

                # Forward pass through student model
                
                

                # Compute loss (distillation + label loss)
                loss, ce_loss = distillation_loss_fn(
                    student_logits, 
                    teacher_logits, 
                    labels
                )

                # After computing loss
                if torch.isnan(loss):
                    logging.warning(f"NaN loss at Epoch {epoch + 1}, Batch {batch_idx}. Skipping batch.")
                    continue

                # Backward pass with gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                epoch_loss += loss.item()
                batch_count += 1
                global_batch_count += 1  # Increment global batch count here

                # Gradient Accumulation Step
                if batch_idx % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), config["training"]["max_norm"])
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Logging and validation at specified intervals
                if global_batch_count % config["training"]["validation_frequency"] == 0:
                    avg_val_loss, accuracy = validate(
                        student_model,
                        tokenizer,
                        val_loader,
                        device,
                        pad_token_id,
                        sample_fraction=config["validation"]["val_percent"],  # Use 5% of the validation data
                    )
                    logging.info(
                        f"Epoch {epoch + 1}, Batch {global_batch_count}, "
                        f"Training Loss: {epoch_loss / batch_count:.4f}, "
                        f"Validation Loss: {avg_val_loss:.4f}, "
                        f"Accuracy: {accuracy:.4f}"
                    )
                
                    # Log metrics to TensorBoard
                    writer.add_scalar('Loss/Training', epoch_loss / batch_count, global_batch_count)
                    writer.add_scalar('Loss/Validation', avg_val_loss, global_batch_count)
                    writer.add_scalar('Metrics/Accuracy', accuracy, global_batch_count)
                    
                    # Reset epoch_loss and batch_count after logging
                    epoch_loss = 0.0
                    batch_count = 0

                # Checkpointing logic
                if global_batch_count % config["checkpointing"]["checkpoint_frequency_batches"] == 0:
                    save_checkpoint(
                        student_model, 
                        optimizer, 
                        scheduler, 
                        epoch, 
                        batch_idx if batch_idx>=0 else 0,
                        global_batch_count, 
                        checkpoint_dir, 
                        is_epoch_end=False, 
                        config=config)
                if global_batch_count % config["checkpointing"]["checkpoint_frequency_milestone"] == 0:  # Milestone checkpoint
                    checkpoint_milestone = os.path.join(
                        checkpoint_dir, f"checkpoint-epoch{epoch + 1}-batch{global_batch_count}.pth"
                    )
                    save_checkpoint(
                        student_model, 
                        optimizer, 
                        scheduler, 
                        epoch, 
                        batch_idx if batch_idx>=0 else 0,
                        global_batch_count, 
                        checkpoint_dir, 
                        is_epoch_end=False, 
                        custom_path=checkpoint_milestone, 
                        config=config)                
                    logging.info(f"Milestone checkpoint saved to {checkpoint_milestone}")

            # Log the average loss for this epoch
            if batch_count > 0:
                # Save checkpoint marking epoch as completed
                save_checkpoint(
                    student_model, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    batch_idx if batch_idx >= 0 else 0, 
                    global_batch_count,
                    checkpoint_dir, 
                    is_epoch_end=True, 
                    config=config
                )

                # Log the average loss for this epoch
                avg_epoch_loss = epoch_loss / batch_count
                logging.info(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_epoch_loss:.4f}")
            else:
                logging.info(f"Epoch {epoch + 1} had no batches processed. Not marking epoch as completed.")
                # Save checkpoint without marking epoch as completed
                save_checkpoint(
                    student_model, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    batch_idx if batch_idx >= 0 else 0, 
                    global_batch_count,
                    checkpoint_dir, 
                    is_epoch_end=False, 
                    config=config
                )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving checkpoint...")
        # Save the current state with is_epoch_end=False
        save_checkpoint(
            student_model, 
            optimizer, 
            scheduler,
            epoch, 
            batch_idx if batch_idx >= 0 else 0, 
            global_batch_count,
            checkpoint_dir, 
            is_epoch_end=False, 
            config=config
        )
        logging.info("Checkpoint saved. Exiting.")
        return

    # Save the final trained student model
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    student_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logging.info(f"Final trained student model saved to {final_model_path}")

    logging.info("Training complete.")
    writer.close()

if __name__ == "__main__":
    main()
