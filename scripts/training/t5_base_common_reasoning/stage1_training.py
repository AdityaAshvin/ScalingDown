# training_stage1.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup, T5Config
from torch.optim import AdamW
import yaml
import argparse
import logging
import os
import random
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence




class CustomLoss(nn.Module):
    def __init__(self, ignore_index=-100, pad_token_id=0, blank_penalty=10.0, length_penalty_weight=0.5, single_token_bonus=0.2):
        super(CustomLoss, self).__init__()
        self.seq2seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.pad_token_id = pad_token_id
        self.blank_penalty = blank_penalty
        self.length_penalty_weight = length_penalty_weight
        self.single_token_bonus = single_token_bonus

    def forward(self, logits, labels, model_output_ids):
        """
        Computes the custom loss by combining standard cross-entropy loss with additional penalties and bonuses.

        Args:
            logits (Tensor): The model's raw output logits of shape (batch_size, seq_length, vocab_size).
            labels (Tensor): The ground truth labels of shape (batch_size, seq_length).
            model_output_ids (Tensor): The model's generated token IDs of shape (batch_size, seq_length).

        Returns:
            Tuple[Tensor, Tensor]: The computed total loss and the cross-entropy loss.
        """
        # Compute cross-entropy loss over all tokens per sample
        loss_per_token = self.seq2seq_loss(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        ).view(labels.size())  # Shape: (batch_size, seq_length)

        # Sum loss over tokens for each sample and handle division by zero
        valid_token_mask = (labels != self.seq2seq_loss.ignore_index)
        valid_token_counts = valid_token_mask.sum(dim=1).float()
        loss_per_sample = (loss_per_token * valid_token_mask.float()).sum(dim=1) / valid_token_counts.clamp(min=1.0)

        # Initialize total loss with per-sample cross-entropy loss
        total_loss = loss_per_sample.clone()

        # Compute lengths of generated sequences per sample (excluding padding tokens)
        gen_lengths = (model_output_ids != self.pad_token_id).sum(dim=1).float()  # Shape: (batch_size,)

        # Identify samples that are fully padded (i.e., generated sequences are empty)
        blank_mask = gen_lengths.eq(0)  # Shape: (batch_size,)

        # Apply blank penalty to samples that generated no tokens
        total_loss += self.blank_penalty * blank_mask.float()

        # Compute length penalty per sample
        length_penalty = self.length_penalty_weight * torch.clamp(gen_lengths - 1, min=0.0)
        total_loss += length_penalty

        # Apply single token bonus to samples where the generated sequence length is exactly 1
        single_token_mask = gen_lengths.eq(1).float()
        total_loss = total_loss * (1 - self.single_token_bonus * single_token_mask)

        # Average total loss and cross-entropy loss over the batch
        total_loss = total_loss.mean()
        cross_entropy_loss = loss_per_sample.mean()

        return total_loss, cross_entropy_loss


def apply_repetition_penalty(logits, previous_answers, tokenizer, penalty=1.2):
    """
    Applies a repetition penalty to logits if the model attempts to generate an answer it has given previously.

    Args:
    - logits: Tensor of shape (batch_size, seq_length, vocab_size) containing model logits.
    - previous_answers: List of previously generated answers (strings).
    - tokenizer: The tokenizer used to encode/decode tokens.
    - penalty: Float, the factor by which to penalize repeated answers.

    Returns:
    - logits: Tensor with applied penalties.
    """
    batch_size, seq_length, vocab_size = logits.size()

    # Create a set of unique first token IDs from previous answers
    penalized_token_ids = set()
    for answer in previous_answers:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(answer_ids) == 0:
            continue
        first_token_id = answer_ids[0]
        if 0 <= first_token_id < vocab_size:
            penalized_token_ids.add(first_token_id)

    # Convert to a list for indexing
    penalized_token_ids = list(penalized_token_ids)

    if len(penalized_token_ids) == 0:
        return logits  # No tokens to penalize

    # Penalize the logits for the first token position in each sample
    for i in range(batch_size):
        logits[i, 0, penalized_token_ids] /= penalty  # Penalize the first token position

    return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Train T5-base model on Winograd dataset.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--dataset_percentage", type=float, default=1.0, help="Percentage of the dataset to use for training.")
    parser.add_argument("--validation_frequency", type=int, default=25, help="Frequency of validation.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Directory path to save/load checkpoints.")
    parser.add_argument("--log_file", type=str, default=None, help="File path for logging.")
    return parser.parse_args()


def setup_logging(log_file):
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_from_file(path):
    dataset = torch.load(path)
    return dataset


def split_dataset(dataset, validation_frequency):
    total_samples = len(dataset)
    num_val = total_samples // (validation_frequency + 1)
    num_train = total_samples - num_val
    train_subset, val_subset = random_split(dataset, [num_train, num_val])
    return train_subset, val_subset


def create_dataloaders(train_subset, val_subset, config, tokenizer):
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {'input_ids': input_ids, 'labels': labels}

    train_loader = DataLoader(
        train_subset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count(),
        collate_fn=collate_fn
    )
    return train_loader, val_loader


def validate(model, tokenizer, val_loader, device, pad_token_id):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # Randomly select one batch from val_loader
        val_batch = random.choice(list(val_loader))
        
        input_ids = val_batch['input_ids'].to(device)
        labels = val_batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        val_loss = loss.item()  # Single sample loss

        # Generate model's answer
        generated_ids = model.generate(input_ids, max_length=2)
        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        true = tokenizer.decode(labels[0], skip_special_tokens=True).strip()

        # Decode input to get question and options for display
        question_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        correct_answer = true  # Assuming the label contains the correct answer
        model_answer = pred

        # Display question, correct answer, and model's answer
        print("\nValidation Sample:")
        print(f"Question and Options: {question_text}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")

        correct = (pred == true)
        total = 1

    model.train()
    accuracy = correct / total if total > 0 else 0
    return val_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, batch, checkpoint_dir, is_epoch_end=False):
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
        dict: A dictionary with the latest epoch, batch, best_val_loss, and best_val_accuracy, or defaults if no checkpoint found.
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        logging.info("No checkpoint found. Starting from scratch.")
        return {'epoch': 0, 'batch': 0, 'best_val_loss': float('inf'), 'best_val_accuracy': 0.0}

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
        'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0)
    }

def load_tokenizer(config):
    tokenizer_path = os.path.join(config["tokenizer"]["save_dir"], config["tokenizer"]["name"])
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_model(config, device):
    t5config = T5Config.from_pretrained(config["tokenizer"]["name"])
    t5config.dropout_rate = 0.3
    t5config.attention_dropout_rate = 0.3
    
    model = T5ForConditionalGeneration.from_pretrained(
        config["tokenizer"]["name"],
        config=t5config
        ).to(device)
    return model


def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    # Set up logging
    checkpoint_dir = args.checkpoint_path if args.checkpoint_path else config["checkpointing"]["save_dir"]
    log_file = args.log_file if args.log_file else config["logging"]["log_file"]
    setup_logging(log_file)

    # Load tokenizer
    tokenizer = load_tokenizer(config)
    pad_token_id = tokenizer.pad_token_id
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["training"].get("learning_rate", 5e-5))
    scheduler = None

    # Initialize the custom loss function
    custom_loss_fn = CustomLoss(
        ignore_index=-100, 
        pad_token_id=pad_token_id,
        blank_penalty=config["training"].get("blank_penalty", 10.0),  # Use blank_penalty here
        length_penalty_weight=config["training"].get("length_penalty_weight", 0.5),
        single_token_bonus=config["training"].get("single_token_bonus", 0.2)
    )
    
    # Load checkpoint if exists and retrieve training state
    checkpoint = load_checkpoint(model, optimizer, scheduler, checkpoint_dir)
    start_epoch = checkpoint.get('epoch', 0)
    start_batch = checkpoint.get('batch', 0)
    
    # Load dataset
    dataset = load_dataset_from_file(config["datasets"]["winograd"]["path"])

    # Apply dataset percentage if specified
    subset_size = int(len(dataset) * args.dataset_percentage)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    subset = Subset(dataset, subset_indices)
    
    # Split into training and validation
    train_subset, val_subset = split_dataset(subset, args.validation_frequency)
    train_loader, val_loader = create_dataloaders(train_subset, val_subset, config, tokenizer)
    
    # Initialize scheduler after calculating total steps for the entire stage
    total_steps = len(train_loader) * config["training"]["num_epochs_stage1"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    global_batch_count = checkpoint.get('batch', 0)  # Initialize from checkpoint
    recent_val_accuracies = []  # List to store recent validation losses

    # Add this right before the training loop

    previous_answers = []

    # Training loop
    for epoch in range(start_epoch, config["training"]["num_epochs_stage1"]):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # Wrap train_loader in tqdm for progress bar display
        train_loader_iter = iter(tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{config['training']['num_epochs_stage1']}", 
            leave=False
        ))

        # If resuming mid-epoch, continue from the last batch
        if epoch == start_epoch and start_batch:
            for _ in range(start_batch):
                try:
                    next(train_loader_iter)
                except StopIteration:
                    break
            start_batch = 0
            
        for batch_idx, batch in enumerate(train_loader_iter):
            global_batch_count += 1  # Increment global batch count
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass, loss calculation, backward pass, optimizer step
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            
            # Apply repetition penalty to the logits based on previous answers
            logits = apply_repetition_penalty(logits, previous_answers, tokenizer, penalty=config['training']['repeat_penalty'])
    
            # Generate sequences to update previous_answers
            generated_ids = model.generate(input_ids, max_length=labels.size(1))

            # Compute loss with custom loss function
            loss, ce_loss = custom_loss_fn(logits, labels, generated_ids)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            batch_count += 1

            # Generate sequences to update previous_answers
            generated_ids = model.generate(input_ids, max_length=labels.size(1))

            # Decode generated sequences into answers
            generated_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            previous_answers.extend(generated_answers)

            # Optionally, limit the size of previous_answers to prevent it from growing indefinitely
            if len(previous_answers) > 1000:  # For example, keep the last 1000 answers
                previous_answers = previous_answers[-1000:]

            # Logging and validation at specified intervals
            if global_batch_count % args.validation_frequency == 0:
                val_loss, val_accuracy = validate(model, tokenizer, val_loader, device, pad_token_id)
                
                # Store the validation accuracy in recent_val_accuracies and keep only the last 25
                recent_val_accuracies.append(val_accuracy)
                if len(recent_val_accuracies) > 25:
                    recent_val_accuracies.pop(0)

                # Calculate average validation loss over the last 25
                avg_val_accuracy = sum(recent_val_accuracies) / len(recent_val_accuracies)

                # Log the average validation loss along with the current validation details
                logging.info(f"Epoch {epoch + 1}, Global Batch {global_batch_count}, Training Loss: {loss.item():.4f}, "
                             f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%, "
                             f"Average Validation Accuracy (last 25): {avg_val_accuracy:.4f}")
            
            # Save checkpoints periodically based on configuration
            if global_batch_count % config["checkpointing"]["checkpoint_frequency_batches"] == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_batch_count, checkpoint_dir, False
                )
                
        # End of epoch checkpoint (outside of batch loop)
        if batch_count == 0:
            avg_epoch_loss = epoch_loss / (batch_count+1)
        else:
            avg_epoch_loss = epoch_loss / batch_count
        logging.info(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_epoch_loss:.4f}")
        save_checkpoint(
            model, optimizer, scheduler, epoch, 0,  # Set batch to 0 for end-of-epoch checkpoint
            checkpoint_dir, True
        )

    logging.info("Training complete.")


if __name__ == "__main__":
    main()