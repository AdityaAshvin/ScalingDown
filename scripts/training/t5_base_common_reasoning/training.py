# training.py

import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
import yaml
import argparse
import logging
import os
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence


class CustomLoss(nn.Module):
    def __init__(self, ignore_index=-100, penalty_weight=10.0, pad_token_id=0):
        """
        Initializes the CustomLoss module.

        Args:
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            penalty_weight (float): Weight to scale the penalty term.
            pad_token_id (int): Token ID used for padding. Used to ignore padding tokens in penalty calculation.
        """
        super(CustomLoss, self).__init__()
        self.seq2seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.classification_loss = nn.CrossEntropyLoss()
        self.penalty_weight = penalty_weight
        self.pad_token_id = pad_token_id

    def forward(self, logits, labels, stage):
        if stage == 2:
            # Only consider the logits of the final token in the sequence
            logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            
            # Flatten labels to 1D for classification loss
            labels = labels[:, 0] if labels.dim() > 1 else labels  # Ensure labels are 1D
            
            # Calculate cross-entropy loss for exact integer match
            classification_loss = self.classification_loss(logits, labels)
            
            # Penalty for non-exact matches (if logits do not match the exact integer label)
            penalty = torch.where(labels != logits.argmax(dim=-1), self.penalty_weight, 0.0).mean()
            
            # Combine classification loss with the penalty
            total_loss = classification_loss + penalty
            return total_loss
        else:
            # Use sequence-based cross-entropy for Stage 1
            return self.seq2seq_loss(logits.view(-1, logits.size(-1)), labels.view(-1))





def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train T5-base model on Winograd and SocialIQA datasets.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--dataset_percentage",
        type=float,
        default=1.0,
        help="Percentage of the dataset to use for training (between 0 and 1)."
    )
    parser.add_argument(
        "--validation_frequency",
        type=int,
        default=25,
        help="Frequency of validation (e.g., 25 means 1 validation per 25 training samples)."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Directory path to save and load checkpoints. Overrides config if provided."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="File path for logging training and validation metrics. Overrides config if provided."
    )
    args = parser.parse_args()
    return args


def setup_logging(log_file):
    """
    Sets up logging to both console and a log file.

    Args:
        log_file (str): Path to the log file. If None, logging is only to the console.
    """
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
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_from_file(path):
    """
    Loads a preprocessed dataset from a file.

    Args:
        path (str): Path to the preprocessed dataset file.

    Returns:
        torch.utils.data.Dataset: Loaded dataset.
    """
    if not os.path.exists(path):
        logging.error(f"Dataset file {path} does not exist.")
        raise FileNotFoundError(f"Dataset file {path} does not exist.")
    dataset = torch.load(path)
    return dataset


def split_dataset(dataset, validation_frequency):
    """
    Splits the dataset into training and validation subsets based on validation frequency.
    For every 'validation_frequency' training samples, one validation sample is reserved.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        validation_frequency (int): Frequency of validation.

    Returns:
        tuple: (train_subset, val_subset)
    """
    total_samples = len(dataset)
    num_val = total_samples // (validation_frequency + 1)
    num_train = total_samples - num_val
    train_subset, val_subset = random_split(dataset, [num_train, num_val])
    return train_subset, val_subset


def custom_collate_fn(batch, tokenizer):
    """
    Custom collate function to handle variable-length sequences in incorrect_option_ids.
    """
    # Separate out each type of data from the batch
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    incorrect_option_ids = [item['incorrect_option_ids'] for item in batch]

    # Pad sequences to the same length
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Padding labels with -100 to ignore
    incorrect_option_ids = pad_sequence(incorrect_option_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {'input_ids': input_ids, 'labels': labels, 'incorrect_option_ids': incorrect_option_ids}


def create_dataloaders(train_subset, val_subset, config, tokenizer):
    """
    Creates DataLoader instances for training and validation datasets.

    Args:
        train_subset (torch.utils.data.Subset): Training subset.
        val_subset (torch.utils.data.Subset): Validation subset.
        config (dict): Configuration parameters.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_subset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count() if os.name != 'nt' else 0,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer)
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,  # Batch size of 1 for validation to display individual examples
        shuffle=False,
        num_workers=os.cpu_count() if os.name != 'nt' else 0,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer)
    )
    return train_loader, val_loader


def load_model(config, device):
    """
    Loads the T5 model based on the tokenizer name specified in the configuration.

    Args:
        config (dict): Configuration parameters.
        device (torch.device): Device to load the model on.

    Returns:
        transformers.T5ForConditionalGeneration: Loaded T5 model.
    """
    logging.info("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained(config["tokenizer"]["name"])  # Load from Hugging Face
    model.to(device)
    return model


def load_tokenizer(config):
    """
    Loads the T5 tokenizer from the saved directory.

    Args:
        config (dict): Configuration parameters.

    Returns:
        transformers.T5Tokenizer: Loaded tokenizer.
    """
    tokenizer_path = os.path.join(config["tokenizer"]["save_dir"], config["tokenizer"]["name"])
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer directory {tokenizer_path} does not exist.")
        raise FileNotFoundError(f"Tokenizer directory {tokenizer_path} does not exist.")
    logging.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    """
    Loads the latest checkpoint from the specified directory if available.
    Considers both regular and periodic checkpoints.
    Returns a dictionary with epoch, stage, and batch information if a checkpoint is loaded, else empty dict.

    Args:
        model (transformers.T5ForConditionalGeneration): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler.LambdaLR or similar): The scheduler to load state into.
        checkpoint_dir (str): Directory containing checkpoint files.

    Returns:
        dict: Dictionary containing 'epoch', 'stage', and 'batch' if a checkpoint is loaded, else empty dict.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoint directory {checkpoint_dir} created.")
        return {}
    
    checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.endswith('.pth')]
    if not checkpoints:
        logging.info("No checkpoint files found.")
        return {}
    
    # Patterns to match regular and periodic checkpoints
    regular_pattern = r'checkpoint_stage(\d+)_epoch(\d+)\.pth'
    periodic_pattern = r'checkpoint_stage(\d+)_epoch(\d+)_batch(\d+)\.pth'
    
    checkpoints_with_info = []
    for ckpt in checkpoints:
        regular_match = re.match(regular_pattern, ckpt)
        periodic_match = re.match(periodic_pattern, ckpt)
        if regular_match:
            stage = int(regular_match.group(1))
            epoch = int(regular_match.group(2))
            batch = None  # No batch number for regular checkpoints
            checkpoints_with_info.append((ckpt, stage, epoch, batch))
        elif periodic_match:
            stage = int(periodic_match.group(1))
            epoch = int(periodic_match.group(2))
            batch = int(periodic_match.group(3))
            checkpoints_with_info.append((ckpt, stage, epoch, batch))
    
    if not checkpoints_with_info:
        logging.info("No valid checkpoints with stage and epoch information found.")
        return {}
    
    # Sort checkpoints: higher stage > higher epoch > higher batch
    def checkpoint_sort_key(ckpt_info):
        # For sorting, None batch should be considered higher than any batch number
        # So that regular checkpoints are preferred over periodic ones at the same epoch
        _, stage, epoch, batch = ckpt_info
        return (stage, epoch, batch if batch is not None else float('inf'))
    
    # Get the latest checkpoint based on the sorting key
    latest_ckpt_info = max(checkpoints_with_info, key=checkpoint_sort_key)
    latest_ckpt, latest_stage, latest_epoch, latest_batch = latest_ckpt_info
    
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info("Scheduler state loaded from checkpoint.")
    else:
        logging.warning("No scheduler state found in checkpoint or scheduler is None. Scheduler state not loaded.")
    
    epoch = checkpoint['epoch']
    stage = checkpoint['stage']
    batch = checkpoint.get('batch', latest_batch)  # Get batch number if available
    
    logging.info(f"Resumed from stage {stage}, epoch {epoch}, batch {batch if batch else 'N/A'}")
    
    return {'epoch': epoch, 'stage': stage, 'batch': batch}


def save_checkpoint(model, optimizer, scheduler, epoch, stage, checkpoint_dir, best=False, best_accuracy=False, periodic=False, batch=None):
    """
    Saves the current state of the model and optimizer to a checkpoint file.

    Args:
        model (transformers.T5ForConditionalGeneration): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler.LambdaLR or similar): The scheduler to save.
        epoch (int): Current epoch number.
        stage (int): Current training stage.
        checkpoint_dir (str): Directory to save the checkpoint.
        best (bool, optional): If True, saves as the best model based on loss. Defaults to False.
        best_accuracy (bool, optional): If True, saves as the best model based on accuracy. Defaults to False.
        periodic (bool, optional): If True, saves as a periodic checkpoint. Defaults to False.
        batch (int, optional): Current batch number for periodic checkpoints. Defaults to None.
    """
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_stage{stage}_loss.pth")
    elif best_accuracy:
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_stage{stage}_accuracy.pth")
    elif periodic and batch is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_stage{stage}_epoch{epoch}_batch{batch}.pth")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_stage{stage}_epoch{epoch}.pth")
    
    # Log the type of checkpoint being saved
    if best:
        logging.info(f"Saving best model based on loss at epoch {epoch} to {checkpoint_path}")
    elif best_accuracy:
        logging.info(f"Saving best model based on accuracy at epoch {epoch} to {checkpoint_path}")
    elif periodic and batch is not None:
        logging.info(f"Saving periodic checkpoint at epoch {epoch}, batch {batch} to {checkpoint_path}")
    else:
        logging.info(f"Saving regular checkpoint at epoch {epoch} to {checkpoint_path}")
    
    try:
        # Prepare the checkpoint dictionary
        checkpoint_dict = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None  # Save scheduler state if available
        }
        
        # Include batch number if it's a periodic checkpoint
        if periodic and batch is not None:
            checkpoint_dict['batch'] = batch
        
        # Save the checkpoint
        torch.save(checkpoint_dict, checkpoint_path)
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
        print(f"Error saving checkpoint to {checkpoint_path}: {e}")



def validate(model, tokenizer, val_subset, device, config, stage, pad_token_id):
    """
    Performs validation on a single validation example from the validation subset.
    Displays and logs the question, true answer, and model's answer.
    Returns the validation loss and accuracy for that example.

    Args:
        model (transformers.T5ForConditionalGeneration): The model to validate.
        tokenizer (transformers.T5Tokenizer): The tokenizer used.
        val_subset (torch.utils.data.Subset): Validation subset.
        device (torch.device): Device to perform validation on.
        config (dict): Configuration parameters.
        stage (int): Current training stage.
        pad_token_id (int): Token ID used for padding.

    Returns:
        tuple: (average validation loss, accuracy percentage)
    """
    model.eval()
    total_val_loss = 0.0
    val_loss_count = 0
    correct = 0
    total = 0

    # Check if the validation subset is empty
    if len(val_subset) == 0:
        logging.warning("No validation samples available.")
        return 0.0, 0.0

    # Select a random validation sample
    random_idx = random.randint(0, len(val_subset) - 1)
    sample = val_subset[random_idx]

    input_ids = sample['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
    labels = sample['labels'].unsqueeze(0).to(device)
    incorrect_option_ids = sample.get('incorrect_option_ids', None)
    if incorrect_option_ids is not None:
        incorrect_option_ids = sample['incorrect_option_ids'].unsqueeze(0).to(device)  # Shape: (1, max_incorrect_length)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        total_val_loss += loss.item()

        # Decode input and labels
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        labels_decoded = torch.where(labels == -100, pad_token_id, labels)
        true_answer = tokenizer.decode(labels_decoded[0], skip_special_tokens=True).strip()

        # Generate prediction
        generated_ids = model.generate(
            input_ids=input_ids, 
            max_length=config["preprocessing"][f"max_length_labels_stage{stage}"],
            num_beams=1, 
            temperature=0.9,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        predicted_answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # Determine if the prediction is correct
        if true_answer.lower() == predicted_answer.lower():
            correct += 1
        total += 1

        # Display on command line
        print("\nValidation Example:")
        print(f"Question: {input_text}")
        print(f"True Answer: {true_answer}")
        print(f"Model Answer: {predicted_answer}")
        print(f"Validation Accuracy: {correct / total * 100:.2f}%\n")

        # Log validation
        logging.info(f"Validation Example:")
        logging.info(f"Question: {input_text}")
        logging.info(f"True Answer: {true_answer}")
        logging.info(f"Model Answer: {predicted_answer}")
        logging.info(f"Validation Accuracy: {correct / total * 100:.2f}%")

        val_loss_count += 1

    avg_val_loss = total_val_loss / val_loss_count if val_loss_count > 0 else 0.0
    accuracy = correct / total * 100 if total > 0 else 0.0
    model.train()
    return avg_val_loss, accuracy


def main():
    # Parse command-line arguments
    args = parse_args()
    print(f"Checkpoint #1")
    # Load configuration
    config = load_config(args.config)

    # After loading configuration and before the training loop
    checkpoint_frequency = config["checkpointing"].get("checkpoint_frequency_batches", None)

    # **Add configuration validation here**
    required_keys = ["tokenizer", "datasets", "preprocessing", "training", "logging", "checkpointing"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration key: '{key}'")
    
    # Override config with command-line arguments if provided
    checkpoint_dir = args.checkpoint_path if args.checkpoint_path else config["checkpointing"]["save_dir"]
    log_file = args.log_file if args.log_file else config["logging"]["log_file"]  # Default log file name
    
    # Set up logging
    setup_logging(log_file)
    print(f"Checkpoint #2")
    
    # Log the configuration
    logging.info(f"Loaded configuration from {args.config}")
    
    # Load tokenizer
    try:
        tokenizer = load_tokenizer(config)
        logging.info("Tokenizer loaded successfully.")
    except FileNotFoundError as e:
        logging.error(str(e))
        return
    
    pad_token_id = tokenizer.pad_token_id
    print(f"Checkpoint #3")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = load_model(config, device)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=config["training"].get("learning_rate", 5e-5))
    
    # Define scheduler
    scheduler = None  # Will be initialized per stage
    
    # Initialize the custom loss function
    custom_loss_fn = CustomLoss(
        ignore_index=-100, 
        penalty_weight=config["training"].get("penalty_weight", 10.0),
        pad_token_id=pad_token_id
    )
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(model, optimizer, scheduler, checkpoint_dir)
    start_epoch = checkpoint.get('epoch', 0)
    current_stage = checkpoint.get('stage', 1)
    starting_batch = checkpoint.get('batch', 0)

    # Define training stages
    stages = [
        {
            "stage_num": 1,
            "name": "Stage 1 - Pronoun Disambiguation",
            "dataset_path": config["datasets"]["winograd"]["path"],
            "max_length_labels": config["preprocessing"]["max_length_labels_stage1"],
            "num_epochs": config["training"]["num_epochs_stage1"]
        },
        {
            "stage_num": 2,
            "name": "Stage 2 - Common Sense Reasoning",
            "dataset_path": config["datasets"]["socialiqa"]["path"],
            "max_length_labels": config["preprocessing"]["max_length_labels_stage2"],
            "num_epochs": config["training"]["num_epochs_stage2"]
        }
    ]
    
    # Remove early stopping variables
    # early_stopping_patience = config["checkpointing"].get("early_stopping_patience", 5)
    # epochs_no_improve = 0
    
    # Iterate over each stage
    for stage in stages:
        stage_num = stage["stage_num"]
        stage_name = stage["name"]
        dataset_path = stage["dataset_path"]
        max_length_labels = stage["max_length_labels"]
        num_epochs = stage["num_epochs"]
        
        # Skip stages that have already been completed
        if current_stage > stage_num:
            logging.info(f"Skipping {stage_name} as it has already been completed.")
            continue
        elif current_stage < stage_num:
            # Starting a new stage
            current_stage = stage_num
            start_epoch = 0
            starting_batch = None  # Reset starting_batch for new stage
        
        logging.info(f"Starting {stage_name}...")
        print(f"Starting {stage_name}...")

        best_val_loss = float('inf')       # Initialize best validation loss to infinity
        best_val_accuracy = 0.0            # Initialize best validation accuracy to zero

        # Load dataset
        try:
            dataset = load_dataset_from_file(dataset_path)
        except FileNotFoundError as e:
            logging.error(str(e))
            continue
        
        # Apply dataset percentage
        subset_size = int(len(dataset) * args.dataset_percentage)
        if subset_size < 1:
            logging.warning(f"Dataset percentage {args.dataset_percentage} is too small. Using full dataset.")
            subset_size = len(dataset)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        subset = Subset(dataset, subset_indices)
        
        # Split into training and validation
        train_subset, val_subset = split_dataset(subset, args.validation_frequency)
        logging.info(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
        print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
        
        # Create DataLoaders
        train_loader, val_loader = create_dataloaders(train_subset, val_subset, config, tokenizer)
        
        # Initialize scheduler if it's not set
        if scheduler is None:
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(0.1 * total_steps), 
                num_training_steps=total_steps
            )
        
        # Initialize loss tracking variables
        train_loss_window = []
        val_loss_window = []

        # Training loop for the current stage
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            logging.info(f"{stage_name} - Epoch {epoch +1}/{num_epochs}")
            print(f"{stage_name} - Epoch {epoch +1}/{num_epochs}")
            
            # Create an iterator for the DataLoader
            train_loader_iter = iter(train_loader)
            
            # If resuming from a periodic checkpoint, skip batches
            if epoch == start_epoch and starting_batch is not None:
                logging.info(f"Resuming from batch {starting_batch} in epoch {epoch +1}")
                print(f"Resuming from batch {starting_batch} in epoch {epoch +1}")
                try:
                    for _ in range(starting_batch):
                        next(train_loader_iter)
                        batch_count +=1
                except StopIteration:
                    logging.warning(f"Starting batch {starting_batch} exceeds the number of batches in the DataLoader.")
                    starting_batch = None  # Reset if batch number is too high
            
            for batch in tqdm(train_loader_iter, desc=f"Training {stage_name} Epoch {epoch +1}"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
                
                # Compute custom loss
                loss = custom_loss_fn(logits, labels, stage_num)
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                logging.info(f"Stage {stage_num}, Epoch {epoch + 1}, Batch {batch_count + 1}, Training Loss: {loss.item():.4f}")

                # Update loss tracking
                train_loss_window.append(loss.item())
                if len(train_loss_window) > 100:
                    train_loss_window.pop(0)
                
                # Log the raw training loss for the current batch
                logging.info(f"Stage {stage_num}, Epoch {epoch + 1}, Batch {batch_count + 1}, Training Loss: {loss.item():.4f}")

                epoch_loss += loss.item()
                batch_count +=1

                # Log training loss to the log file
                avg_train_loss = sum(train_loss_window) / len(train_loss_window) if train_loss_window else 0.0
                logging.info(f"Stage {stage_num}, Epoch {epoch + 1}, Batch {batch_count}, Training Loss (avg over last 100): {avg_train_loss:.4f}")


                # Perform validation at specified frequency
                if batch_count % args.validation_frequency ==0:
                    avg_train_loss = sum(train_loss_window) / len(train_loss_window) if train_loss_window else 0.0
                    logging.info(f"Stage {stage_num}, Epoch {epoch +1}, Batch {batch_count}, Training Loss (avg over last 100): {avg_train_loss:.4f}")
                    print(f"Stage {stage_num}, Epoch {epoch +1}, Batch {batch_count}, Training Loss (avg over last 100): {avg_train_loss:.4f}")
                    
                    # Perform validation on a single sample
                    avg_val_loss, accuracy = validate(model, tokenizer, val_subset, device, config, stage_num, pad_token_id)
                    
                    # Update validation loss window
                    val_loss_window.append(avg_val_loss)
                    if len(val_loss_window) > 100:
                        val_loss_window.pop(0)
                    
                    # Compute average validation loss
                    avg_val_loss_display = sum(val_loss_window) / len(val_loss_window) if val_loss_window else 0.0
                    logging.info(f"Stage {stage_num}, Epoch {epoch +1}, Batch {batch_count}, Validation Loss (avg over last 100): {avg_val_loss_display:.4f}")
                    print(f"Stage {stage_num}, Epoch {epoch +1}, Batch {batch_count}, Validation Loss (avg over last 100): {avg_val_loss_display:.4f}")
                    
                    # Check and save the best model based on validation loss
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_checkpoint(model, optimizer, scheduler, epoch, stage_num, checkpoint_dir, best=True)
                        logging.info("Best model updated based on validation loss.")
                    
                    # Optionally, track best accuracy
                    if accuracy > best_val_accuracy:
                        best_val_accuracy = accuracy
                        save_checkpoint(model, optimizer, scheduler, epoch, stage_num, checkpoint_dir, best_accuracy=True)
                        logging.info("Best model updated based on validation accuracy.")

                # Inside the batch loop, after updating batch_count
                if checkpoint_frequency is not None and batch_count % checkpoint_frequency == 0:
                    save_checkpoint(
                        model, 
                        optimizer, 
                        scheduler, 
                        epoch, 
                        stage_num, 
                        checkpoint_dir, 
                        periodic=True, 
                        batch=batch_count
                    )
                    logging.info(f"Periodic checkpoint saved at epoch {epoch}, batch {batch_count}.")
                    print(f"Periodic checkpoint saved at epoch {epoch}, batch {batch_count}.")

            # End of epoch
            avg_epoch_loss = epoch_loss / batch_count if batch_count >0 else 0.0
            logging.info(f"Stage {stage_num}, Epoch {epoch +1}, Average Training Loss: {avg_epoch_loss:.4f}")
            print(f"Stage {stage_num} - Epoch {epoch +1}: Average Training Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint at the end of the epoch
            save_checkpoint(model, optimizer, scheduler, epoch, stage_num, checkpoint_dir)
        
        # After finishing all epochs for the current stage
        logging.info(f"Completed {stage_name}.")
        print(f"Completed {stage_name}.")



if __name__ == "__main__":
    main()
