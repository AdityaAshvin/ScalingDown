# evaluate_models.py

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk
import argparse
import yaml
import os
import logging
from tqdm import tqdm
import re
import numpy as np
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Teacher and Student Models on Validation Dataset.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--teacher_model_name", type=str, default="google/flan-t5-xl", help="Pre-trained teacher model name.")
    parser.add_argument("--student_model_name", type=str, default="google/flan-t5-large", help="Pre-trained student model name.")
    parser.add_argument("--student_checkpoint_dir", type=str, required=True, help="Directory path to the trained student model checkpoints.")
    parser.add_argument("--student_checkpoint_file", type=str, default=None, help="Specific student checkpoint file to load (optional).")
    parser.add_argument("--validation_dataset_path", type=str, required=True, help="Path to the preprocessed validation dataset.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the evaluation on.")
    parser.add_argument("--log_file", type=str, default=None, help="File path for logging.")
    
    # Unified argument for evaluation percentage
    parser.add_argument("--evaluation_percentage", type=float, default=100.0, help="Percentage of the validation set to evaluate all models on (0-100).")
    
    # Optional seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # New arguments for controlling sample printing
    parser.add_argument("--print_all_samples", action='store_true', help="If set, prints all evaluation samples.")
    parser.add_argument("--print_every_n_samples", type=int, default=100, help="Print a summary every N samples.")
    
    return parser.parse_args()




def sample_dataset(dataset, percentage, seed):
    """
    Samples a percentage of the dataset.

    Args:
        dataset (Dataset): The dataset to sample from.
        percentage (float): The percentage of the dataset to sample (0-100).
        seed (int): Random seed for reproducibility.

    Returns:
        Dataset: The sampled subset of the dataset.
    """
    if not (0 < percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")

    total_samples = len(dataset)
    num_samples = int((percentage / 100.0) * total_samples)

    sampled_indices = set(np.random.RandomState(seed).choice(list(range(total_samples)), size=num_samples, replace=False))
    sampled_dataset = dataset.select(list(sampled_indices))
    return sampled_dataset



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

def load_tokenizer(tokenizer_path):
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_model(model_name, tokenizer, device):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return model

def get_latest_checkpoint(checkpoint_dir):
    # Pattern to match checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-epoch*-batch*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        # If no milestone checkpoints found, try to load the default checkpoint.pth
        default_checkpoint = os.path.join(checkpoint_dir, "checkpoint.pth")
        if os.path.exists(default_checkpoint):
            return default_checkpoint
        else:
            return None

    # Sort checkpoint files based on epoch and batch number
    def extract_numbers(file_path):
        basename = os.path.basename(file_path)
        match = re.search(r"checkpoint-epoch(\d+)-batch(\d+).pth", basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        else:
            return 0, 0

    checkpoint_files_sorted = sorted(checkpoint_files, key=extract_numbers, reverse=True)
    return checkpoint_files_sorted[0]  # Return the latest checkpoint

def load_trained_student_model(student_checkpoint_dir, student_checkpoint_file, tokenizer, device):
    if student_checkpoint_file:
        specific_checkpoint = os.path.join(student_checkpoint_dir, student_checkpoint_file)
        if os.path.exists(specific_checkpoint):
            checkpoint_path = specific_checkpoint
            logging.info(f"Loading trained student model from specified checkpoint: {checkpoint_path}")
        else:
            logging.error(f"Specified student checkpoint does not exist: {specific_checkpoint}")
            exit(1)
    else:
        latest_student_checkpoint = get_latest_checkpoint(student_checkpoint_dir)
        if latest_student_checkpoint:
            checkpoint_path = latest_student_checkpoint
            logging.info(f"Loading trained student model from latest checkpoint: {checkpoint_path}")
        else:
            logging.error("No student checkpoint found. Please ensure that checkpoints exist in the specified directory.")
            exit(1)
    
    # Initialize student model architecture
    student_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    student_model.resize_token_embeddings(len(tokenizer))
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']
    student_model.load_state_dict(state_dict)
    
    student_model.to(device)
    student_model.eval()
    logging.info("Trained student model loaded and set to evaluation mode.")
    return student_model

def load_validation_dataset(validation_dataset_path):
    validation_dataset = load_from_disk(validation_dataset_path)
    logging.info(f"Loaded validation dataset from {validation_dataset_path}.")
    return validation_dataset

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

def evaluate_model(model, tokenizer, validation_loader, device, pad_token_id, model_name="Model", print_all_samples=False, print_every_n_samples=100):
    total_correct = 0
    total_samples = 0
    samples_printed = 0  # To control sample printing
    
    with torch.no_grad():
        pbar = tqdm(validation_loader, desc=f"Evaluating {model_name}", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            input_texts = batch['input_texts']  # Original questions with options
            label_texts = batch['label_texts']  # Original correct answers

            # Generate predictions
            generated_ids = model.generate(
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

            # Compute accuracy
            for input_text, pred, label_text in zip(input_texts, preds, label_texts):
                pred_norm = normalize_text(pred)
                label_norm = normalize_text(label_text)
                if pred_norm == label_norm:
                    total_correct += 1
                total_samples += 1

                # Dynamic Accuracy Update
                current_accuracy = total_correct / total_samples
                pbar.set_postfix({'Accuracy': f'{current_accuracy:.4f}'})

                # Print samples based on user preference
                if print_all_samples:
                    print("\nEvaluation Sample:")
                    print(f"Question: {input_text}")
                    print(f"Correct Answer: {label_text}")
                    print(f"{model_name}'s Response: {pred}")
                    print(f"Normalized {model_name}'s Response: {pred_norm}")
                    print(f"Normalized Correct Answer: {label_norm}")
                elif print_every_n_samples and total_samples % print_every_n_samples == 0:
                    print(f"\nProcessed {total_samples} samples. Current Accuracy: {current_accuracy:.4f}")

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy


def create_validation_dataloader(validation_dataset, batch_size, tokenizer):
    def collate_fn(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100).long()
        input_texts = [item['input_text'] for item in batch]
        label_texts = [item['label_text'] for item in batch]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_texts': input_texts,
            'label_texts': label_texts
        }
    
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count()
    )
    logging.info(f"Created DataLoader for validation dataset with batch size {batch_size}.")
    return validation_loader

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(args.log_file)
    logging.info("Starting evaluation of Teacher and Student models.")
    
    # Load tokenizer
    tokenizer = load_tokenizer(config["tokenizer"]["save_dir"])
    pad_token_id = tokenizer.pad_token_id
    logging.info("Loaded tokenizer.")
    
    # Load pre-trained teacher model
    teacher_model = load_model(args.teacher_model_name, tokenizer, args.device)
    
    # Load untrained student model (pre-trained)
    student_model = load_model(args.student_model_name, tokenizer, args.device)
    logging.info(f"Loaded untrained student model '{args.student_model_name}' and set to evaluation mode.")
    
    # Load validation dataset
    validation_dataset = load_validation_dataset(args.validation_dataset_path)
    
    # Validate that the evaluation percentage is within bounds
    if not (0 < args.evaluation_percentage <= 100):
        logging.error("Evaluation percentage must be between 0 and 100.")
        exit(1)
    
    # Inform about unused data if evaluation percentage is less than 100%
    if args.evaluation_percentage < 100.0:
        remaining = 100.0 - args.evaluation_percentage
        logging.info(f"{remaining}% of the validation set will remain unused.")
    
    # Sample a subset of the validation dataset
    try:
        evaluation_subset = sample_dataset(validation_dataset, args.evaluation_percentage, args.seed)
    except ValueError as e:
        logging.error(f"Sampling Error: {e}")
        exit(1)
    
    # Create a single DataLoader for the evaluation subset
    evaluation_loader = create_validation_dataloader(evaluation_subset, config.get("evaluation", {}).get("batch_size", 16), tokenizer)
    
    # Evaluate Untrained Teacher Model
    logging.info("Evaluating Untrained Teacher model.")
    teacher_accuracy = evaluate_model(
        teacher_model,
        tokenizer,
        evaluation_loader,
        args.device,
        pad_token_id,
        model_name="Untrained Teacher",
        print_all_samples=args.print_all_samples,
        print_every_n_samples=args.print_every_n_samples
    )
    logging.info(f"Untrained Teacher Model Accuracy on {args.evaluation_percentage}% of Validation Set: {teacher_accuracy:.4f}")
    
    # Evaluate Untrained Student Model
    logging.info("Evaluating Untrained Student model.")
    student_accuracy = evaluate_model(
        student_model,
        tokenizer,
        evaluation_loader,
        args.device,
        pad_token_id,
        model_name="Untrained Student",
        print_all_samples=args.print_all_samples,
        print_every_n_samples=args.print_every_n_samples
    )
    logging.info(f"Untrained Student Model Accuracy on {args.evaluation_percentage}% of Validation Set: {student_accuracy:.4f}")
    
    # Load Trained Student Model from Checkpoint
    logging.info("Loading Trained Student model from checkpoint.")
    trained_student_model = load_trained_student_model(
        args.student_checkpoint_dir,
        args.student_checkpoint_file,
        tokenizer,
        args.device
    )
    
    # Evaluate Trained Student Model
    logging.info("Evaluating Trained Student model.")
    trained_student_accuracy = evaluate_model(
        trained_student_model,
        tokenizer,
        evaluation_loader,
        args.device,
        pad_token_id,
        model_name="Trained Student",
        print_all_samples=args.print_all_samples,
        print_every_n_samples=args.print_every_n_samples
    )
    logging.info(f"Trained Student Model Accuracy on {args.evaluation_percentage}% of Validation Set: {trained_student_accuracy:.4f}")
    
    # Display Accuracies
    logging.info("Evaluation Complete.")
    print(f"\n=== Evaluation Results ===")
    print(f"Untrained Teacher Model Accuracy ({args.evaluation_percentage}%): {teacher_accuracy:.4f}")
    print(f"Untrained Student Model Accuracy ({args.evaluation_percentage}%): {student_accuracy:.4f}")
    print(f"Trained Student Model Accuracy ({args.evaluation_percentage}%): {trained_student_accuracy:.4f}")




if __name__ == "__main__":
    main()
