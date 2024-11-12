# bert_training.py

import argparse
import torch
import gc
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import os
import logging
import random
import numpy as np
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from datasets import Dataset
from scripts.training.callback import LossCollectorCallback

# Set up logging
logging.basicConfig(
    filename='bert_training_progress.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="Train model on specified dataset")
    parser.add_argument('--data_portion', type=float, default=1.0, help="Portion of dataset to use (e.g., 0.01 for 1%)")
    parser.add_argument('--output_report', type=str, default='',
                        help="Directory to save the output report")
    parser.add_argument('--dataset', type=str, default='phrasebank',
                        choices=['phrasebank'],
                        help="Dataset to use: 'phrasebank'")
    args = parser.parse_args()
    return args


def generate_training_graph(training_losses, mse_losses,
                            training_graph_path):
    # Plotting training loss against steps
    if len(training_losses) == 0:
        print("No training losses to plot.")
        return

    steps = range(1, len(training_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, training_losses, label='Training Loss')
    if mse_losses is not None:
        plt.plot(steps, mse_losses, label='Hidden-layer Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(training_graph_path)
    plt.close()
    print(f"Saved training loss graph in {training_graph_path}")


def main():
    logger.info("Starting script...")
    # Set random seed
    set_seed(42)

    # Parse arguments
    args = parse_args()
    data_portion = args.data_portion
    output_report_dir = args.output_report
    dataset_name = args.dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_report_path = os.path.join(output_report_dir, f"{timestamp}-training_report.txt")
    training_graph_path = os.path.join(output_report_dir, f"{timestamp}-training_graph.png")

    # Get device
    device = get_device()

    # Load hyperparameters from config file
    config_path = os.path.join('../../config', 'hyperparameters.yaml')
    with open(config_path, 'r') as f:
        hyperparams = yaml.safe_load(f)

    # Ensure correct data types
    hyperparams['learning_rate'] = float(hyperparams['learning_rate'])
    hyperparams['num_train_epochs'] = int(hyperparams['num_train_epochs'])
    hyperparams['per_device_train_batch_size'] = int(hyperparams['per_device_train_batch_size'])
    hyperparams['per_device_eval_batch_size'] = int(hyperparams['per_device_eval_batch_size'])
    hyperparams['eval_steps'] = int(hyperparams['eval_steps'])
    hyperparams['save_steps'] = int(hyperparams['save_steps'])
    hyperparams['logging_steps'] = int(hyperparams['logging_steps'])
    hyperparams['save_total_limit'] = int(hyperparams['save_total_limit'])
    hyperparams['load_best_model_at_end'] = bool(hyperparams['load_best_model_at_end'])
    hyperparams['greater_is_better'] = bool(hyperparams['greater_is_better'])
    hyperparams['save_strategy'] = hyperparams.get('save_strategy', 'steps')
    hyperparams['bf16'] = bool(hyperparams['bf16'])
    hyperparams['fp16'] = bool(hyperparams['fp16'])
    hyperparams['gradient_accumulation_steps'] = int(hyperparams['gradient_accumulation_steps'])
    hyperparams['gradient_checkpointing'] = bool(hyperparams['gradient_checkpointing'])
    hyperparams['eval_accumulation_steps'] = int(hyperparams['eval_accumulation_steps'])
    hyperparams['hidden_weight'] = float(hyperparams.get('hidden_weight', 0.5))

    logger.info("Loaded hyperparameters from config file.")

    # Load data
    from scripts.data_preprocessing.data_preprocessing_bert_phrasebank import get_preprocessed_data as get_phrasebank_data
    train_dataset, val_dataset, test_dataset = get_phrasebank_data(save_dir='')

    num_labels = 3  # Negative, Neutral, Positive
    tokenizer_name = "distilbert-base-uncased"
    student_model_name = "distilbert-base-uncased"
    teacher_model_name = "bert-base-uncased"  # You can use BERT as teacher

    model_class = AutoModelForSequenceClassification

    logger.info(f"Loaded datasets for {dataset_name}.")

    # Use portion of dataset
    if data_portion < 1.0:
        num_train_examples = max(5, int(len(train_dataset) * data_portion))
        num_val_examples = max(10, int(len(val_dataset) * data_portion))
        train_dataset = train_dataset.select(range(num_train_examples))
        val_dataset = val_dataset.select(range(num_val_examples))

    logger.info(f"Using {len(train_dataset)} training examples after applying data_portion={data_portion}.")
    logger.info(f"Using {len(val_dataset)} validation examples after applying data_portion={data_portion}.")

    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after applying data_portion. Please use a larger data_portion.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty after applying data_portion. Please use a larger data_portion.")

    # Initialize teacher and student models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    teacher_model = model_class.from_pretrained(teacher_model_name, num_labels=num_labels).to(device)
    teacher_model.eval()

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_model = model_class.from_pretrained(student_model_name, num_labels=num_labels).to(device)

    # Enable gradient checkpointing if needed
    if hyperparams['gradient_checkpointing']:
        student_model.gradient_checkpointing_enable()

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=student_tokenizer)

    # Define custom trainer with knowledge distillation
    class CustomTrainer(Trainer):
        def __init__(self, *args, teacher_model=None, hidden_weight=0.5, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = teacher_model.to(self.args.device)
            self.teacher_model.eval()
            self.hidden_weight = hidden_weight

            # Add a projection layer if hidden sizes differ
            if self.teacher_model.config.hidden_size != self.model.config.hidden_size:
                self.projection_layer = nn.Linear(
                    self.teacher_model.config.hidden_size,
                    self.model.config.hidden_size,
                    bias=False
                ).to(self.args.device)
            else:
                self.projection_layer = None

            # For collecting MSE loss
            self.mse_losses = []
            self.steps = []

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Move inputs to the same device as the model
            current_device = next(model.parameters()).device
            inputs = {k: v.to(current_device) for k, v in inputs.items()}
            labels = inputs.get("labels")

            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            student_loss = outputs.loss
            student_logits = outputs.logits
            student_hidden_states = outputs.hidden_states[-1]

            if model.training:
                with torch.no_grad():
                    # Move teacher model to the same device as student model
                    self.teacher_model.to(current_device)
                    teacher_outputs = self.teacher_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        output_hidden_states=True,
                    )
                    teacher_logits = teacher_outputs.logits
                    teacher_hidden_states = teacher_outputs.hidden_states[-1]

                # Compute KL divergence loss between student and teacher logits
                kd_loss = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(student_logits / 1.0, dim=-1),
                    F.softmax(teacher_logits / 1.0, dim=-1)
                )

                # If hidden sizes differ, project teacher hidden states
                if self.projection_layer is not None:
                    teacher_hidden_states = self.projection_layer(teacher_hidden_states)

                # Compute MSE loss between student and teacher hidden states
                mse_loss = F.mse_loss(student_hidden_states, teacher_hidden_states)

                # Total loss
                total_loss = student_loss + self.hidden_weight * kd_loss + self.hidden_weight * mse_loss

                self.log(
                    {
                        "student_loss": student_loss.detach().item(),
                        "kd_loss": kd_loss.detach().item(),
                        "mse_loss": mse_loss.detach().item()
                    }
                )

                # Collect losses
                self.mse_losses.append(mse_loss.detach().item())
                self.steps.append(self.state.global_step)

            else:
                # During evaluation, only compute student_loss
                total_loss = student_loss

            return (total_loss, outputs) if return_outputs else total_loss

    # Training arguments from hyperparameters
    training_args = TrainingArguments(
        output_dir=f'./{dataset_name}_student_model',
        num_train_epochs=hyperparams['num_train_epochs'],
        per_device_train_batch_size=hyperparams['per_device_train_batch_size'],
        per_device_eval_batch_size=hyperparams['per_device_eval_batch_size'],
        learning_rate=hyperparams['learning_rate'],
        eval_strategy=hyperparams['evaluation_strategy'],
        eval_steps=hyperparams['eval_steps'],
        save_strategy=hyperparams['save_strategy'],
        save_steps=hyperparams['save_steps'],
        logging_steps=hyperparams['logging_steps'],
        save_total_limit=hyperparams['save_total_limit'],
        load_best_model_at_end=hyperparams['load_best_model_at_end'],
        metric_for_best_model=hyperparams['metric_for_best_model'],
        greater_is_better=hyperparams['greater_is_better'],
        gradient_accumulation_steps=hyperparams['gradient_accumulation_steps'],
        fp16=hyperparams['fp16'],
        bf16=hyperparams['bf16'],
        report_to='none',
        gradient_checkpointing=hyperparams['gradient_checkpointing'],
        eval_accumulation_steps=hyperparams['eval_accumulation_steps'],
    )

    loss_collector = LossCollectorCallback()
    hidden_weight = hyperparams.get('hidden_weight', 0.5)

    # Initialize trainer
    trainer = CustomTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=student_tokenizer,
        data_collator=data_collator,
        teacher_model=teacher_model,
        hidden_weight=hidden_weight,
        compute_metrics=None,  # We will compute metrics manually
        callbacks=[loss_collector],
    )

    # Training
    print("Starting model training...")
    logger.info("Starting model training...")
    start_time = time.time()
    train_result = trainer.train()
    train_runtime = time.time() - start_time

    # Save the model
    trainer.save_model(f'./{dataset_name}_student_model')
    logger.info(f"Model saved to './{dataset_name}_student_model'.")

    # Evaluate the student model on validation data
    logger.info("Evaluation on student model started.")
    print("Evaluating student model on validation data...")
    student_model.eval()
    student_predictions = []
    label_answers = []

    # Also collect teacher predictions for comparison
    teacher_predictions = []

    for example in tqdm(val_dataset, desc="Evaluating student and teacher models"):
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)
        labels = example['labels']

        with torch.no_grad():
            # Student model prediction
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits
            student_pred_label = torch.argmax(student_logits, dim=-1).cpu().numpy().item()
            student_predictions.append(student_pred_label)

            # Teacher model prediction
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
            teacher_pred_label = torch.argmax(teacher_logits, dim=-1).cpu().numpy().item()
            teacher_predictions.append(teacher_pred_label)

            label_answers.append(labels)

        # Free up memory
        del input_ids, attention_mask, student_outputs, teacher_outputs
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Evaluation completed.")

    # Compute student metrics
    student_accuracy = accuracy_score(label_answers, student_predictions)
    student_precision, student_recall, student_f1, _ = precision_recall_fscore_support(
        label_answers, student_predictions, average='weighted')

    # Compute teacher metrics
    teacher_accuracy = accuracy_score(label_answers, teacher_predictions)
    teacher_precision, teacher_recall, teacher_f1, _ = precision_recall_fscore_support(
        label_answers, teacher_predictions, average='weighted')

    # Write report
    with open(output_report_path, 'w') as f:
        f.write("Training complete.\n\n")

        # Number of examples trained
        f.write(f"Number of training examples: {len(train_dataset)}\n")
        f.write(f"Number of validation examples: {len(val_dataset)}\n\n")

        # Training metrics
        f.write("Training Metrics:\n")
        f.write(f"Total training time: {train_runtime:.2f} seconds\n")
        f.write(f"Training samples per second: {train_result.metrics['train_samples_per_second']:.3f}\n")
        f.write(f"Training steps per second: {train_result.metrics['train_steps_per_second']:.3f}\n")
        f.write(f"Final training loss: {train_result.metrics['train_loss']:.4f}\n")
        f.write(f"Epochs completed: {train_result.metrics['epoch']:.1f}\n\n")

        # Extract training metrics from loss_collector and trainer
        training_losses = loss_collector.student_losses
        mse_losses = trainer.mse_losses  # Collected in CustomTrainer
        steps = trainer.steps

        # Training progress evaluation
        if len(training_losses) > 0:
            f.write("Training Progress Evaluation:\n")
            f.write("Step | Training Loss | Hidden-layer Loss\n")
            f.write("------------------------------------------\n")
            for i in range(len(training_losses)):
                step = steps[i]
                train_loss = training_losses[i]
                mse_loss = mse_losses[i]
                f.write(f"{step:<5} | {train_loss:<13} | {mse_loss:<16}\n")
            f.write("\nSee 'loss_curve.png' for visualization of training loss.\n\n")
        else:
            f.write("No training metrics available.\n\n")

        # Student evaluation
        f.write("\nStudent Model Evaluation on Validation Data:\n")
        f.write(f"Accuracy: {student_accuracy * 100:.2f}%\n")
        f.write(f"Precision: {student_precision * 100:.2f}%\n")
        f.write(f"Recall: {student_recall * 100:.2f}%\n")
        f.write(f"F1 Score: {student_f1 * 100:.2f}%\n\n")

        # Teacher evaluation
        f.write("Teacher Model Evaluation on Validation Data:\n")
        f.write(f"Accuracy: {teacher_accuracy * 100:.2f}%\n")
        f.write(f"Precision: {teacher_precision * 100:.2f}%\n")
        f.write(f"Recall: {teacher_recall * 100:.2f}%\n")
        f.write(f"F1 Score: {teacher_f1 * 100:.2f}%\n\n")

        # Write validation examples
        num_examples_to_show = min(30, len(val_dataset))
        f.write(f"\nFirst {num_examples_to_show} Validation Examples:\n")
        label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        for idx in range(num_examples_to_show):
            input_ids = val_dataset[idx]['input_ids']
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            actual_label = label_mapping[label_answers[idx]]
            student_pred_label = label_mapping[student_predictions[idx]]
            teacher_pred_label = label_mapping[teacher_predictions[idx]]

            f.write(f"\nExample {idx + 1}:\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Actual Label: {actual_label}\n")
            f.write(f"Student Predicted Label: {student_pred_label}\n")
            f.write(f"Teacher Predicted Label: {teacher_pred_label}\n")

    print(f"Training complete. Report saved to {output_report_path}")
    logger.info(f"Training complete. Report saved to {output_report_path}")

    generate_training_graph(training_losses, mse_losses, training_graph_path)


if __name__ == "__main__":
    main()