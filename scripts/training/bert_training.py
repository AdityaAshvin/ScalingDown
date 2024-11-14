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


def generate_training_graph(training_losses, mse_losses, training_graph_path):
    # Check if there are training losses to plot
    if len(training_losses) == 0:
        print("No training losses to plot.")
        return

    steps = range(1, len(training_losses) + 1)

    # Create the primary plot for Training Loss
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss', color=color1)
    ax1.plot(steps, training_losses, label='Training Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    # If mse_losses is provided, plot it on the secondary y-axis
    if mse_losses is not None:
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

        color2 = 'tab:red'
        ax2.set_ylabel('Hidden-layer Loss', color=color2)
        ax2.plot(steps, mse_losses, label='Hidden-layer Loss', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    else:
        # If no mse_losses, just add the legend for training loss
        ax1.legend(loc='upper right')

    plt.title('Loss Over Steps')

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
    teacher_model_names = [
        "bert-base-uncased",                   # Existing Teacher
        "ProsusAI/finbert",                    # FinBert
        "microsoft/deberta-large-mnli"         # DeBERTa
    ]

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

    # Initialize student tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    student_model = model_class.from_pretrained(student_model_name, num_labels=num_labels).to(device)

    # Initialize teacher models and tokenizers
    teacher_models = []
    projection_layers = []
    for teacher_name in teacher_model_names:
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        teacher_model = model_class.from_pretrained(teacher_name, num_labels=num_labels).to(device)
        teacher_model.eval()
        teacher_models.append(teacher_model)

        # Add a projection layer if hidden sizes differ
        if teacher_model.config.hidden_size != student_model.config.hidden_size:
            projection_layer = nn.Linear(
                teacher_model.config.hidden_size,
                student_model.config.hidden_size,
                bias=False
            ).to(device)
        else:
            projection_layer = None
        projection_layers.append(projection_layer)

    logger.info(f"Loaded {len(teacher_models)} teacher models.")

    if hasattr(student_model.config, 'id2label'):
        print(f"DistilBert Label Mapping:", teacher_models[1].config.id2label)
    else:
        print(f"student_model model does not have id2label mapping defined.")
    for t_idx, teacher_model in enumerate(teacher_models):
        if hasattr(teacher_model.config, 'id2label'):
            print(f"{teacher_model_names[t_idx]} Label Mapping:", teacher_models[1].config.id2label)
        else:
            print(f"{teacher_model_names[t_idx]} model does not have id2label mapping defined.")

    # Enable gradient checkpointing if needed
    if hyperparams['gradient_checkpointing']:
        student_model.gradient_checkpointing_enable()

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define custom trainer with knowledge distillation
    class CustomTrainer(Trainer):
        def __init__(self, *args, teacher_models=None, projection_layers=None, hidden_weight=0.5, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_models = [tm.to(self.args.device) for tm in teacher_models]
            for tm in self.teacher_models:
                tm.eval()
            self.projection_layers = projection_layers
            self.hidden_weight = hidden_weight

            # For collecting MSE losses
            self.mse_losses = []
            self.kd_losses = []
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
                total_kd_loss = 0.0
                total_mse_loss = 0.0
                num_teachers = len(self.teacher_models)

                for idx, teacher in enumerate(self.teacher_models):
                    with torch.no_grad():
                        teacher_outputs = teacher(
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
                    projection_layer = self.projection_layers[idx]
                    if projection_layer is not None:
                        teacher_hidden_states = projection_layer(teacher_hidden_states)

                    # Compute MSE loss between student and teacher hidden states
                    mse_loss = F.mse_loss(student_hidden_states, teacher_hidden_states)

                    total_kd_loss += kd_loss
                    total_mse_loss += mse_loss

                # Average the losses over the number of teachers
                avg_kd_loss = total_kd_loss / num_teachers
                avg_mse_loss = total_mse_loss / num_teachers

                # Total loss
                total_loss = student_loss + self.hidden_weight * avg_kd_loss + self.hidden_weight * avg_mse_loss

                self.log(
                    {
                        "student_loss": student_loss.detach().item(),
                        "kd_loss": avg_kd_loss.detach().item(),
                        "mse_loss": avg_mse_loss.detach().item()
                    }
                )

                # Collect losses
                self.kd_losses.append(avg_kd_loss.detach().item())
                self.mse_losses.append(avg_mse_loss.detach().item())
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
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_models=teacher_models,
        projection_layers=projection_layers,
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

    # Evaluate the student & teacher models on validation data
    logger.info("Evaluation on student and teacher models started.")
    print("Evaluating student and teacher models on validation data...")

    student_predictions = []
    teacher_predictions_list = [[] for _ in teacher_models]  # List of lists
    label_answers = []

    # Change student model to eval mode
    student_model.eval()

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

            # Teacher models predictions
            for idx, teacher in enumerate(teacher_models):
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
                teacher_pred_label = torch.argmax(teacher_logits, dim=-1).cpu().numpy().item()
                teacher_predictions_list[idx].append(teacher_pred_label)

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
    teacher_metrics = []
    for idx, teacher_preds in enumerate(teacher_predictions_list):
        teacher_accuracy = accuracy_score(label_answers, teacher_preds)
        teacher_precision, teacher_recall, teacher_f1, _ = precision_recall_fscore_support(
            label_answers, teacher_preds, average='weighted')
        teacher_metrics.append({
            'accuracy': teacher_accuracy,
            'precision': teacher_precision,
            'recall': teacher_recall,
            'f1': teacher_f1
        })

    # Write report
    with open(output_report_path, 'w') as f:
        f.write("Training complete.\n\n")

        # Number of examples trained
        f.write(f"Number of training examples: {len(train_dataset)}\n")
        f.write(f"Number of validation examples: {len(val_dataset)}\n\n")

        # Training metrics
        f.write("Training Metrics:\n")
        f.write(f"Total training time: {train_runtime:.2f} seconds\n")
        f.write(f"Training samples per second: {train_result.metrics.get('train_samples_per_second', 'N/A')}\n")
        f.write(f"Training steps per second: {train_result.metrics.get('train_steps_per_second', 'N/A')}\n")
        f.write(f"Final training loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}\n")
        f.write(f"Epochs completed: {train_result.metrics.get('epoch', 'N/A'):.1f}\n\n")

        # Extract training metrics from loss_collector and trainer
        training_losses = loss_collector.student_losses
        mse_losses = trainer.mse_losses  # Collected in CustomTrainer
        kd_losses = trainer.kd_losses
        steps = trainer.steps

        # Training progress evaluation
        if len(training_losses) > 0:
            f.write("Training Progress Evaluation:\n")
            f.write("Step | Training Loss | KL Divergence Loss | MSE Loss\n")
            f.write("-------------------------------------------------------\n")
            for i in range(len(training_losses)):
                step = steps[i]
                train_loss = training_losses[i]
                kd_loss = kd_losses[i]
                mse_loss = mse_losses[i]
                f.write(f"{step:<5} | {train_loss:<13.4f} | {kd_loss:<18.4f} | {mse_loss:<8.4f}\n")
            f.write("\nSee training_graph.png for visualization of training loss.\n\n")
        else:
            f.write("No training metrics available.\n\n")

        # Student evaluation
        f.write("\nStudent Model Evaluation on Validation Data:\n")
        f.write(f"Accuracy: {student_accuracy * 100:.2f}%\n")
        f.write(f"Precision: {student_precision * 100:.2f}%\n")
        f.write(f"Recall: {student_recall * 100:.2f}%\n")
        f.write(f"F1 Score: {student_f1 * 100:.2f}%\n\n")

        # Teacher evaluations
        for idx, metrics in enumerate(teacher_metrics):
            f.write(f"Teacher Model {idx + 1} ({teacher_model_names[idx]}) Evaluation on Validation Data:\n")
            f.write(f"Accuracy: {metrics['accuracy'] * 100:.2f}%\n")
            f.write(f"Precision: {metrics['precision'] * 100:.2f}%\n")
            f.write(f"Recall: {metrics['recall'] * 100:.2f}%\n")
            f.write(f"F1 Score: {metrics['f1'] * 100:.2f}%\n\n")

        # Write validation examples
        num_examples_to_show = min(30, len(val_dataset))
        f.write(f"\nFirst {num_examples_to_show} Validation Examples:\n")
        label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        for idx in range(num_examples_to_show):
            input_ids = val_dataset[idx]['input_ids']
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            actual_label = label_mapping[label_answers[idx]]
            student_pred_label = label_mapping[student_predictions[idx]]
            teacher_preds = [label_mapping[teacher_predictions_list[t_idx][idx]] for t_idx in range(len(teacher_models))]

            f.write(f"\nExample {idx + 1}:\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Actual Label: {actual_label}\n")
            f.write(f"Student Predicted Label: {student_pred_label}\n")
            for t_idx, teacher_pred in enumerate(teacher_preds):
                f.write(f"Teacher {t_idx + 1} Predicted Label: {teacher_pred}\n")

    print(f"Training complete. Report saved to {output_report_path}")
    logger.info(f"Training complete. Report saved to {output_report_path}")

    generate_training_graph(training_losses, mse_losses, training_graph_path)


if __name__ == "__main__":
    main()
