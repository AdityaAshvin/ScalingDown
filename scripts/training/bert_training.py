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
    EarlyStoppingCallback,
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
from scripts.training.callback import LossCollectorCallback, EvalLossCollectorCallback

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

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def generate_training_graph(training_losses, mse_losses, training_steps, training_graph_path, window_size=1, eval_losses=None, eval_steps=None):
    # Check if there are training losses to plot
    if len(training_losses) == 0:
        print("No training losses to plot.")
        return

    if window_size > 1:
        training_losses_smooth = moving_average(training_losses, window_size)
        mse_losses_smooth = moving_average(mse_losses, window_size) if mse_losses is not None else None
        steps = training_steps[window_size - 1:]
    else:
        training_losses_smooth = training_losses
        mse_losses_smooth = mse_losses
        steps = training_steps

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(steps, training_losses_smooth, label='Training Loss', color='tab:blue', alpha=0.7)

    # Plot MSE loss
    if mse_losses_smooth is not None:
        plt.plot(steps, mse_losses_smooth, label='Hidden-layer Loss', color='tab:red', alpha=0.7)

    # Plot eval losses with distinct style
    if eval_losses is not None and eval_steps is not None and len(eval_losses) > 0:
        plt.plot(eval_steps, eval_losses, label='Eval Loss', color='tab:green', linestyle='--', marker='o', markersize=8)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Over Steps (Smoothed)')
    plt.grid(True)
    plt.legend(loc='upper right')

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

    # Set the best hyperparameters directly
    best_hyperparams = {
        'learning_rate': 1e-5,
        'hidden_weight': 0.5,
        'temperature': 4.0,
        # Include other best hyperparameters if necessary
    }
    hyperparams.update(best_hyperparams)

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
    hyperparams['early_stopping_patience'] = int(hyperparams.get('early_stopping_patience', 2))
    hyperparams['temperature'] = float(hyperparams.get('temperature', 1.0))

    logger.info("Loaded hyperparameters from config file.")

    num_labels = 3  # Negative, Neutral, Positive
    tokenizer_name = "distilbert-base-uncased"
    student_model_name = "distilbert-base-uncased"
    teacher_model_names = [
        "bert-base-uncased",
        "ProsusAI/finbert" ,
        "langecod/Financial_Phrasebank_RoBERTa"
    ]

    # Load data
    from scripts.data_preprocessing.data_preprocessing_bert_phrasebank import get_preprocessed_data
    train_dataset, val_dataset, test_dataset = get_preprocessed_data(save_dir='',
                                                                    teacher_model_names=teacher_model_names)

    columns = ['input_ids', 'attention_mask', 'labels']
    for teacher_name in teacher_model_names:
        columns.extend([f'{teacher_name}_input_ids', f'{teacher_name}_attention_mask'])
    # After loading the datasets
    train_dataset.set_format(type='torch')
    val_dataset.set_format(type='torch')
    test_dataset.set_format(type='torch')

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
    student_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    student_model = model_class.from_pretrained(student_model_name, num_labels=num_labels).to(device)

    # Initialize teacher models and tokenizers
    teacher_models = []
    projection_layers = []
    teacher_tokenizers = []
    for teacher_name in teacher_model_names:
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        teacher_tokenizers.append(teacher_tokenizer)
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

    # Enable gradient checkpointing if needed
    if hyperparams['gradient_checkpointing']:
        student_model.gradient_checkpointing_enable()

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=student_tokenizer)

    # Define custom trainer with knowledge distillation
    class CustomTrainer(Trainer):
        def __init__(self, *args, teacher_models=None, teacher_model_names=None, projection_layers=None,
                     hidden_weight=0.5, temperature=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_models = [tm.to(self.args.device) for tm in teacher_models]
            for tm in self.teacher_models:
                tm.eval()
            self.teacher_model_names = teacher_model_names
            self.projection_layers = projection_layers
            self.hidden_weight = hidden_weight
            self.temperature = temperature

            # For collecting MSE and KL losses
            self.mse_losses = []
            self.kd_losses = []
            self.steps = []

        def compute_loss(self, model, inputs, return_outputs=False):
            # Move inputs to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = inputs.pop("labels")

            # Extract student inputs
            student_inputs = {
                'input_ids': inputs.pop('input_ids'),
                'attention_mask': inputs.pop('attention_mask'),
                'labels': labels,
            }

            # Forward pass for the student model
            outputs = model(**student_inputs, output_hidden_states=True)
            student_loss = outputs.loss
            student_logits = outputs.logits
            student_hidden_states = outputs.hidden_states[-1]

            if model.training:
                total_kd_loss = 0.0
                total_mse_loss = 0.0
                num_teachers = len(self.teacher_models)

                for idx, teacher in enumerate(self.teacher_models):
                    projection_layer = self.projection_layers[idx]
                    teacher_name = self.teacher_model_names[idx]

                    # Retrieve teacher inputs from inputs
                    teacher_input_ids = inputs.pop(f'{teacher_name}_input_ids')
                    teacher_attention_mask = inputs.pop(f'{teacher_name}_attention_mask')

                    # Forward pass for the teacher model
                    with torch.no_grad():
                        teacher_outputs = teacher(
                            input_ids=teacher_input_ids,
                            attention_mask=teacher_attention_mask,
                            output_hidden_states=True,
                        )
                        teacher_logits = teacher_outputs.logits
                        teacher_hidden_states = teacher_outputs.hidden_states[-1]

                    # Compute losses
                    temperature = self.temperature
                    kd_loss = nn.KLDivLoss(reduction='batchmean')(
                        F.log_softmax(student_logits / temperature, dim=-1),
                        F.softmax(teacher_logits / temperature, dim=-1)
                    ) * (temperature ** 2)

                    if projection_layer is not None:
                        teacher_hidden_states = projection_layer(teacher_hidden_states)

                    mse_loss = F.mse_loss(student_hidden_states, teacher_hidden_states)

                    total_kd_loss += kd_loss
                    total_mse_loss += mse_loss

                # Average the losses and compute total loss
                avg_kd_loss = total_kd_loss / num_teachers
                avg_mse_loss = total_mse_loss / num_teachers
                total_loss = student_loss + self.hidden_weight * (avg_kd_loss + avg_mse_loss)

                # Log and collect losses
                self.log({
                    "student_loss": student_loss.detach().item(),
                    "kd_loss": avg_kd_loss.detach().item(),
                    "mse_loss": avg_mse_loss.detach().item()
                })
                self.kd_losses.append(avg_kd_loss.detach().item())
                self.mse_losses.append(avg_mse_loss.detach().item())
                self.steps.append(self.state.global_step)
            else:
                total_loss = outputs.loss

            return (total_loss, outputs) if return_outputs else total_loss

    # Initialize the callbacks
    loss_collector = LossCollectorCallback()
    eval_loss_collector = EvalLossCollectorCallback()

    hidden_weight = hyperparams.get('hidden_weight', 0.5)
    temperature = hyperparams.get('temperature', 1.0)

    # Training arguments from hyperparameters
    training_args = TrainingArguments(
        output_dir=f'./{dataset_name}_student_model',
        num_train_epochs=hyperparams['num_train_epochs'],
        per_device_train_batch_size=hyperparams['per_device_train_batch_size'],
        per_device_eval_batch_size=hyperparams['per_device_eval_batch_size'],
        learning_rate=hyperparams['learning_rate'],
        evaluation_strategy=hyperparams['evaluation_strategy'],
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
        remove_unused_columns=False,
        prediction_loss_only=False,  # Ensure that evaluation loss is computed
    )

    # Initialize trainer with EarlyStoppingCallback
    trainer = CustomTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=student_tokenizer,
        data_collator=data_collator,
        teacher_models=teacher_models,
        teacher_model_names=teacher_model_names,
        projection_layers=projection_layers,
        hidden_weight=hidden_weight,
        temperature=temperature,
        compute_metrics=None,
        callbacks=[
            loss_collector,
            eval_loss_collector,
            EarlyStoppingCallback(early_stopping_patience=hyperparams['early_stopping_patience'])
        ],
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

    # Evaluate the student & teacher models on test data
    logger.info("Evaluation on student and teacher models started.")
    print("Evaluating student and teacher models on test data...")

    student_predictions = []
    teacher_predictions_list = [[] for _ in teacher_models]  # List of lists
    label_answers = []

    student_total_inference_time = 0.0
    teacher_total_inference_time = [0.0 for _ in teacher_models]

    # Change student model to eval mode
    student_model.eval()

    # Evaluation loop
    for example in tqdm(test_dataset, desc="Evaluating student and teacher models"):
        labels = example['labels']

        with torch.no_grad():
            # Student model prediction
            student_inputs = {
                'input_ids': example['input_ids'].unsqueeze(0).to(device),
                'attention_mask': example['attention_mask'].unsqueeze(0).to(device),
            }
            start_time_eval = time.time()
            student_outputs = student_model(**student_inputs)
            inference_time = time.time() - start_time_eval
            student_total_inference_time += inference_time

            student_logits = student_outputs.logits
            student_pred_label = torch.argmax(student_logits, dim=-1).cpu().numpy().item()
            student_predictions.append(student_pred_label)

            # Teacher models predictions
            for idx, teacher in enumerate(teacher_models):
                teacher_name = teacher_model_names[idx]
                teacher_inputs = {
                    'input_ids': example[f'{teacher_name}_input_ids'].unsqueeze(0).to(device),
                    'attention_mask': example[f'{teacher_name}_attention_mask'].unsqueeze(0).to(device),
                }
                start_time_eval = time.time()
                teacher_outputs = teacher(**teacher_inputs)
                inference_time = time.time() - start_time_eval
                teacher_total_inference_time[idx] += inference_time

                teacher_logits = teacher_outputs.logits
                teacher_pred_label = torch.argmax(teacher_logits, dim=-1).cpu().numpy().item()
                teacher_predictions_list[idx].append(teacher_pred_label)

        label_answers.append(labels.item())

    logger.info("Evaluation completed.")

    num_examples = len(test_dataset)
    student_avg_inference_time = student_total_inference_time / num_examples
    teacher_avg_inference_times = [total_time / num_examples for total_time in teacher_total_inference_time]

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
        f.write(f"Number of test examples: {len(test_dataset)}\n\n")

        # Hyperparameters used
        f.write("Hyperparameters Used:\n")
        for param, value in hyperparams.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")

        # Training metrics
        f.write("Training Metrics:\n")
        f.write(f"Total training time: {train_runtime:.2f} seconds\n")
        f.write(f"Training samples per second: {train_result.metrics.get('train_samples_per_second', 'N/A')}\n")
        f.write(f"Training steps per second: {train_result.metrics.get('train_steps_per_second', 'N/A')}\n")
        f.write(f"Final training loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}\n")
        f.write(f"Epochs completed: {train_result.metrics.get('epoch', 'N/A'):.1f}\n\n")

        # Student evaluation
        f.write("\nStudent Model Evaluation on Test Data:\n")
        f.write(f"Accuracy: {student_accuracy * 100:.2f}%\n")
        f.write(f"Precision: {student_precision * 100:.2f}%\n")
        f.write(f"Recall: {student_recall * 100:.2f}%\n")
        f.write(f"F1 Score: {student_f1 * 100:.2f}%\n")
        f.write(f"Average Inference Time per Example: {student_avg_inference_time * 1000:.2f} ms\n\n")

        # Teacher evaluations
        for idx, metrics in enumerate(teacher_metrics):
            f.write(f"Teacher Model {idx + 1} ({teacher_model_names[idx]}) Evaluation on Test Data:\n")
            f.write(f"Accuracy: {metrics['accuracy'] * 100:.2f}%\n")
            f.write(f"Precision: {metrics['precision'] * 100:.2f}%\n")
            f.write(f"Recall: {metrics['recall'] * 100:.2f}%\n")
            f.write(f"F1 Score: {metrics['f1'] * 100:.2f}%\n")
            f.write(f"Average Inference Time per Example: {teacher_avg_inference_times[idx] * 1000:.2f} ms\n\n")

        # Write test examples
        num_examples_to_show = min(10, len(test_dataset))
        f.write(f"\nFirst {num_examples_to_show} Test Examples:\n")
        label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        for idx in range(num_examples_to_show):
            input_ids = test_dataset[idx]['input_ids']
            input_text = student_tokenizer.decode(input_ids, skip_special_tokens=True)
            actual_label = label_mapping[label_answers[idx]]
            student_pred_label = label_mapping[student_predictions[idx]]
            teacher_preds = [label_mapping[teacher_predictions_list[t_idx][idx]] for t_idx in range(len(teacher_models))]

            f.write(f"\nExample {idx + 1}:\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Actual Label: {actual_label}\n")
            f.write(f"Student Predicted Label: {student_pred_label}\n")
            for t_idx, teacher_pred in enumerate(teacher_preds):
                f.write(f"Teacher {t_idx + 1} Predicted Label: {teacher_pred}\n")

        # Training progress evaluation (moved to the bottom)
        training_losses = loss_collector.student_losses
        mse_losses = trainer.mse_losses  # Collected in CustomTrainer
        kd_losses = trainer.kd_losses
        steps = loss_collector.steps  # Use the steps collected by LossCollectorCallback

        if len(training_losses) > 0:
            f.write("\nTraining Progress Evaluation:\n")
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

    print(f"Training complete. Report saved to {output_report_path}")
    logger.info(f"Training complete. Report saved to {output_report_path}")

    # Call generate_training_graph with window_size parameter and eval losses
    generate_training_graph(
        training_losses,
        mse_losses,
        steps,  # Use the steps collected
        training_graph_path,
        window_size=5,
        eval_losses=eval_loss_collector.eval_losses,
        eval_steps=eval_loss_collector.steps
    )

if __name__ == "__main__":
    main()
