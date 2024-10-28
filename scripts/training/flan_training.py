import argparse
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import os
import logging
import random
import numpy as np
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time
from scripts.training.callback import PrintSampleCallback
from scripts.training.util import extract_answer, extract_rationale
from scripts.data_preprocessing.data_preprocessing_flan import get_preprocessed_data

# Set up logging
logging.basicConfig(
    filename='flan_training_progress.log',
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
    on_gpu = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
        on_gpu = True
        print("Using CUDA. DeepSpeed is enabled for training")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device, on_gpu


def parse_args():
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation from teacher model")
    parser.add_argument('--data_portion', type=float, default=1.0, help="Portion of dataset to use (e.g., 0.01 for 1%)")
    parser.add_argument('--output_report', type=str, default='training_report.txt',
                        help="Filename to save the output report")
    args = parser.parse_args()
    return args


def main():
    logger.info("Starting script...")
    # Set random seed
    set_seed(42)

    # Parse arguments
    args = parse_args()
    data_portion = args.data_portion
    output_report = args.output_report

    # Get device
    device, on_gpu = get_device()

    # Load hyperparameters from config file
    config_path = os.path.join('../../config', 'hyperparameters.yaml')
    ds_config_path = os.path.join('../../config', 'ds_config.json')
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
    hyperparams['hidden_weight'] = float(hyperparams.get('hidden_weight', 0.5))
    hyperparams['bf16'] = bool(hyperparams['bf16'])
    hyperparams['fp16'] = bool(hyperparams['fp16'])
    hyperparams['gradient_accumulation_steps'] = int(hyperparams['gradient_accumulation_steps'])
    hyperparams['gradient_checkpointing'] = bool(hyperparams['gradient_checkpointing'])
    hyperparams['eval_accumulation_steps'] = int(hyperparams['eval_accumulation_steps'])

    logger.info("Loaded hyperparameters from config file.")
    # Load data
    train_dataset, val_dataset = get_preprocessed_data(save_dir='')
    logger.info(f"Loaded training dataset with {len(train_dataset)} examples.")
    logger.info(f"Loaded validation dataset with {len(val_dataset)} examples.")

    # Use portion of dataset
    if data_portion < 1.0:
        num_train_examples = max(5, int(len(train_dataset) * data_portion))
        num_val_examples = max(200, int(len(val_dataset) * data_portion))
        train_dataset = train_dataset.shuffle(seed=42).select(range(num_train_examples))
        val_dataset = val_dataset.shuffle(seed=42).select(range(num_val_examples))

    logger.info(f"Using {len(train_dataset)} training examples after applying data_portion={data_portion}.")
    logger.info(f"Using {len(val_dataset)} validation examples after applying data_portion={data_portion}.")

    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after applying data_portion. Please use a larger data_portion.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty after applying data_portion. Please use a larger data_portion.")

    # Initialize teacher and student models and tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    teacher_model.eval()

    student_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    student_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

    # Enable gradient checkpointing
    student_model.gradient_checkpointing_enable()

    # Prepare data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=student_tokenizer, model=student_model)

    # Generate teacher outputs for training data
    print("Generating teacher outputs for training data...")
    logger.info("Generating teacher outputs for training data...")
    teacher_outputs = []
    for idx, example in enumerate(tqdm(train_dataset, desc="Processing training data")):
        if idx % 500 == 0:
            logger.info(f"Processed {idx} training examples.")
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)
        with torch.no_grad():
            output_ids = teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_beams=3,
                early_stopping=True
            )
        output_text = teacher_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        teacher_outputs.append(output_text)

    # Add teacher outputs to train_dataset
    train_dataset = train_dataset.add_column('labels', teacher_outputs)

    # Generate teacher outputs for validation data and collect actual answers
    logger.info("Generating teacher outputs for validation data...")
    print("Generating teacher outputs for validation data...")
    val_teacher_outputs = []
    val_actual_answers = []
    val_actual_rationales = []
    val_teacher_rationales = []
    teacher_predictions = []
    for idx, example in enumerate(tqdm(val_dataset, desc="Processing validation data")):
        if idx % 100 == 0:
            logger.info(f"Processed {idx} validation examples.")
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)
        with torch.no_grad():
            output_ids = teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,  # Increase max_length if needed
                num_beams=3,
                early_stopping=True
            )
        output_text = teacher_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        val_teacher_outputs.append(output_text)
        val_actual_answers.append(example['correct'])  # Collect actual answers
        val_actual_rationales.append(example['rationale'])  # Collect actual rationales

        # Extract teacher's rationale
        teacher_answer = extract_answer(output_text)
        teacher_rationale = extract_rationale(output_text)
        val_teacher_rationales.append(teacher_rationale)
        teacher_predictions.append(teacher_answer)

        # Print the teacher's rationale vs actual rationale for the first few examples
        # if idx < 5:
        #     input_text = teacher_tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        #     print(f"Example {idx + 1}:")
        #     print(f"Input: {input_text}")
        #     print(f"Actual Answer: {example['correct']}")
        #     print(f"Actual Rationale: {example['rationale']}")
        #     print(f"Teacher's Output: {output_text}")
        #     print(f"Teacher's Answer: {teacher_answer}")
        #     print(f"Teacher's Rationale: {teacher_rationale}\n")

    # Add collected data to the validation dataset
    val_dataset = val_dataset.add_column('labels', val_teacher_outputs)
    val_dataset = val_dataset.add_column('actual_answers', val_actual_answers)
    val_dataset = val_dataset.add_column('actual_rationales', val_actual_rationales)
    val_dataset = val_dataset.add_column('teacher_rationales', val_teacher_rationales)

    # Tokenize labels for student model
    def tokenize_labels(examples):
        labels = examples['labels']
        tokenized_labels = student_tokenizer(
            labels,
            max_length=128,
            truncation=True
        )
        examples['labels'] = tokenized_labels['input_ids']
        return examples

    train_dataset = train_dataset.map(tokenize_labels, batched=True)
    val_dataset = val_dataset.map(tokenize_labels, batched=True)

    def move_to_cpu(batch):
        if isinstance(batch, dict):
            return {k: v.to('cpu') for k, v in batch.items()}
        elif isinstance(batch, list):
            return [v.to('cpu') for v in batch]
        else:
            return batch.to('cpu')

    # Define custom trainer with hidden state distillation
    class CustomTrainer(Trainer):
        def __init__(self, *args, teacher_model=None, hidden_weight=0.5, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = teacher_model.to(self.args.device)
            self.teacher_model.eval()
            self.hidden_weight = hidden_weight

            # Add a projection layer
            self.projection_layer = nn.Linear(768, 512, bias=False).to(self.args.device)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            # Prepare decoder input IDs
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
            inputs['decoder_input_ids'] = decoder_input_ids

            outputs = model(**inputs, output_hidden_states=True)
            student_loss = outputs.loss

            # Get the student's last hidden state
            student_hidden_states = outputs.decoder_hidden_states[-1]  # Shape: [batch_size, seq_length, 512]

            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=inputs['decoder_input_ids'],
                    output_hidden_states=True,
                )
                teacher_hidden_states = teacher_outputs.decoder_hidden_states[
                    -1]  # Shape: [batch_size, seq_length, 768]

                # Project teacher hidden states to match student hidden size
                projected_teacher_hidden_states = self.projection_layer(
                    teacher_hidden_states)  # Shape: [batch_size, seq_length, 512]

            # Compute MSE loss between student and projected teacher hidden states
            mse_loss = F.mse_loss(student_hidden_states, projected_teacher_hidden_states)

            # Total loss
            total_loss = student_loss + self.hidden_weight * mse_loss

            # Clear CUDA cache after each loss computation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return (total_loss, outputs) if return_outputs else total_loss

        def evaluation_loop(self, *args, **kwargs):
            """
            Override the evaluation loop to clear CUDA cache before evaluation.
            """
            # Clear CUDA cache before evaluation starts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            output = super().evaluation_loop(*args, **kwargs)

            # if isinstance(output, dict):
            #     output = {k: v.to('cpu') for k, v in output.items()}
            # elif isinstance(output, list):
            #     output = [v.to('cpu') for v in output]
            # else:
            #     output = output.to('cpu')

            return output

    # Training arguments from hyperparameters
    training_args = TrainingArguments(
        output_dir='./student_model',
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
        fp16=hyperparams['fp16'] if on_gpu else None,
        bf16=hyperparams['bf16'] if on_gpu else None,
        report_to='none',
        gradient_checkpointing=hyperparams['gradient_checkpointing'],
        eval_accumulation_steps=hyperparams['eval_accumulation_steps']
        # deepspeed=ds_config_path if on_gpu else None
    )

    # Hidden weight from hyperparameters
    hidden_weight = hyperparams['hidden_weight']

    print_sample_callback = PrintSampleCallback(
        tokenizer=student_tokenizer,
        interval_steps=500
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Convert predictions to a flat tensor
        predictions = torch.cat([torch.tensor(pred) for pred in predictions])

        # Replace -100 in labels as we do in compute_loss
        labels = np.where(labels != -100, labels, student_tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = student_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = student_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Extract answers from predictions and labels
        pred_answers = [extract_answer(text) for text in decoded_preds]
        label_answers = [extract_answer(text) for text in decoded_labels]

        # Compute accuracy
        accuracy = accuracy_score(label_answers, pred_answers)

        return {'accuracy': accuracy}

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
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), print_sample_callback],
    )

    # Training
    print("Starting model training...")
    logger.info("Starting model training...")
    start_time = time.time()
    train_result = trainer.train()
    train_runtime = time.time() - start_time

    # Save the model
    trainer.save_model('./student_model')
    logger.info("Model saved to './student_model'.")

    # Compute teacher accuracy from initial predictions
    print("Computing teacher model accuracy on validation data...")
    teacher_accuracy = accuracy_score(val_actual_answers, teacher_predictions)

    # Evaluate the student model on validation data
    print("Evaluating student model on validation data...")
    student_model.eval()
    student_predictions = []
    student_rationales = []
    for example in tqdm(val_dataset, desc="Evaluating student model"):
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)
        with torch.no_grad():
            output_ids = student_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_beams=3,
                early_stopping=True
            )
        output_text = student_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        student_predictions.append(output_text)
        rationale = extract_rationale(output_text)
        student_rationales.append(rationale)
    logger.info("Training completed.")

    # Extract predicted answers for student
    student_pred_answers = [extract_answer(text) for text in student_predictions]

    # Compute student accuracy
    student_accuracy = accuracy_score(val_actual_answers, student_pred_answers)

    # Write report
    with open(output_report, 'w') as f:
        f.write("Training complete.\n\n")

        # Number of questions trained
        f.write(f"Number of training examples: {len(train_dataset)}\n")
        f.write(f"Number of validation examples: {len(val_dataset)}\n\n")

        # Training metrics
        f.write("Training Metrics:\n")
        f.write(f"Total training time: {train_runtime:.2f} seconds\n")
        f.write(f"Training samples per second: {train_result.metrics['train_samples_per_second']:.3f}\n")
        f.write(f"Training steps per second: {train_result.metrics['train_steps_per_second']:.3f}\n")
        f.write(f"Final training loss: {train_result.metrics['train_loss']:.4f}\n")
        f.write(f"Epochs completed: {train_result.metrics['epoch']:.1f}\n\n")

        # Training progress evaluation
        f.write("Training Progress Evaluation:\n")
        for log in trainer.state.log_history:
            if 'loss' in log and 'learning_rate' in log:
                f.write(
                    f"Epoch {log.get('epoch', '')}: Step {log.get('step', '')}, Training Loss = {log['loss']:.4f}\n")
            if 'eval_loss' in log:
                f.write(
                    f"Epoch {log.get('epoch', '')}: Step {log.get('step', '')}, Validation Loss = {log['eval_loss']:.4f}, Validation Accuracy = {log.get('eval_accuracy', ''):.4f}\n")

        # Teacher evaluation
        f.write("\nTeacher Model Evaluation on Validation Data:\n")
        f.write(f"Teacher Model Accuracy: {teacher_accuracy * 100:.2f}%\n")

        # Student evaluation
        f.write("\nStudent Model Evaluation on Validation Data:\n")
        f.write(f"Student Model Accuracy: {student_accuracy * 100:.2f}%\n")

        # Write validation examples
        num_examples_to_show = min(30, len(val_dataset))
        f.write(f"\nFirst {num_examples_to_show} Validation Examples:\n")
        for idx in range(num_examples_to_show):
            example = val_dataset[idx]
            input_text = teacher_tokenizer.decode(example['input_ids'], skip_special_tokens=True)

            f.write(f"\nExample {idx + 1}:\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Actual Answer: {val_actual_answers[idx]}\n")
            f.write(f"Teacher's Output: {val_teacher_outputs[idx]}\n")
            f.write(f"Teacher's Answer: {teacher_predictions[idx]}\n")
            f.write(f"Teacher's Rationale: {val_teacher_rationales[idx]}\n")
            f.write(f"Student's Output: {student_predictions[idx]}\n")
            f.write(f"Student's Answer: {student_pred_answers[idx]}\n")
            f.write(f"Student's Rationale: {student_rationales[idx]}\n")

    print(f"Training complete. Report saved to {output_report}")
    logger.info(f"Training complete. Report saved to {output_report}")


if __name__ == "__main__":
    main()
