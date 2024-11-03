import argparse
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from sklearn.model_selection import KFold
import os
import logging
import random
import numpy as np
import pandas as pd
from collections import deque
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the CustomTrainer class
class CustomTrainer(Trainer):
    """
    A custom Trainer class that renames 'loss' to 'training loss' and 'eval_loss' to 'validation loss' in the logs.
    """
    def log(self, logs: dict, **kwargs):
        # Rename 'loss' to 'training loss' and 'eval_loss' to 'validation loss'
        renamed_logs = {}
        for key, value in logs.items():
            if key == 'loss':
                renamed_logs['training loss'] = value
            elif key == 'eval_loss':
                renamed_logs['validation loss'] = value
            else:
                renamed_logs[key] = value
        # Call the superclass log method with renamed logs
        super().log(renamed_logs, **kwargs)

class SingleExampleEvalCallback(TrainerCallback):
    """
    A custom callback to perform evaluation on a single validation example at specified intervals.
    """
    def __init__(self, eval_steps, validation_dataset):
        self.eval_steps = eval_steps
        self.validation_dataset = validation_dataset
        self.current_val_idx = 0
        self.total_val_examples = len(validation_dataset)
        self.trainer = None  # Will be set after trainer initialization

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step != 0:
            trainer = self.trainer
            if trainer is not None:
                # Select the current validation example
                example = self.validation_dataset[self.current_val_idx]
                self.current_val_idx = (self.current_val_idx + 1) % self.total_val_examples

                # Prepare inputs
                input_ids = torch.tensor(example['input_ids'], device=trainer.args.device).unsqueeze(0)
                attention_mask = torch.tensor(example['attention_mask'], device=trainer.args.device).unsqueeze(0)
                labels = torch.tensor(example['labels'], device=trainer.args.device).unsqueeze(0)

                # Evaluate the model
                model = trainer.model
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss.item()

                # Log the validation loss
                logging.info(f"Step {state.global_step}: Single Example Validation Loss: {loss:.4f}")

                # Update logs with validation loss
                if logs is not None:
                    logs['validation loss'] = loss

                # Ensure the control object knows to update logs
                control.should_log = True
            else:
                logging.warning("Trainer instance not found. Skipping single example evaluation.")
        return control



class RollingAverageCallback(TrainerCallback):
    """
    A custom callback to maintain rolling averages of training and validation losses.
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.training_losses = deque(maxlen=window_size)
        self.validation_losses = deque(maxlen=window_size)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Collect training loss
            if 'training loss' in logs:
                self.training_losses.append(logs['training loss'])

            # Collect validation loss
            if 'validation loss' in logs:
                self.validation_losses.append(float(logs['validation loss']))

            # Compute rolling averages
            avg_training_loss = sum(self.training_losses) / len(self.training_losses) if len(self.training_losses) > 0 else None
            avg_validation_loss = sum(self.validation_losses) / len(self.validation_losses) if len(self.validation_losses) > 0 else None

            # Log the rolling averages
            if avg_training_loss is not None:
                logs['rolling training loss'] = avg_training_loss
            if avg_validation_loss is not None:
                logs['rolling validation loss'] = avg_validation_loss

            # Make sure the rolling averages are written into log history (and thus trainer_state.json)
            if state and state.log_history is not None:
                log_entry = {
                    'step': state.global_step,
                    'rolling training loss': avg_training_loss,
                    'rolling validation loss': avg_validation_loss,
                }
                state.log_history.append(log_entry)
            
            # Print rolling loss logs for better visibility
            if avg_training_loss is not None and avg_validation_loss is not None:
                logging.info(f"Rolling Training Loss (last {self.window_size} steps): {avg_training_loss:.4f}")
                logging.info(f"Rolling Validation Loss (last {self.window_size} steps): {avg_validation_loss:.4f}")
        return control


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a T5 model with 5-fold cross-validation on the AQuA-Rat dataset.")
    parser.add_argument(
        '--train_pct',
        type=float,
        default=1.0,
        help="Percentage of the dataset to train on (between 0 and 1). Default is 1.0 (100%)."
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help="Number of training steps between each model checkpoint save. Default is 500."
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help="Number of training steps between each evaluation. Default is 500."
    )
    parser.add_argument(
        '--num_folds',
        type=int,
        default=5,
        help="Number of folds for cross-validation. Default is 5."
    )
    parser.add_argument(
        '--start_fold',
        type=int,
        default=1,
        help="The fold number to start from (1-based index)."
    )
    args = parser.parse_args()
    if not 0 < args.train_pct <= 1.0:
        parser.error("train_pct must be a float between 0 and 1.")
    if args.save_steps <= 0:
        parser.error("save_steps must be a positive integer.")
    if args.eval_steps <= 0:
        parser.error("eval_steps must be a positive integer.")
    if args.num_folds <= 1:
        parser.error("num_folds must be an integer greater than 1.")
    if not 1 <= args.start_fold <= args.num_folds:
        parser.error("start_fold must be between 1 and num_folds.")
    return args

def load_preprocessed_data(save_path='dataset_trainer_preprocessed_data.pkl'):
    """
    Load preprocessed data from a pickle file.

    Args:
        save_path (str): Path to the preprocessed data file.

    Returns:
        Dataset: A Hugging Face Dataset containing the training data.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"The specified path '{save_path}' does not exist. Please preprocess the data first.")

    logger.info(f"Loading preprocessed data from '{save_path}'...")
    preprocessed_data = torch.load(save_path, weights_only=True)
    logger.info("Data loaded successfully.\n")

    # Convert dictionaries to Hugging Face Dataset
    # Assuming preprocessed_data is a dictionary with 'input_ids', 'attention_mask', and 'labels'
    data = {
        'input_ids': preprocessed_data['train']['input_ids'],
        'attention_mask': preprocessed_data['train']['attention_mask'],
        'labels': preprocessed_data['train']['labels'],
    }
    dataset = Dataset.from_dict(data)
    logger.info("Dataset created successfully.\n")
    return dataset

def initialize_model_and_tokenizer(model_name='t5-small', tokenizer_path='tokenizers/t5_tokenizer/', max_length=512):
    """
    Initialize the tokenizer and model.

    Args:
        model_name (str): Name of the pre-trained model.
        tokenizer_path (str): Path where the tokenizer is saved.
        max_length (int): Maximum sequence length for inputs and outputs.

    Returns:
        tuple: (tokenizer, model)
    """
    logger.info(f"Loading the tokenizer and model '{model_name}'...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, model_max_length=max_length)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Resize token embeddings if new tokens have been added
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized token embeddings to match tokenizer.\n")

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    logger.info("Tokenizer and model loaded successfully.\n")
    return tokenizer, model


def get_latest_checkpoint(output_dir):
    import os
    import re
    from glob import glob

    checkpoints = list(sorted(glob(os.path.join(output_dir, 'checkpoint-*')), key=lambda x: int(re.findall(r'checkpoint-(\d+)', x)[0])))
    if len(checkpoints) > 0:
        return checkpoints[-1]  # Return the latest checkpoint
    else:
        return None



def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.

    Args:
        eval_pred: A tuple of (logits, labels)

    Returns:
        dict: A dictionary containing the validation loss.
    """
    import numpy as np
    logits, labels = eval_pred

    # Convert logits and labels to torch tensors
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Shift logits and labels for loss computation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    return {"validation loss": loss.item()}


def main():
    # ======================================
    # 1. Set Random Seed for Reproducibility
    # ======================================
    set_seed(42)

    # ======================================
    # 2. Parse Command-Line Arguments
    # ======================================
    args = parse_args()
    train_pct = args.train_pct
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    num_folds = args.num_folds
    logger.info(f"Training on {train_pct * 100}% of the dataset.\n")
    logger.info(f"Model checkpoint will be saved every {save_steps} steps.")
    logger.info(f"Evaluation will occur every {eval_steps} steps.")
    logger.info(f"Number of folds for cross-validation: {num_folds}\n")

    # ======================================
    # 3. Load Preprocessed Data
    # ======================================
    dataset = load_preprocessed_data(save_path='dataset_trainer_preprocessed_data.pkl')  # Path to preprocessed data

    # ======================================
    # 4. Sample the Dataset Based on train_pct
    # ======================================
    if train_pct < 1.0:
        logger.info(f"Sampling {train_pct * 100}% of the training data.")
        train_size = int(len(dataset) * train_pct)
        dataset = dataset.shuffle(seed=42).select(range(train_size))
        logger.info("Dataset sampling completed.\n")
    else:
        logger.info("Using the entire dataset for training.\n")

    # ======================================
    # 5. Initialize Tokenizer and Model
    # ======================================
    model_name = 't5-small'  # Ensure this matches the tokenizer
    tokenizer_path = 'tokenizers/t5_tokenizer/'  # Path where tokenizer is saved
    tokenizer, model = initialize_model_and_tokenizer(
        model_name=model_name,
        tokenizer_path=tokenizer_path,
        max_length=512  # Adjust based on your data
    )

    # ======================================
    # 6. Prepare K-Fold Cross-Validation Splits
    # ======================================
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold = 1

    # To store metrics for each fold
    fold_metrics = []

    for train_index, val_index in kf.split(dataset):
        for fold_num, (train_index, val_index) in enumerate(kf.split(dataset), start=1):
            if fold_num < args.start_fold:
                logger.info(f"Skipping fold {fold_num} as per start_fold argument.")
                continue
            fold = fold_num
            logger.info(f"========== Fold {fold} ==========")
        
            # Split the dataset into training and validation for this fold
            train_dataset = dataset.select(train_index).shuffle(seed=42)  # Shuffle training dataset        
            val_dataset = dataset.select(val_index).shuffle(seed=42)      # Shuffle validation dataset        
            
            logger.info(f"Training samples: {len(train_dataset)}")
            logger.info(f"Validation samples: {len(val_dataset)}\n")
            
            # ======================================
            # 7. Define Training Arguments for This Fold
            # ======================================
            # Create a unique output directory for each fold
            output_dir = f'./t5_aqua_rat_finetuned_fold{fold}'

            training_args = TrainingArguments(
                output_dir=output_dir,                       # Directory to save model checkpoints
                num_train_epochs=3,                          # Total number of training epochs
                per_device_train_batch_size=8,               # Reduced batch size to fit GPU memory
                gradient_accumulation_steps=2,               # To maintain effective batch size of 16
                gradient_checkpointing=True,                 # Enable gradient checkpointing
                warmup_steps=500,                            # Number of warmup steps for learning rate scheduler
                weight_decay=0.01,                           # Strength of weight decay
                logging_dir=f'./t5_aqua_rat_logs_fold{fold}',# Directory for storing logs
                logging_steps=eval_steps,                    # Log training metrics every eval_steps
                save_steps=save_steps,                       # Save checkpoint every save_steps
                save_total_limit=2,                          # Keep only the latest 5 checkpoints
                evaluation_strategy='no',                 # Enable evaluation at specific steps
                load_best_model_at_end=False,                 # Load the best model at the end of training
                greater_is_better=False,                     # Lower validation loss is better
                fp16=torch.cuda.is_available(),              # Use mixed precision if available
                per_device_eval_batch_size=1,  # Reduce batch size to minimize memory usage
                # max_eval_samples=10,  # Limit evaluation to 10 samples
            )

            checkpoint = get_latest_checkpoint(output_dir)
            if checkpoint:
                logger.info(f"Resuming training from checkpoint: {checkpoint}")
            else:
                logger.info("No checkpoint found. Starting training from scratch.")

            # ======================================
            # 8. Define Data Collator
            # ======================================
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding='longest',                             # Dynamic padding to save memory
                return_tensors='pt',
            )

            # ======================================
            # 9. Initialize the CustomTrainer with Built-in Evaluation
            # ======================================
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=None,
                callbacks=[],  # We'll add callbacks after initialization
            )


            # Initialize and Add Callbacks
            single_example_eval_callback = SingleExampleEvalCallback(eval_steps=eval_steps, validation_dataset=val_dataset)
            single_example_eval_callback.trainer = trainer

            rolling_average_callback = RollingAverageCallback(window_size=20)

            trainer.add_callback(single_example_eval_callback)
            trainer.add_callback(rolling_average_callback)

            # ======================================
            # 10. Start Training for This Fold
            # ======================================
            trainer.train(resume_from_checkpoint=checkpoint)

            # ======================================
            # 11. Evaluate the Model on the Validation Set
            # ======================================
            eval_results = trainer.evaluate()

            # Log the evaluation results
            logger.info(f"Evaluation results for Fold {fold}:")
            for key, value in eval_results.items():
                logger.info(f"  {key}: {value}")
            logger.info("\n")

            # Store metrics
            fold_metrics.append(eval_results)

            # Optionally, save metrics to a file
            metrics_df = pd.DataFrame(fold_metrics)
            metrics_df.to_csv('cross_validation_metrics.csv', index=False)

            fold += 1

    # ======================================
    # 12. Aggregate and Report Cross-Validation Results
    # ======================================
    logger.info("========== Cross-Validation Summary ==========")
    metrics_df = pd.DataFrame(fold_metrics)
    logger.info(metrics_df)

    # Calculate average and standard deviation for each metric
    summary = metrics_df.describe().loc[['mean', 'std']]
    logger.info("\nCross-Validation Metrics Summary:")
    logger.info(summary)

    logger.info("\nTraining and cross-validation completed successfully!")

if __name__ == "__main__":
    main()
