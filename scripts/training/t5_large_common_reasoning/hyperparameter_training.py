import random
import numpy as np
import yaml
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from sklearn.model_selection import KFold
from datasets import load_from_disk
import torch
import logging
import os

# Configure logging
logging.basicConfig(
    filename="logs/training_t5_large.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config(config_path='config/config.yaml'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
tokenizer = T5Tokenizer.from_pretrained(config["tokenizer"]["name"])

# Load the preprocessed dataset using load_from_disk
try:
    tokenized_dataset = load_from_disk(config["datasets"]["socialiqa"]["path"])
    logging.info(f"Successfully loaded tokenized dataset from {config['datasets']['socialiqa']['path']}.")
except FileNotFoundError:
    logging.error(f"Tokenized dataset directory {config['datasets']['socialiqa']['path']} not found.")
    raise
except Exception as e:
    logging.error(f"Error loading tokenized dataset: {e}")
    raise

# Apply subset percentage
subset_percentage = config["hyperparameter_tuning"]["subset_percentage"]
print(f"subset_percentage is {subset_percentage} and length of total dataset is {len(tokenized_dataset)}")  # Add this line for debugging
subset_size = int(subset_percentage * len(tokenized_dataset))

# Always apply the subset selection
subset_indices = random.sample(range(len(tokenized_dataset)), subset_size)
tokenized_dataset = tokenized_dataset.select(subset_indices)
logging.info(f"Applied subset: {subset_size} samples out of {len(tokenized_dataset)}.")

print(f"Final dataset size after subsetting: {len(tokenized_dataset)}")

# Define cross-validation split
kfold_splits = config["hyperparameter_tuning"]["kfold_splits"]
kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=config["random_seed"])
splits = list(kf.split(tokenized_dataset))


# Define hyperparameter ranges
def get_random_hyperparameters():
    return {
        "learning_rate": random.uniform(1e-6, 5e-5),
        "batch_size": random.choice([1, 2, 4]),
        "max_length": random.choice([64, 128]),
        "dropout_rate": random.uniform(0.1, 0.3),
        "weight_decay": random.uniform(0, 0.1),
        "warmup_steps": random.choice([100, 500]),
        "scheduler_type": random.choice(["linear", "cosine"])
    }

results = []
num_trials = config["hyperparameter_tuning"]["num_trials"]

for i in range(num_trials):
    # Sample hyperparameters
    hyperparams = get_random_hyperparameters()
    logging.info(f"Trial {i+1}/{num_trials} with hyperparameters: {hyperparams}")

    # Select training and validation splits
    train_indices, val_indices = splits[i % kfold_splits]
    train_split = tokenized_dataset.select(train_indices)
    val_split = tokenized_dataset.select(val_indices)

    # Load the model
    try:
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        logging.info("Loaded T5-base model successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        continue

    # Update dropout rates
    model.config.dropout = hyperparams["dropout_rate"]
    model.config.layerdrop = hyperparams["dropout_rate"]
    model.config.attention_dropout = hyperparams["dropout_rate"]

    # Inspection Code: Print first 3 samples of training and validation splits
    if i == 0:
        print("\n--- Inspecting Training Samples ---")
        for j in range(3):
            sample = train_split[j]
            decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            labels_tensor = sample['labels'].clone().detach()
            decoded_label = tokenizer.decode(
                torch.where(labels_tensor == -100, torch.tensor(tokenizer.pad_token_id), labels_tensor),
                skip_special_tokens=True
            ).strip()
            print(f"Sample {j + 1} Training Input Text: {decoded_input}")
            print(f"Sample {j + 1} Training Label Text: {decoded_label}\n")
        
        print("\n--- Inspecting Validation Samples ---")
        for j in range(3):
            sample = val_split[j]
            decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            labels_tensor = sample['labels'].clone().detach()
            decoded_label = tokenizer.decode(
                torch.where(labels_tensor == -100, torch.tensor(tokenizer.pad_token_id), labels_tensor),
                skip_special_tokens=True
            ).strip()
            print(f"Sample {j + 1} Validation Input Text: {decoded_input}")
            print(f"Sample {j + 1} Validation Label Text: {decoded_label}\n")

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="temp_output",
        per_device_train_batch_size=hyperparams["batch_size"],
        per_device_eval_batch_size=hyperparams["batch_size"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        warmup_steps=hyperparams["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="no",
        num_train_epochs=config["training"]["num_train_epochs"],
        logging_dir=config["logging"]["log_file"],
        logging_steps=10,
        disable_tqdm=False,
        report_to="none",
        fp16=False,               # Disable mixed precision
        max_grad_norm=config["training"]["max_norm"],
        gradient_accumulation_steps=config["training"]["accumulation_steps"],
        dataloader_num_workers=2,  # Adjust based on your system
        seed=config["random_seed"]
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=val_split
    )

    try:
        # Train the model
        trainer.train()
        logging.info(f"Training completed for trial {i+1}.")

        # Evaluate the model
        metrics = trainer.evaluate()
        metrics["hyperparameters"] = hyperparams
        results.append(metrics)
        logging.info(f"Evaluation metrics for trial {i+1}: {metrics}")

    except torch.cuda.OutOfMemoryError:
        print(f"Out of memory with hyperparameters: {hyperparams}. Skipping this configuration.")
        logging.error(f"Out of memory with hyperparameters: {hyperparams}. Skipping this configuration.")
        torch.cuda.empty_cache()
        continue
    except Exception as e:
        logging.error(f"Error during training: {e}")
        torch.cuda.empty_cache()
        continue
    finally:
        # Free up memory
        del model, trainer, train_split, val_split
        torch.cuda.empty_cache()

# Sort results by evaluation metric (e.g., F1)
if results:
    sorted_results = sorted(results, key=lambda x: x.get("eval_f1", 0), reverse=True)
    print("Best hyperparameters set:", sorted_results[0]["hyperparameters"])
    logging.info(f"Best hyperparameters set: {sorted_results[0]['hyperparameters']}")
else:
    print("No successful training runs to report.")
    logging.warning("No successful training runs to report.")