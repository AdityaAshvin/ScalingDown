import random
import numpy as np
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from datasets import load_dataset
from sklearn.model_selection import KFold
import torch

# 1. Define hyperparameter ranges
def get_random_hyperparameters():
    return {
        "learning_rate": random.uniform(1e-6, 5e-5),
        "batch_size": random.choice([8, 16, 32]),
        "max_length": random.choice([64, 128, 256]),
        "dropout_rate": random.uniform(0.1, 0.5),
        "weight_decay": random.uniform(0, 0.1),
        "warmup_steps": random.choice([0, 10, 100, 500]),
        "scheduler_type": random.choice(["linear", "cosine"])
    }

# 2. Load and split the dataset
tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
dataset = load_dataset("winogrande", "winogrande_xl", split="train", trust_remote_code=True)

# Select 3% of the dataset
subset_size = int(0.07 * len(dataset))
subset_indices = random.sample(range(len(dataset)), subset_size)
dataset = dataset.select(subset_indices)

# Tokenize dataset
def preprocess_data(example):
    input_str = (
        f"Sentence: {example['sentence']}\n"
        f"Choose the correct option:\n"
        f"1: {example['option1']}\n"
        f"2: {example['option2']}\n"
        f"Answer:"
    )

    encoding = tokenizer(
        input_str, 
        padding="max_length", 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )

    # Get the correct answer text based on the 'answer' field
    answer_text = example[f"option{example['answer']}"]

    # Tokenize the labels (the correct answer)
    label_encoding = tokenizer(
        answer_text,
        padding="max_length",
        truncation=True,
        max_length=128,  # Ensure this matches the input's max_length
        return_tensors="pt"
    )

    # Return squeezed tensors for each required key
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": label_encoding["input_ids"].squeeze(0)
    }



tokenized_dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names)
# Convert the dataset to the proper format
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# 3. Define cross-validation split
kf = KFold(n_splits=5)
splits = list(kf.split(tokenized_dataset))

# 4. Training and evaluation
results = []
for i in range(25):
    # Random hyperparameter set
    hyperparams = get_random_hyperparameters()

    # Pick 10% of the data for training
    train_indices, val_indices = splits[i % 5]
    train_split = tokenized_dataset.select(train_indices)
    val_split = tokenized_dataset.select(val_indices)
    
    # Set up model and training arguments
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # Update dropout rates if necessary
    model.config.dropout = hyperparams["dropout_rate"]
    model.config.layerdrop = hyperparams["dropout_rate"]
    model.config.attention_dropout = hyperparams["dropout_rate"]
    
    training_args = TrainingArguments(
        output_dir="temp_output",
        per_device_train_batch_size=hyperparams["batch_size"],
        per_device_eval_batch_size=hyperparams["batch_size"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        warmup_steps=hyperparams["warmup_steps"],
        eval_strategy="epoch",  # Updated from 'evaluation_strategy'
        save_strategy="no",
        num_train_epochs=1,  # Shorter epoch for tuning
        logging_dir="logs",
        logging_steps=10,  # Adjust as needed
        disable_tqdm=False,  # Enable progress bars
        report_to="none",  # Disable reporting to WandB or other services
        fp16=True
    )
    
    # Set scheduler
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=hyperparams["learning_rate"], 
        weight_decay=hyperparams["weight_decay"]
    )

    # Calculate total training steps
    num_training_steps = training_args.num_train_epochs * (len(train_split) // hyperparams["batch_size"])
    scheduler = get_scheduler(
        hyperparams["scheduler_type"], 
        optimizer=optim, 
        num_warmup_steps=hyperparams["warmup_steps"], 
        num_training_steps=num_training_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=val_split,
        optimizers=(optim, scheduler)  # Pass custom optimizer and scheduler
    )
    
    try:
        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()
        metrics["hyperparameters"] = hyperparams
        results.append(metrics)
    except torch.cuda.OutOfMemoryError:
        print(f"Out of memory with hyperparameters: {hyperparams}. Skipping this configuration.")
        torch.cuda.empty_cache()
        continue
    finally:
        # Free up memory after each training loop
        del model, trainer, train_split, val_split, optim, scheduler
        torch.cuda.empty_cache()

# 5. Sort results by evaluation metric, e.g., accuracy or F1
sorted_results = sorted(results, key=lambda x: x["eval_loss"])
print("Best hyperparameters set:", sorted_results[0]["hyperparameters"])
