# train_stage2.py

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
import os
import logging
import random
import numpy as np
from collections import deque
from transformers.trainer_callback import TrainerCallback, TrainerControl
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define helper functions
def extract_rationale(text):
    """
    Extract the rationale part from the target text.
    
    Args:
        text (str): The decoded label text containing "Answer: X\nRationale: ..."
    
    Returns:
        str: The extracted rationale.
    """
    if "Rationale:" in text:
        try:
            rationale_part = text.split("Rationale:")[1]
            return rationale_part.strip()
        except IndexError:
            return ""
    else:
        return ""

def extract_answer(text):
    """
    Extract the answer from the generated text.

    Args:
        text (str): Generated text containing "Answer: X".

    Returns:
        str: The extracted answer option (e.g., "A", "B", etc.).
    """
    if "Answer:" in text:
        try:
            answer_part = text.split("Answer:")[1]
            answer = answer_part.strip().split()[0]
            return answer
        except IndexError:
            return ""
    else:
        return ""

# Define the CustomTrainer class with semantic loss
class CustomTrainer(Trainer):
    """
    A custom Trainer class that includes semantic loss for rationale evaluation.
    """
    def __init__(self, *args, semantic_weight=0.5, sbert_model=None, **kwargs):
        """
        Initialize the CustomTrainer.
        
        Args:
            semantic_weight (float): Weight to balance cross-entropy loss and semantic loss.
            sbert_model (SentenceTransformer): Pre-trained SBERT model for semantic embeddings.
        """
        super().__init__(*args, **kwargs)
        self.semantic_weight = semantic_weight  # Hyperparameter to balance losses
        self.sbert_model = sbert_model.to(self.args.device)  # Move SBERT to the same device as the model
        self.sbert_model.eval()  # Set SBERT to evaluation mode
    

    def compute_semantic_loss(self, generated_texts, target_texts):
        """
        Compute the semantic loss between generated and target rationales.
        """
        # Generate embeddings
        with torch.no_grad():
            generated_embeddings = self.sbert_model.encode(
                generated_texts,
                convert_to_tensor=True,
                device=self.args.device,
                show_progress_bar=False  # Disable progress bar
            )
            target_embeddings = self.sbert_model.encode(
                target_texts,
                convert_to_tensor=True,
                device=self.args.device,
                show_progress_bar=False  # Disable progress bar
            )

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(generated_embeddings, target_embeddings)

        # Semantic loss is 1 - cosine similarity
        semantic_loss = 1 - cosine_sim

        # Return the mean semantic loss
        return semantic_loss.mean()

    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute standard cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_ce = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Decode generated rationales for semantic loss
        # Generate outputs
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=80,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Decode target rationales
        # Extract the rationale from labels
        labels_ids = labels.cpu().numpy()
        target_texts = []
        for label in labels_ids:
            # Replace -100 with pad_token_id before decoding
            label = [id if id != -100 else self.tokenizer.pad_token_id for id in label]
            label_decoded = self.tokenizer.decode(label, skip_special_tokens=True)
            rationale = extract_rationale(label_decoded)
            target_texts.append(rationale)
        
        # Compute semantic loss
        semantic_loss = self.compute_semantic_loss(generated_texts, target_texts)
        
        # Combine losses
        total_loss = loss_ce + self.semantic_weight * semantic_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

# Define callbacks (reuse existing callbacks or define new ones if needed)
class SingleExampleEvalCallback(TrainerCallback):
    """
    A custom callback to perform evaluation on a single validation example at specified intervals.
    Logs the question, options, validation loss, and the generated answer and rationale.
    Additionally, computes and outputs the average percentage of correct answers for the last 50 questions.
    """
    def __init__(self, eval_steps, validation_dataset, tokenizer, is_stage1=False):
        self.eval_steps = eval_steps
        self.validation_dataset = validation_dataset
        self.current_val_idx = 0
        self.total_val_examples = len(validation_dataset)
        self.trainer = None  # Will be set after trainer initialization
        self.tokenizer = tokenizer
        self.is_stage1 = is_stage1
        self.last_50_correct_answers = deque(maxlen=50)  # Rolling deque to store the last 50 results

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.global_step % self.eval_steps == 0 and state.global_step != 0:
            trainer = self.trainer
            if trainer is not None:
                # Select the current validation example
                example = self.validation_dataset[self.current_val_idx]
                self.current_val_idx = (self.current_val_idx + 1) % self.total_val_examples

                # Decode the input to get question and options
                input_text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)

                # Replace -100 with pad_token_id for decoding labels
                label_ids = example['labels'].tolist()
                label_ids = [id if id != -100 else self.tokenizer.pad_token_id for id in label_ids]
                label_text = self.tokenizer.decode(label_ids, skip_special_tokens=True)

                # Prepare inputs
                input_ids = example['input_ids'].unsqueeze(0).to(trainer.args.device)
                attention_mask = example['attention_mask'].unsqueeze(0).to(trainer.args.device)
                labels = example['labels'].unsqueeze(0).to(trainer.args.device)

                # Evaluate the model
                model = trainer.model
                model.eval()
                with torch.no_grad():
                    # Compute loss
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss.item()

                    # Generate the model's answer with sampling enabled
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=80,               # Adjust as needed
                        num_beams=3,                 # Beam search for better quality
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        do_sample=True
                    )
                    generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Extract answers
                correct_answer = extract_answer(label_text)
                given_answer = extract_answer(generated_text)

                # Determine if the generated answer is correct
                is_correct = correct_answer == given_answer
                self.last_50_correct_answers.append(is_correct)  # Add result to rolling deque

                # Calculate the percentage of correct answers over the last 50 questions
                correct_percentage = (sum(self.last_50_correct_answers) / len(self.last_50_correct_answers)) * 100

                # Format the metrics
                training_loss = logs.get('loss', None)
                if training_loss is not None:
                    training_loss = f"{training_loss:.4f}"
                grad_norm = logs.get('grad_norm', None)
                if grad_norm is not None:
                    grad_norm = f"{grad_norm:.4f}"
                learning_rate = logs.get('learning_rate', None)
                if learning_rate is not None:
                    learning_rate = f"{learning_rate:.6f}"
                rolling_validation_loss = logs.get('rolling validation loss', None)
                if rolling_validation_loss is not None:
                    rolling_validation_loss = f"{rolling_validation_loss:.4f}"

                # Print the desired information to the console (Terminal Output)
                print(f"\n=== Step {state.global_step} ===")
                print(f"Training Loss: {training_loss}")
                print(f"Grad Norm: {grad_norm}")
                print(f"Learning Rate: {learning_rate}")
                print(f"Rolling Validation Loss: {rolling_validation_loss}")
                print(f"Input Question:\n{input_text}\n")

                if self.is_stage1:
                    print(f"Correct Answer: {correct_answer}")
                    print(f"Given Answer: {given_answer}\n")
                else:
                    print(f"Generated Answer and Rationale:\n{generated_text}\n")

                # Output the average percentage of correct answers for the last 50 validation questions
                print(f"Average Correct Answers (last 50): {correct_percentage:.2f}%\n")

                # Log the validation loss
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
            if 'loss' in logs:
                self.training_losses.append(logs['loss'])

            # Collect validation loss
            if 'validation loss' in logs:
                self.validation_losses.append(float(logs['validation loss']))

            # Compute rolling averages
            avg_training_loss = sum(self.training_losses) / len(self.training_losses) if len(self.training_losses) > 0 else None
            avg_validation_loss = sum(self.validation_losses) / len(self.validation_losses) if len(self.validation_losses) > 0 else None

            # Log the rolling averages with 4 decimal places
            if avg_training_loss is not None:
                avg_training_loss_formatted = f"{avg_training_loss:.4f}"
                logs['rolling training loss'] = avg_training_loss_formatted
            if avg_validation_loss is not None:
                avg_validation_loss_formatted = f"{avg_validation_loss:.4f}"
                logs['rolling validation loss'] = avg_validation_loss_formatted

            # Ensure the rolling averages are written into log history
            if state and state.log_history is not None:
                log_entry = {
                    'step': state.global_step,
                    'rolling training loss': float(avg_training_loss_formatted) if avg_training_loss else None,
                    'rolling validation loss': float(avg_validation_loss_formatted) if avg_validation_loss else None,
                }
                state.log_history.append(log_entry)
            
            # Print rolling loss logs for better visibility
            if avg_training_loss is not None and avg_validation_loss is not None:
                logger.info(f"Rolling Training Loss (last {self.window_size} steps): {avg_training_loss_formatted}")
                logger.info(f"Rolling Validation Loss (last {self.window_size} steps): {avg_validation_loss_formatted}")
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
    parser = argparse.ArgumentParser(description="Train Stage 2 of the T5 model with Semantic Loss on the AQuA-Rat dataset.")
    parser.add_argument(
        '--save_steps',
        type=int,
        default=150,
        help="Number of training steps between each model checkpoint save. Default is 150."
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=25,
        help="Number of training steps between each evaluation. Default is 25."
    )
    parser.add_argument(
        '--semantic_weight',
        type=float,
        default=0.5,
        help="Weight to balance cross-entropy loss and semantic loss. Default is 0.5."
    )
    parser.add_argument(
        '--stage1_model_path',
        type=str,
        default='./t5_aqua_rat_finetuned_stage1_final',
        help="Path to the Stage 1 trained model. Default is './t5_aqua_rat_finetuned_stage1_final'."
    )
    parser.add_argument(
        '--stage2_output_dir',
        type=str,
        default='./t5_aqua_rat_finetuned_stage2',
        help="Directory to save Stage 2 checkpoints. Default is './t5_aqua_rat_finetuned_stage2'."
    )
    args = parser.parse_args()
    if args.save_steps <= 0:
        parser.error("save_steps must be a positive integer.")
    if args.eval_steps <= 0:
        parser.error("eval_steps must be a positive integer.")
    if args.semantic_weight < 0:
        parser.error("semantic_weight must be non-negative.")
    return args

def load_preprocessed_data(save_path):
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
    preprocessed_data = torch.load(save_path)
    logger.info("Data loaded successfully.\n")

    # Convert dictionaries to Hugging Face Dataset
    data = {
        'input_ids': preprocessed_data['train']['input_ids'],
        'attention_mask': preprocessed_data['train']['attention_mask'],
        'labels': preprocessed_data['train']['labels'],
    }
    dataset = Dataset.from_dict(data)
    logger.info("Dataset created successfully.\n")

    # Set the dataset format to PyTorch tensors
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    logger.info("Dataset format set to PyTorch tensors.\n")

    return dataset

def initialize_model_and_tokenizer(model_name='t5-small', tokenizer_path='tokenizers/t5_tokenizer/', max_length=512, stage1_model_path='./t5_aqua_rat_finetuned_stage1_final'):
    """
    Initialize the tokenizer and model.

    Args:
        model_name (str): Name of the pre-trained model.
        tokenizer_path (str): Path where the tokenizer is saved.
        max_length (int): Maximum sequence length for inputs and outputs.
        stage1_model_path (str): Path to the Stage 1 trained model.

    Returns:
        tuple: (tokenizer, model)
    """
    logger.info(f"Loading the tokenizer from '{tokenizer_path}'...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, model_max_length=max_length)
    logger.info("Tokenizer loaded successfully.\n")

    logger.info(f"Loading the Stage 1 trained model from '{stage1_model_path}'...")
    model = T5ForConditionalGeneration.from_pretrained(stage1_model_path)
    logger.info("Stage 1 model loaded successfully.\n")

    # Resize token embeddings if new tokens have been added
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized token embeddings to match tokenizer.\n")

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    logger.info("Tokenizer and model initialized successfully.\n")
    return tokenizer, model

def get_latest_checkpoint(output_dir):
    """
    Get the latest checkpoint from the specified directory.

    Args:
        output_dir (str): Directory where checkpoints are saved.

    Returns:
        str or None: Path to the latest checkpoint or None if no checkpoint exists.
    """
    import re
    from glob import glob

    checkpoints = list(sorted(glob(os.path.join(output_dir, 'checkpoint-*')), key=lambda x: int(re.findall(r'checkpoint-(\d+)', x)[0])))
    if len(checkpoints) > 0:
        latest_checkpoint = checkpoints[-1]
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")
        return latest_checkpoint
    else:
        logger.warning("No checkpoint found. Starting training from scratch.")
        return None

def compute_metrics_factory(tokenizer):
    import evaluate
    rouge = evaluate.load("rouge")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        # Convert logits and labels to torch tensors
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Shift logits and labels for loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

        # Decode predictions and labels
        preds = torch.argmax(shift_logits, dim=-1)
        decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
        decoded_labels = []
        for label in labels:
            # Replace -100 with pad_token_id before decoding
            label_ids = label.cpu().numpy()
            label_ids = [id if id != -100 else tokenizer.pad_token_id for id in label_ids]
            decoded_label = tokenizer.decode(label_ids, skip_special_tokens=True)
            decoded_labels.append(decoded_label)

        # Compute accuracy for the answer part
        correct = 0
        total = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            # Extract the answer from the prediction and label
            pred_answer = extract_answer(pred)
            label_answer = extract_answer(label)
            if pred_answer == label_answer:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        # Extract rationales
        generated_rationales = [extract_rationale(pred) for pred in decoded_preds]
        reference_rationales = [extract_rationale(label) for label in decoded_labels]

        # Compute ROUGE scores for rationales
        rouge_output = rouge.compute(predictions=generated_rationales, references=reference_rationales)
        rouge_l = rouge_output['rougeL'].mid.fmeasure

        return {
            "validation loss": round(loss.item(), 4),
            "accuracy": round(accuracy, 4),
            "rougeL": round(rouge_l, 4),
        }

    return compute_metrics




def main():
    # ======================================
    # 1. Set Random Seed for Reproducibility
    # ======================================
    set_seed(42)

    # ======================================
    # 2. Parse Command-Line Arguments
    # ======================================
    args = parse_args()
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    semantic_weight = args.semantic_weight
    stage1_model_path = args.stage1_model_path
    stage2_output_dir = args.stage2_output_dir
    logger.info(f"Model checkpoint will be saved every {save_steps} steps.")
    logger.info(f"Evaluation will occur every {eval_steps} steps.")
    logger.info(f"Semantic loss weight set to {semantic_weight}.\n")

    # ======================================
    # 3. Load Preprocessed Stage 2 Data
    # ======================================
    stage2_path = 'dataset_stage2.pkl'  # Training data with answers and rationales
    logger.info("Loading Stage 2 preprocessed data (Answer and Rationale Generation)...")
    dataset_stage2 = load_preprocessed_data(stage2_path)

    # ======================================
    # 4. Initialize Tokenizer and Model from Stage 1
    # ======================================
    model_name = 't5-small'  # Ensure this matches your tokenizer
    tokenizer_path = 'tokenizers/t5_tokenizer/'  # Path where tokenizer is saved
    tokenizer, model = initialize_model_and_tokenizer(
        model_name=model_name,
        tokenizer_path=tokenizer_path,
        max_length=512,  # Adjust based on your data
        stage1_model_path=stage1_model_path
    )

    # ======================================
    # 5. Load the SBERT Model for Semantic Embeddings
    # ======================================
    logger.info("Loading SBERT model for semantic embeddings...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient
    sbert_model = sbert_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    sbert_model.eval()
    logger.info("SBERT model loaded successfully.\n")

    # ======================================
    # 6. Define Data Collator
    # ======================================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest',                             # Dynamic padding to save memory
        return_tensors='pt',
    )

    # ======================================
    # 7. Define Callbacks
    # ======================================
    rolling_average_callback = RollingAverageCallback(window_size=20)

    # ======================================
    # 8. Define Training Arguments for Stage 2
    # ======================================
    training_args_stage2 = TrainingArguments(
        output_dir=stage2_output_dir,                   # Directory to save Stage 2 checkpoints
        num_train_epochs=3,                              # Number of training epochs for Stage 2
        per_device_train_batch_size=2,                   # Batch size per device
        gradient_accumulation_steps=4,                   # To maintain effective batch size of 8
        gradient_checkpointing=True,                     # Enable gradient checkpointing
        warmup_steps=500,                                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                               # Strength of weight decay
        logging_dir='./t5_aqua_rat_logs_stage2',         # Directory for storing logs
        logging_steps=eval_steps,                        # Log training metrics every eval_steps
        save_steps=save_steps,                           # Save checkpoint every save_steps
        save_total_limit=5,                              # Keep only the latest 5 checkpoints
        eval_strategy='no',                              # Disable built-in evaluation for Stage 2
        save_strategy='steps',
        load_best_model_at_end=False,                    # Not required for Stage 2
        metric_for_best_model='validation loss',
        greater_is_better=False,                         # Lower validation loss is better
        fp16=torch.cuda.is_available(),                  # Use mixed precision if available
        per_device_eval_batch_size=1,                    # Reduce batch size to minimize memory usage
    )

    # ======================================
    # 9. Initialize the CustomTrainer for Stage 2
    # ======================================
    trainer_stage2 = CustomTrainer(
        model=model,
        args=training_args_stage2,
        train_dataset=dataset_stage2,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(tokenizer),
        callbacks=[],  # Initialize with empty list
        semantic_weight=semantic_weight,
        sbert_model=sbert_model,
    )

    # Add SingleExampleEvalCallback and RollingAverageCallback to Stage 2
    single_example_eval_callback_stage2 = SingleExampleEvalCallback(
        eval_steps=eval_steps,
        validation_dataset=dataset_stage2,
        tokenizer=tokenizer,
        is_stage1=False  # Indicating that this is Stage 2 (for answers + rationale)
    )
    single_example_eval_callback_stage2.trainer = trainer_stage2  # Associate with trainer
    trainer_stage2.add_callback(single_example_eval_callback_stage2)

    # Add RollingAverageCallback after SingleExampleEvalCallback
    trainer_stage2.add_callback(rolling_average_callback)

    # ======================================
    # 10. Get Latest Checkpoint for Stage 2
    # ======================================
    checkpoint_stage2 = get_latest_checkpoint(stage2_output_dir)
    if checkpoint_stage2:
        logger.info(f"Resuming Stage 2 training from checkpoint: {checkpoint_stage2}")
    else:
        logger.info("No Stage 2 checkpoint found. Starting training from scratch.")

    # ======================================
    # 11. Train Stage 2
    # ======================================
    trainer_stage2.train(resume_from_checkpoint=checkpoint_stage2)
    logger.info("Saving model after Stage 2")
    trainer_stage2.save_model(os.path.join(stage2_output_dir, 'final'))
    logger.info("Stage 2 training completed and model saved.\n")

    # ======================================
    # 12. Aggregate and Report Training Completion
    # ======================================
    logger.info("========== Stage 2 Training Completed ==========")
    logger.info(f"Final Stage 2 model is saved at '{os.path.join(stage2_output_dir, 'final')}'.")
    logger.info("You can now use this model for inference or further fine-tuning as needed.")

if __name__ == "__main__":
    main()
