import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5Tokenizer
)
import os

def load_tokenizers():
    """
    Load the default T5 student tokenizer and teacher tokenizers (llemma and GPT-Neo) directly from Hugging Face.
    """
    # Load student tokenizer directly from the model repository
    t5_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
    
    # Load teacher tokenizers directly from the model repositories
    teacher_tokenizers = {
        'llemma': AutoTokenizer.from_pretrained('EleutherAI/llemma_7b'),
        'gptneo': AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
    }
    
    # Ensure pad_token and eos_token are set for teacher tokenizers
    for name, tokenizer in teacher_tokenizers.items():
        # Set eos_token if not already set
        if tokenizer.eos_token is None:
            if name == 'gptneo':
                tokenizer.add_special_tokens({'eos_token': ''})  # GPT-Neo's default eos_token
                print(f"Added eos_token for {name} tokenizer.")
            else:
                tokenizer.eos_token = tokenizer.pad_token  # Fallback for other models
                print(f"Set eos_token for {name} tokenizer to pad_token.")
        
        # Set pad_token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token for {name} tokenizer to eos_token.")
        
        # Verify pad_token_id and eos_token_id are set
        print(f"{name} tokenizer - pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
        print(f"{name} tokenizer - eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}\n")
    
    return t5_tokenizer, teacher_tokenizers

def load_models(device, teacher_tokenizers):
    """
    Load the student model (T5-small) and teacher models (llemma and GPT-Neo) directly from Hugging Face.
    """
    # Load student model
    try:
        student_model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-small').to(device)
        student_model.eval()
        print("Successfully loaded student model: T5-small")
    except Exception as e:
        print(f"Error loading student model: {e}")
        exit(1)
    
    # Define teacher model paths
    teacher_model_names = {
        'llemma': "EleutherAI/llemma_7b",
        'gptneo': "EleutherAI/gpt-neo-2.7B"
    }
    
    # Load teacher models
    teacher_models = {}
    for name, model_path in teacher_model_names.items():
        try:
            teacher_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            teacher_model.eval()
            teacher_models[name] = teacher_model
            print(f"Successfully loaded teacher model: {name}")
        except Exception as e:
            print(f"Error loading teacher model '{name}': {e}")
            exit(1)
    
    # Resize token embeddings if new tokens were added
    for name, tokenizer in teacher_tokenizers.items():
        if name in teacher_models:
            teacher_model = teacher_models[name]
            teacher_model.resize_token_embeddings(len(tokenizer))
            print(f"Resized token embeddings for {name} model.\n")
    
    return student_model, teacher_models

def generate_response(model, tokenizer, input_ids, attention_mask, max_new_tokens=256, min_length=100):
    """
    Generate a response from the given model and tokenizer with adjusted generation parameters.
    """
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,    # Allows more tokens in the output
            min_length=min_length,            # Ensures a minimum length for the response
            num_beams=5,                      # Beam search for better quality
            no_repeat_ngram_size=2,           # Prevents repeating n-grams
            early_stopping=False,             # Allows generation to continue until max_new_tokens
            temperature=0.7,                  # Controls randomness; lower is more deterministic
            top_p=0.9                          # Nucleus sampling for diversity
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Configuration
    device = "cpu"
    print(f"Using device: {device}\n")
    
    # Load tokenizers
    t5_tokenizer, teacher_tokenizers = load_tokenizers()
    
    # Load models
    student_model, teacher_models = load_models(device, teacher_tokenizers)
    
    # Define the test question
    test_question = (
        "Question: How many positive integers will divide evenly into 230? "
        "Options: ['A)4', 'B)6', 'C)8', 'D)12', 'E)16'] "
        "Please provide the answer and explain your reasoning."
    )
    
    print(f"Test Question:\n{test_question}\n")
    
    # Tokenize the input for each model
    # Student model (T5)
    student_encoding = t5_tokenizer(
        test_question,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=512
    ).to(device)
    
    # Teacher models
    teacher_encodings = {}
    for name, tokenizer in teacher_tokenizers.items():
        encoding = tokenizer(
            test_question,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=512
        ).to(device)
        teacher_encodings[name] = encoding
    
    # Generate and print responses
    print("Generating responses...\n")
    
    # Student model response
    student_response = generate_response(
        student_model,
        t5_tokenizer,
        student_encoding['input_ids'],
        student_encoding['attention_mask']
    )
    print("=== Student Model (T5-small) Response ===")
    print(student_response)
    print("\n" + "="*50 + "\n")
    
    # Teacher models responses
    for name, model in teacher_models.items():
        tokenizer = teacher_tokenizers[name]
        encoding = teacher_encodings[name]
        response = generate_response(
            model,
            tokenizer,
            encoding['input_ids'],
            encoding['attention_mask']
        )
        print(f"=== Teacher Model ({name}) Response ===")
        print(response)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
