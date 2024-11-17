from transformers import T5Tokenizer
from huggingface_hub import login

def main():
    login("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")
    tokenizer = T5Tokenizer.from_pretrained("tokenizers/t5-base")
    print("Token IDs for '<1>':", tokenizer.encode('1', add_special_tokens=False))
    print("Token IDs for '<2>':", tokenizer.encode('2', add_special_tokens=False))
    print("Token IDs for '<3>':", tokenizer.encode('3', add_special_tokens=False))
    print(f"Vocabulary Size: {len(tokenizer)}")

if __name__ == "__main__":
    main()
