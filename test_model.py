from unsloth import FastLanguageModel
import torch

# --- Configuration ---
MODEL_PATH = "hes_brain_l40s_final" # Path to your saved model/adapter
INSTRUCTION_TEMPLATE = "How would Heshwa respond?"

# 1. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable 2x faster inference

import json
import random

def generate_response(context):
    prompt = f"### Instruction:\n{INSTRUCTION_TEMPLATE}\n\n### Context:\n{context}\n\n### Response:"
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 128,
        use_cache = True,
        temperature = 0.7,
        top_p = 0.9,
    )
    
    response = tokenizer.batch_decode(outputs)[0]
    return response.split("### Response:")[-1].strip()

def run_sample_test(dataset_path, num_samples=5):
    print(f"\n--- Running Sample Test ({num_samples} random samples) ---")
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    samples = random.sample(data, min(num_samples, len(data)))
    
    for i, sample in enumerate(samples):
        print(f"\n[Sample {i+1}]")
        print(f"Context:\n{sample['context']}")
        print(f"\nOriginal Response: {sample['response']}")
        
        ai_response = generate_response(sample['context'])
        print(f"AI Response: {ai_response}")
        print("-" * 30)

if __name__ == "__main__":
    print("--- MyselfClone Inference & Testing ---")
    
    choice = input("Enter '1' for Interactive Chat or '2' for Sample Test: ")
    
    if choice == '1':
        print("\nInteractive Chat Mode. Type 'quit' to exit.")
        history = []
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
                
            history.append(f"Friend: {user_input}")
            context = "\n".join(history[-5:])
            
            response = generate_response(context)
            print(f"\nHeshwa (AI): {response}")
            history.append(f"Heshwa: {response}")
    elif choice == '2':
        run_sample_test("finetuning_dataset.jsonl")
    else:
        print("Invalid choice.")
