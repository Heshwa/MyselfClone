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
    # Extract only the response part after the prompt
    return response.split("### Response:")[-1].strip()

if __name__ == "__main__":
    print("--- MyselfClone Inference ---")
    print("Type 'quit' to exit.")
    
    history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        history.append(f"Friend: {user_input}")
        # Keep only last 5 turns for context
        context = "\n".join(history[-5:])
        
        response = generate_response(context)
        print(f"\nHeshwa (AI): {response}")
        history.append(f"Heshwa: {response}")
