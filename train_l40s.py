from unsloth import FastLanguageModel
import torch
import os
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import wandb

# --- Configuration ---
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
DATASET_PATH = "finetuning_dataset.jsonl"
OUTPUT_DIR = "hes_brain_l40s"
WANDB_PROJECT = "SecondBrain-Heshwa"

# 1. Initialize WandB
wandb.init(project=WANDB_PROJECT, name="llama3.1-8b-l40s-finetune")

# 2. Load Model & Tokenizer
# L40S supports bfloat16 and Flash Attention 2
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Higher rank for more nuance on powerful hardware
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
)

# 4. Load & Format Dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_prompts(examples):
    instructions = examples["instruction"]
    contexts = examples["context"]
    responses = examples["response"]
    texts = []
    for instruction, context, response in zip(instructions, contexts, responses):
        text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched = True)

# 5. Define Trainer with L40S Optimized Args
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 8, # L40S can handle higher batch sizes
        gradient_accumulation_steps = 2,
        warmup_steps = 10,
        max_steps = 200, 
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), # Use bf16 on L40S
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        report_to = "wandb", # Enable WandB logging
    ),
)

# 6. Train!
trainer.train()

# 7. Save the model
model.save_pretrained(f"{OUTPUT_DIR}_final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_final")

wandb.finish()
