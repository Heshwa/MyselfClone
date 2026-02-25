## 1. Environment Setup
*   **High-End GPU (Recommended)**: NVIDIA L40S, A100, or H100.
*   **WandB**: Create an account at [wandb.ai](https://wandb.ai) and run `wandb login` in your terminal.

## 2. L40S Optimized Script (`train_l40s.py`)
I have created a specialized script for the L40S that enables:
*   **Flash Attention 2**: For significantly faster training.
*   **bf16 Support**: Higher precision training than standard fp16.
*   **WandB Logging**: Live monitoring of your loss and gradients.

### Run Command:
```bash
python3 train_l40s.py
```

### Installation
```bash
pip install unsloth
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

## 2. Training Script Template
Save this as `train.py` or run it in a Jupyter Notebook.

```python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load Model (Llama 3.1 8B is recommended)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 3. Load Dataset
dataset = load_dataset("json", data_files="finetuning_dataset.jsonl", split="train")

def format_prompts(examples):
    instructions = examples["instruction"]
    contexts = examples["context"]
    responses = examples["response"]
    texts = []
    for instruction, context, response in zip(instructions, contexts, responses):
        # Formatting the prompt for Llama 3 style
        text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched = True)

# 4. Define Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Adjust based on dataset size (e.g. 1 epoch)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

# 5. Train!
trainer.train()

# 6. Save the model
model.save_pretrained("hes_brain_model")
tokenizer.save_pretrained("hes_brain_model")
```

## 3. Key Parameters to Tune
*   **max_steps**: Start with a small number (e.g., 100) to see if it learns your style, then increase it.
*   **r (Rank)**: Lower (8-16) is faster and uses less memory. Higher (32-64) can capture more nuances.
*   **Learning Rate**: `2e-4` is standard, but you can try `5e-5` for more subtle adjustments.

## 4. How to Test
After training, you can use the model to generate responses. I can provide an inference script next once you've finished training!
