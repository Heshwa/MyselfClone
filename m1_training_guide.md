# Fine-Tuning on M1 Mac (8GB RAM)

Fine-tuning a Llama 3 8B model on 8GB of RAM is **challenging but possible** using Apple's **MLX** framework. Standard libraries like `unsloth` or `transformers` are designed for NVIDIA GPUs and will likely fail on your machine.

> [!IMPORTANT]
> Because you have 8GB RAM, the system will use **Swap Memory** (your SSD) during training. This will works, but it will be slower than a higher-RAM machine.

## 1. Install MLX
MLX is built by Apple for Apple Silicon. It is the most efficient way to do this.

```bash
pip install mlx-lm
```

## 2. Prepare Data for MLX
MLX expects a specific format. You need to convert your `finetuning_dataset.jsonl` into three files: `train.jsonl`, `valid.jsonl`, and `test.jsonl`.

### Conversion Script (`prepare_mlx.py`)
```python
import json
import random

with open("finetuning_dataset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Shuffle and split
random.shuffle(data)
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
valid_data = data[train_size:]

def save_mlx(data, filename):
    with open(filename, "w") as f:
        for item in data:
            # MLX format: {"text": "### Instruction... ### Context... ### Response..."}
            text = f"### Instruction:\n{item['instruction']}\n\n### Context:\n{item['context']}\n\n### Response:\n{item['response']}"
            f.write(json.dumps({"text": text}) + "\n")

save_mlx(train_data, "train.jsonl")
save_mlx(valid_data, "valid.jsonl")
print(f"Prepared {len(train_data)} training and {len(valid_data)} validation samples.")
```

## 3. Start Fine-Tuning (4-bit QLoRA)
Run this command in your terminal. We will use **Llama-3.2-1B** or **3.2-3B** instead of 8B for better stability on 8GB RAM.

### Recommended Command:
```bash
mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train \
  --data . \
  --batch-size 1 \
  --iters 200 \
  --lr 1e-5 \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --adapter-path ./adapters
```

### Why these settings?
*   **Model**: Using the `3B` or `1B` model is much safer for 8GB RAM than the `8B` model.
*   **Batch Size 1**: Minimizes the memory spike.
*   **4-bit Model**: Reduces the model size from ~12GB to ~2GB.

## 4. Testing Your Model
Once finished, run this to talk to your fine-tuned assistant:
```bash
mlx_lm.generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path ./adapters \
  --prompt "### Instruction:\nHow would Heshwa respond?\n\n### Context:\nFriend: Are you coming to the party tonight?\n\n### Response:" \
  --max-tokens 50
```
