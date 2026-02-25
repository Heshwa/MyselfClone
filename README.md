# MyselfClone: The "Second Brain" Fine-Tuned Assistant

This repository contains the tools and guides to fine-tune a Large Language Model (LLM) to mimic your conversational style using WhatsApp chat history.

## Project Overview
1.  **Preprocessing**: Convert raw WhatsApp `.txt` exports into a clean, multi-turn `jsonl` dataset.
2.  **Cleaning**: Automatically filter out "media omitted", stickers, links, and system messages.
3.  **Fine-Tuning**: Optimized guides for both NVIDIA GPUs (Unsloth) and Apple Silicon (MLX-LM).

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation
1.  Place your exported WhatsApp `.zip` files in the `data/` folder.
2.  Run the extraction command:
    ```bash
    mkdir -p extracted_chats && for zipfile in data/*.zip; do unzip -o "$zipfile" "*.txt" -d extracted_chats/; done
    ```
3.  Generate the dataset:
    ```bash
    python3 preprocess.py
    ```

### 2. Fine-Tuning
*   **For M1/M2/M3 Macs (8GB+ RAM)**: See [m1_training_guide.md](m1_training_guide.md).
*   **For NVIDIA GPUs / Google Colab**: See [train_config.md](train_config.md).

## Repository Structure
*   `preprocess.py`: Main data processing script.
*   `m1_training_guide.md`: Detailed guide for Apple Silicon optimization.
*   `train_config.md`: General training configuration and script.
*   `walkthrough.md`: Technical explanation of how the preprocessing works.
