# Preprocessing Walkthrough: Turning WhatsApp Logs into a Dataset

This document explains how the `preprocess.py` script transforms your raw WhatsApp exports into a clean, multi-turn dataset for your "Second Brain" assistant.

## How it Works

### 1. Parsing & Multi-line Support
WhatsApp logs are tricky because messages can span multiple lines.
*   **The Regex**: We use a regular expression (`MSG_PATTERN`) to look for the date/time header at the start of a line.
*   **Logic**: If a line doesn't start with a date, the script assumes it's a continuation of the previous message and appends it.

### 2. The "Garbage Filter" (`clean_text`)
We filter out anything that doesn't represent your actual voice:
*   **Media Omissions**: Skips `<Media omitted>`, images, stickers, GIFs, and audio.
*   **System Messages**: Skips "Missed call", "Messages are encrypted", etc.
*   **Link Filter**: Skips any message containing a URL, as links don't help the model learn your personality.
*   **Deleted Messages**: Skips deletion notices.

### 3. Message Grouping
Often you or your friends send 3-4 short messages in a row.
*   **The Problem**: If we treated every single message as a separate turn, the context would be fragmented.
*   **The Solution**: The script merges consecutive messages from the same person into one large block.

### 4. Multi-Turn Context (The "Memory")
This is the most important part for a "Second Brain".
*   **Sliding Window**: We use a `WINDOW_SIZE` of 5.
*   **Process**: For every message you sent (`Heshwa`), the script looks backwards at the previous 5 turns (both yours and the other person's).
*   **Result**: The model doesn't just see the last question; it sees the flow of the conversation leading up to your response.

## Data Output Format
The result is a `jsonl` file where each line is an "example" for the model:
```json
{
  "instruction": "How would Heshwa respond?",
  "context": "Sender1: Dinner tonight?\nHeshwa: Where?\nSender1: That new place.",
  "response": "Yeah, sounds good! See you at 8."
}
```

## Running the Script
1.  Place your zipped WhatsApp chats in the `data/` folder.
2.  Run: `python3 preprocess.py`
3.  The clean dataset is saved as `finetuning_dataset.jsonl`.
