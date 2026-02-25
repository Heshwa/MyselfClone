import os
import re
import json
import zipfile
from pathlib import Path

# --- Configuration ---
DATA_DIR = "/Users/heshwa/Projects/MyselfClone/extracted_chats"
OUTPUT_FILE = "/Users/heshwa/Projects/MyselfClone/finetuning_dataset.jsonl"
YOUR_NAME = "Heshwa"
INSTRUCTION_TEMPLATE = f"How would {YOUR_NAME} respond?"

# Regex for WhatsApp format: "DD/MM/YY, HH:MM - Sender: Message" or "DD/MM/YYYY, HH:MM - Sender: Message"
# This also handles multi-line messages by checking if a line starts with a date pattern.
MSG_PATTERN = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\u202f[am|pm]{2})\s-\s([^:]+):\s(.*)$')

def clean_text(text):
    """Removes garbage like media omitted, missed calls, links, etc."""
    # List of strings that indicate a message should be skipped entirely
    garbage_patterns = [
        r"<Media omitted>",
        r"Missed voice call",
        r"Missed video call",
        r"This message was deleted",
        r"You deleted this message",
        r"Messages and calls are end-to-end encrypted",
        r"waiting for this message",
        r"sticker omitted",
        r"GIF omitted",
        r"audio omitted",
        r"video omitted",
        r"image omitted",
        r"Contact card omitted",
        r"Location: https://maps.google.com",
    ]
    
    # 1. Check for system messages or omitted media
    for pattern in garbage_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return None
            
    # 2. Skip messages containing URLs (links)
    url_pattern = r'http[s]?://'
    if re.search(url_pattern, text, re.IGNORECASE):
        return None

    return text.strip()

def parse_chat_file(file_path):
    conversations = []
    current_sender = None
    current_message_group = []
    
    parsed_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        match = MSG_PATTERN.match(line)
        if match:
            # New message starts
            _, sender, message = match.groups()
            sender = sender.strip()
            message = clean_text(message)
            
            if message:
                parsed_lines.append({"sender": sender, "text": message})
        else:
            # Multi-line message continuation
            if parsed_lines:
                cleaned_extra = clean_text(line)
                if cleaned_extra:
                    parsed_lines[-1]["text"] += " " + cleaned_extra

    # Group consecutive messages from same sender
    grouped_messages = []
    for line in parsed_lines:
        if not grouped_messages or grouped_messages[-1]["sender"] != line["sender"]:
            grouped_messages.append(line)
        else:
            grouped_messages[-1]["text"] += " " + line["text"]

    # Create Context-Response pairs with sliding window context (N-turns)
    WINDOW_SIZE = 5 # Number of previous messages to include
    for i in range(len(grouped_messages)):
        msg = grouped_messages[i]
        
        # We only create a training pair when Heshwa is the one responding
        if msg["sender"] == YOUR_NAME and i > 0:
            # Build context from previous messages
            context_parts = []
            start_idx = max(0, i - WINDOW_SIZE)
            for j in range(start_idx, i):
                prev_msg = grouped_messages[j]
                context_parts.append(f"{prev_msg['sender']}: {prev_msg['text']}")
            
            context_string = "\n".join(context_parts)
            
            conversations.append({
                "instruction": INSTRUCTION_TEMPLATE,
                "context": context_string,
                "response": msg["text"]
            })
            
    return conversations

def main():
    all_data = []
    data_path = Path(DATA_DIR)
    
    # Process all .txt files in the extracted folder
    txt_files = list(data_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {DATA_DIR}. Did you extract them?")
        return

    for txt_file in txt_files:
        print(f"Processing {txt_file.name}...")
        all_data.extend(parse_chat_file(txt_file))

    # Save to JSONL
    print(f"Saving {len(all_data)} pairs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("Done!")

if __name__ == "__main__":
    main()
