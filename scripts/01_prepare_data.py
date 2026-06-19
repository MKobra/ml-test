import json
import os
import re

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs("data", exist_ok=True)


def is_russian(text):
    cyrillic = len(re.findall(r'[а-яА-ЯёЁ]', text))
    total = len(re.findall(r'[а-яА-ЯёЁa-zA-Z]', text))
    return total > 0 and (cyrillic / total) > 0.5


def to_alpaca(item):
    conv = item["conversation"]
    user_text = ""
    assistant_text = ""
    for msg in conv:
        if msg["role"] == "user":
            user_text = msg["content"]
        elif msg["role"] == "assistant":
            assistant_text = msg["content"]
    return {"instruction": user_text, "response": assistant_text}


print("Loading dataset...")
from datasets import load_dataset

dataset = load_dataset(
    "ZeroAgency/ru-big-russian-dataset",
    split="train",
    streaming=True,
).select_columns(["topic", "conversation", "overall_score", "question"])

print("Scanning for cooking examples...")
found = []
seen_questions = set()

for batch in dataset.iter(batch_size=1000):
    topics = batch["topic"]
    scores = batch["overall_score"]
    conversations = batch["conversation"]
    questions = batch["question"]

    for i in range(len(topics)):
        if topics[i] == "cooking" and scores[i] is not None and scores[i] >= 8:
            q = questions[i]
            if q and q not in seen_questions:
                seen_questions.add(q)
                text = conversations[i][0]["content"] + " " + conversations[i][1]["content"]
                if is_russian(text):
                    found.append({"question": q, "conversation": conversations[i], "overall_score": scores[i]})
                    if len(found) >= 500:
                        break
    if len(found) >= 500:
        break

print(f"Found {len(found)} examples")

found.sort(key=lambda x: x["overall_score"], reverse=True)

path = "data/train.jsonl"
with open(path, "w", encoding="utf-8") as f:
    for item in found[:500]:
        f.write(json.dumps(to_alpaca(item), ensure_ascii=False) + "\n")
print(f"  data/train.jsonl ({min(500, len(found))} examples)")
print("\nDone.")
