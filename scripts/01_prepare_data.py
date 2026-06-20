import json
import os
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs("data", exist_ok=True)

TRAIN_SIZE = 1000
TEST_SIZE = 100
MIN_RATING = 4.5

print("Loading UltraFeedback (cleaned)...")
ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")

print(f"Total rows: {len(ds)}")
print(f"Columns: {ds.column_names}")

filtered = ds.filter(lambda x: x["chosen-rating"] >= MIN_RATING)
print(f"After rating >= {MIN_RATING}: {len(filtered)}")

prompts_seen = set()
train_data, test_data = [], []

for item in filtered:
    prompt = item["prompt"]
    if prompt in prompts_seen:
        continue
    prompts_seen.add(prompt)

    chosen_messages = item["chosen"]
    response = chosen_messages[-1]["content"]

    record = {
        "instruction": prompt,
        "response": response,
        "rating": item["chosen-rating"],
        "source": item["source"],
    }

    if len(train_data) < TRAIN_SIZE:
        train_data.append(record)
    elif len(test_data) < TEST_SIZE:
        test_data.append(record)
    else:
        break

print(f"Training examples:  {len(train_data)}")
print(f"Test examples:      {len(test_data)}")

# Count sources in train
from collections import Counter
src_counts = Counter(r["source"] for r in train_data)
print(f"Train sources: {dict(src_counts)}")

for path, data in [("data/train.jsonl", train_data), ("data/test.jsonl", test_data)]:
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps({"instruction": r["instruction"], "response": r["response"]}, ensure_ascii=False) + "\n")
    print(f"  Saved {path} ({len(data)} examples)")

print("Done.")
