import json, os, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_DIR = "model/adapter"
EVAL_DIR = "eval"
os.makedirs(EVAL_DIR, exist_ok=True)

with open("data/test.jsonl", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
PROMPTS = [e["instruction"] for e in data]
N = min(20, len(PROMPTS))

print(f"Loaded {len(PROMPTS)} test prompts, using first {N}\n")


def make_bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def generate(model, tokenizer, prompt):
    text = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(
        **inputs.to(model.device),
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()


def run_model(model_type):
    print(f"=== {model_type} ===")
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR if model_type == "finetuned" else MODEL_ID
    )
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_bnb(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR) if model_type == "finetuned" else base

    results = []
    for i in range(N):
        r = generate(model, tokenizer, PROMPTS[i])
        results.append({"prompt": PROMPTS[i], "response": r})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N}]")

    with open(f"{EVAL_DIR}/{model_type}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved {EVAL_DIR}/{model_type}.json\n")

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


run_model("base")
run_model("finetuned")

print(f"Done. Files saved to {EVAL_DIR}/")
print("Use eval/base.json and eval/finetuned.json for LLM-as-a-judge.")
