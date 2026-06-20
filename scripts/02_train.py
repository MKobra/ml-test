import json
import os
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_TRAIN = "data/train.jsonl"
ADAPTER_DIR = "model/adapter"
MAX_LENGTH = 512
os.makedirs(ADAPTER_DIR, exist_ok=True)

print(f"\nLoading {MODEL_ID}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading training data...")
with open(DATA_TRAIN, encoding="utf-8") as f:
    raw = [json.loads(line) for line in f]
dataset = Dataset.from_list(raw)


def format_chat(example):
    text = (
        f"<s>[INST] {example['instruction']} [/INST]"
        f"{example['response']}</s>"
    )
    return {"text": text}


dataset = dataset.map(format_chat)
print(f"Train examples: {len(dataset)}")

wandb.init(
    project="ultrafeedback-sft",
    name="mistral-7b-qlora-ultrafeedback-1000",
    config={
        "model": MODEL_ID,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "epochs": 3,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "effective_batch_size": 16,
        "max_length": MAX_LENGTH,
        "dataset": "argilla/ultrafeedback-binarized-preferences-cleaned",
        "train_examples": len(dataset),
    },
)

training_args = SFTConfig(
    output_dir=ADAPTER_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
    run_name="mistral-7b-qlora-gazeta",
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    max_length=MAX_LENGTH,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
)

print("\nStarting training...\n")
trainer.train()
print("\nTraining finished!")

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\nAdapter saved to {ADAPTER_DIR}/")

wandb.finish()
print("\nDone.")
