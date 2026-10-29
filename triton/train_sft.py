import argparse
import hashlib
import json
import pathlib
import random
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-1.7B"
N_HELD_OUT_EXAMPLES = 15  # in-domain rows set aside for verification; never trained on
EPOCHS = 1
SFT_BATCH_SIZE = 2
PEAK_LR = 2e-4
SCHEDULE_TYPE = "linear"
WARMUP_RATIO = 0.1
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
ALLOWED_ROLES = {"system", "user", "assistant"}

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="training JSONL ({'messages': [...]} per line)")
parser.add_argument("--run-dir", required=True)
parser.add_argument("--n-train", type=int, default=6000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=PEAK_LR, help="peak learning rate")
args = parser.parse_args()

run_dir = pathlib.Path(args.run_dir)
lora_dir, results_dir = run_dir / "lora", run_dir / "results"
lora_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
random.seed(args.seed)
torch.manual_seed(args.seed)

# --- Section 2: validate and split ---
rows, seen, n_duplicates = [], set(), 0
for line_number, line in enumerate(open(args.data, encoding="utf-8"), start=1):
    if not line.strip():
        continue
    obj = json.loads(line)
    messages = obj["messages"]
    assert len(messages) >= 2 and messages[-1]["role"] == "assistant", f"line {line_number}: last message must be the assistant"
    assert any(m["role"] == "user" for m in messages), f"line {line_number}: needs a user message"
    assert all(m["role"] in ALLOWED_ROLES and m["content"].strip() for m in messages), f"line {line_number}: bad message"
    digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    n_duplicates += digest in seen
    seen.add(digest)
    # source_id groups paraphrases of the same seed row; lines without one are their own group.
    rows.append({"messages": messages, "source_id": obj.get("source_id") or f"line{line_number}"})

rng = random.Random(args.seed)
rng.shuffle(rows)

# Hold out whole source groups: one prompt per held-out group, and none of that group's
# paraphrases are trained on, so the in-domain check isn't measuring memorised near-duplicates.
source_ids = list(dict.fromkeys(row["source_id"] for row in rows))
assert len(source_ids) > N_HELD_OUT_EXAMPLES, "Not enough source groups to both hold out and train on."
held_out_ids = set(source_ids[:N_HELD_OUT_EXAMPLES])
held_out = list({row["source_id"]: row for row in reversed(rows) if row["source_id"] in held_out_ids}.values())
train_rows = [row for row in rows if row["source_id"] not in held_out_ids][: args.n_train]

with open(results_dir / "held_out_in_domain.jsonl", "w", encoding="utf-8") as f:
    for row in held_out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
# Section 5: training file holds messages only.
with open(lora_dir / "train.jsonl", "w", encoding="utf-8") as f:
    for row in train_rows:
        f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
print(f"Valid rows: {len(rows):,} | exact duplicates: {n_duplicates} | source groups: {len(source_ids)} "
      f"| held out: {len(held_out)} | training: {len(train_rows):,}")

# --- Section 3: model + LoRA. fp16 on V100 (no native bf16), bf16 on Ampere and newer. ---
bf16 = torch.cuda.get_device_capability()[0] >= 8
print("GPU:", torch.cuda.get_device_name(0), "| precision:", "bf16" if bf16 else "fp16")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16 if bf16 else torch.float16, attn_implementation="sdpa", device_map="auto"
)
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# --- Section 6: train with response-only loss ---
train_dataset = load_dataset("json", data_files=str(lora_dir / "train.jsonl"), split="train")
total_steps = max(1, len(train_dataset) // SFT_BATCH_SIZE) * EPOCHS
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=str(lora_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        learning_rate=args.lr,
        lr_scheduler_type=SCHEDULE_TYPE,
        warmup_steps=max(1, int(total_steps * WARMUP_RATIO)),
        fp16=not bf16,
        bf16=bf16,
        assistant_only_loss=True,  # mask everything except the assistant's reply
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    ),
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# Confirms assistant_only_loss is really masking the prompt before a full run.
batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]
masked = int((labels == -100).sum())
print(f"Masked tokens: {masked}/{labels.numel()} (some assistant tokens must remain unmasked)")
assert 0 < masked < labels.numel(), "assistant_only_loss is not masking correctly"
print("Trained-on text of the first example:",
      repr(tokenizer.decode(batch["input_ids"][0][labels != -100])[:300]))

start = time.time()
trainer.train(resume_from_checkpoint=any(lora_dir.glob("checkpoint-*")))
print(f"Training time: {(time.time() - start) / 60:.1f} min")
trainer.save_model(str(lora_dir))
tokenizer.save_pretrained(str(lora_dir))
json.dump(trainer.state.log_history, open(results_dir / "train_log.json", "w"), indent=1)
print("Saved LoRA adapter to:", lora_dir)
