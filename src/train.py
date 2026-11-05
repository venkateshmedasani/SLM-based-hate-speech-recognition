"""Fine-tune TinyLlama-1.1B with LoRA for 3-way hate speech classification.

Usage:
    python src/train.py                      # full run (about 1 hour on an 8 GB GPU)
    python src/train.py --max-train 512 --epochs 1   # smoke test

Outputs:
    models/tinyllama-hatexplain-lora/   LoRA adapter + classification head + tokenizer
    results/tinyllama_lora/             metrics.json, confusion_matrix.png, test_predictions.csv
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import ID2LABEL, LABEL2ID, LABELS, clean_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--output-dir", default=str(ROOT / "models" / "tinyllama-hatexplain-lora"))
    p.add_argument("--results-dir", default=str(ROOT / "results" / "tinyllama_lora"))
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--max-train", type=int, default=None, help="subsample training set (debugging)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class FocalLossTrainer(Trainer):
    """Class-weighted focal loss: down-weights easy examples so the model
    spends more capacity on the hard hate/offensive boundary."""

    def __init__(self, *args, class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        weights = self.class_weights.to(logits.device)
        ce = F.cross_entropy(logits, labels, weight=weights, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
        loss = (((1 - pt) ** self.gamma) * ce).mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def load_split(name, max_rows=None, seed=42):
    df = pd.read_csv(ROOT / "data" / f"{name}.csv")
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=seed)
    df["text"] = df["text"].astype(str).map(clean_text)
    return df


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir, res_dir = Path(args.output_dir), Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train", args.max_train, args.seed)
    val_df = load_split("val", args.max_train and max(args.max_train // 4, 64), args.seed)
    test_df = load_split("test")
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    def to_ds(df):
        ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}), preserve_index=False)
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    counts = np.bincount(train_df["label"], minlength=len(LABELS))
    class_weights = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32)
    print("class weights:", dict(zip(LABELS, class_weights.round(decimals=3).tolist())))

    steps_per_epoch = max(len(train_ds) // (args.batch_size * args.grad_accum), 1)
    eval_steps = max(steps_per_epoch // 4, 10)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=int(0.06 * steps_per_epoch * args.epochs),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        gamma=args.focal_gamma,
    )

    train_result = trainer.train()

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved adapter -> {out_dir}")

    pred = trainer.predict(test_ds)
    y_pred = pred.predictions.argmax(-1)
    y_true = test_df["label"].to_numpy()

    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "base_model": args.base_model,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "per_class": {k: {m: round(v, 4) for m, v in report[k].items()} for k in LABELS},
        "confusion_matrix": cm.tolist(),
        "train_runtime_sec": round(train_result.metrics["train_runtime"], 1),
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("output_dir", "results_dir")},
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    out = test_df[["id", "text", "label"]].copy()
    out["predicted"] = y_pred
    probs = torch.softmax(torch.tensor(pred.predictions, dtype=torch.float32), -1).numpy()
    for i, name in enumerate(LABELS):
        out[f"p_{name}"] = probs[:, i].round(4)
    out.to_csv(res_dir / "test_predictions.csv", index=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"TinyLlama-1.1B + LoRA (test acc {metrics['accuracy']:.1%})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(res_dir / "confusion_matrix.png", dpi=150)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")}, indent=2))


if __name__ == "__main__":
    main()
