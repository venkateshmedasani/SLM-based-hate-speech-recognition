"""Build the comparison tables and figures used in the README.

Usage:
    python scripts/analyze_results.py

Reads the saved test-set predictions of every model and writes
results/analysis.json plus figures in assets/.
"""

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from text_utils import LABELS  # noqa: E402

RES = ROOT / "results"
ASSETS = ROOT / "assets"
COLORS = {"Llama3-8B zero-shot": "#b0b0b0", "TF-IDF + LogReg": "#7fa7d9", "TinyLlama-1.1B + LoRA": "#d9534f"}


def load_predictions():
    test = pd.read_csv(ROOT / "data" / "test.csv")
    zero = pd.read_csv(RES / "zero_shot_llama3" / "results_test_llama3.csv")
    base = pd.read_csv(RES / "baselines_test_predictions.csv")
    slm = pd.read_csv(RES / "tinyllama_lora" / "test_predictions.csv")
    assert (test.id.values == zero.id.values).all() and (test.id.values == slm.id.values).all()
    preds = {
        "Llama3-8B zero-shot": zero.predicted_label.to_numpy(),
        "TF-IDF + LogReg": base["Logistic Regression"].to_numpy(),
        "TinyLlama-1.1B + LoRA": slm.predicted.to_numpy(),
    }
    return test, preds


def dataset_stats():
    rows = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(ROOT / "data" / f"{split}.csv")
        counts = df.label_name.value_counts()
        rows[split] = {
            "posts": len(df),
            **{name: int(counts.get(name, 0)) for name in LABELS},
            "avg_words": round(df.text_length.mean(), 1),
        }
    return rows


def per_class(test, preds):
    out = {}
    for name, p in preds.items():
        prec, rec, f1, _ = precision_recall_fscore_support(test.label, p, labels=range(3), zero_division=0)
        out[name] = {
            "accuracy": round(accuracy_score(test.label, p), 4),
            "macro_f1": round(f1_score(test.label, p, average="macro"), 4),
            **{
                LABELS[i]: {"precision": round(prec[i], 4), "recall": round(rec[i], 4), "f1": round(f1[i], 4)}
                for i in range(3)
            },
        }
    return out


def by_agreement(test, preds):
    groups = {"3/3 annotators agree": 1.0, "2/3 agree": 2 / 3, "no majority (1/3)": 1 / 3}
    out = {}
    for label, value in groups.items():
        mask = np.isclose(test.confidence, value, atol=0.01)
        out[label] = {"posts": int(mask.sum())}
        for name, p in preds.items():
            out[label][name] = round(float((p[mask] == test.label.values[mask]).mean()), 4)
    return out


def by_target_group(test, preds, top=6):
    targets = test.targets.map(ast.literal_eval)
    counts = Counter(t for ts in targets for t in ts if t not in ("None", "Other"))
    out = {}
    for group, _ in counts.most_common(top):
        mask = targets.map(lambda ts: group in ts).to_numpy()
        out[group] = {"posts": int(mask.sum())}
        for name, p in preds.items():
            out[group][name] = round(float((p[mask] == test.label.values[mask]).mean()), 4)
    return out


def confusions(test, pred):
    out = Counter()
    for t, p in zip(test.label, pred):
        if t != p:
            out[f"{LABELS[t]} -> {LABELS[p]}"] += 1
    total = sum(out.values())
    return {k: {"count": v, "share_of_errors": round(v / total, 3)} for k, v in out.most_common()}


def plot_training_curve():
    state = json.loads((RES / "tinyllama_lora" / "trainer_state.json").read_text())
    log = state["log_history"]
    train = [(e["epoch"], e["loss"]) for e in log if "loss" in e]
    evals = [(e["epoch"], e["eval_accuracy"], e["eval_macro_f1"]) for e in log if "eval_macro_f1" in e]

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(*zip(*train), color="#999", lw=1, label="train loss (focal)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss", color="#666")
    ax2 = ax1.twinx()
    ep, acc, f1 = zip(*evals)
    ax2.plot(ep, acc, "o-", color="#337ab7", label="val accuracy")
    ax2.plot(ep, f1, "s-", color="#d9534f", label="val macro F1")
    ax2.set_ylabel("validation score")
    ax2.set_ylim(0.5, 0.72)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax2.legend(lines, labels, loc="lower right")
    ax1.set_title("TinyLlama-1.1B + LoRA fine-tuning on HateXplain")
    fig.tight_layout()
    fig.savefig(ASSETS / "training_curve.png", dpi=150)
    plt.close(fig)


def plot_per_class(pc):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.27
    x = np.arange(len(LABELS) + 1)
    for i, (name, m) in enumerate(pc.items()):
        vals = [m[label]["f1"] for label in LABELS] + [m["macro_f1"]]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=COLORS[name])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [label.capitalize() for label in LABELS] + ["Macro F1"])
    ax.set_ylabel("F1 score (test set)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)
    ax.set_title("Per-class F1 on 2,015 held-out posts")
    fig.tight_layout()
    fig.savefig(ASSETS / "per_class_f1.png", dpi=150)
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    test, preds = load_predictions()
    pc = per_class(test, preds)
    analysis = {
        "dataset": dataset_stats(),
        "per_class": pc,
        "accuracy_by_annotator_agreement": by_agreement(test, preds),
        "accuracy_by_target_group": by_target_group(test, preds),
        "tinyllama_error_breakdown": confusions(test, preds["TinyLlama-1.1B + LoRA"]),
    }
    (RES / "analysis.json").write_text(json.dumps(analysis, indent=2))
    plot_training_curve()
    plot_per_class(pc)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
