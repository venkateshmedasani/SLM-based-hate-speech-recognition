"""Gradio demo: TinyLlama-1.1B + LoRA hate speech classifier with word-level explanations.

Run locally:
    python demo/app.py

Environment variables:
    MODEL_ID   Hugging Face repo id or local path of the LoRA adapter.
               Defaults to the local training output if present, otherwise the Hub repo.
"""

import os
import sys
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
# On Hugging Face Spaces text_utils.py sits next to app.py; locally it lives in src/.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from text_utils import ID2LABEL, LABEL2ID, LABELS, clean_text  # noqa: E402

LOCAL_ADAPTER = HERE.parent / "models" / "tinyllama-hatexplain-lora"
HUB_ADAPTER = "venkateshmedasani/tinyllama-hatexplain-lora"
MODEL_ID = os.environ.get("MODEL_ID") or (str(LOCAL_ADAPTER) if LOCAL_ADAPTER.exists() else HUB_ADAPTER)
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 128
MAX_EXPLAIN_WORDS = 40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

DISPLAY = {"hate": "Hate speech", "offensive": "Offensive", "normal": "Normal"}


def load_model():
    print(f"Loading base={BASE_MODEL} adapter={MODEL_ID} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID, dtype=DTYPE
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, MODEL_ID).merge_and_unload()
    return tokenizer, model.to(DEVICE).eval()


tokenizer, model = load_model()


@torch.inference_mode()
def class_probs(texts):
    enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(DEVICE)
    return torch.softmax(model(**enc).logits.float(), dim=-1).cpu()


def explain(words, probs, target):
    """Occlusion: a word's importance is how much the predicted-class probability
    drops when that word is removed."""
    if len(words) < 2:
        return [(w, None) for w in words]
    variants = [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
    drops = probs[target] - class_probs(variants)[:, target]
    scale = drops.abs().max().clamp(min=1e-6)
    return [(w, round(float(d / scale), 3)) for w, d in zip(words, drops)]


def classify(text):
    text = (text or "").strip()
    if not text:
        return {}, [], ""
    cleaned = clean_text(text)
    if not cleaned:
        return {}, [], "Nothing left to classify after normalisation."
    probs = class_probs([cleaned])[0]
    target = int(probs.argmax())
    label_out = {DISPLAY[LABELS[i]]: float(p) for i, p in enumerate(probs)}

    words = cleaned.split()
    if len(words) <= MAX_EXPLAIN_WORDS:
        highlights = explain(words, probs, target)
        note = (
            f"Red words push the model **towards** *{DISPLAY[LABELS[target]]}*; "
            "blue words push it away. Scores come from removing each word and re-scoring."
        )
    else:
        highlights = [(cleaned, None)]
        note = f"Explanation skipped: input is longer than {MAX_EXPLAIN_WORDS} words."
    return label_out, highlights, f"**Model input after normalisation:** `{cleaned}`\n\n{note}"


EXAMPLES = [
    "Have a great day everyone, thanks for all the support!",
    "I love my gay neighbours, they are the nicest people on the street",
    "shut up bitch nobody asked you",
    "Those immigrants are vermin and should be thrown out of our country",
    "muslims are terrorists and should all be deported",
    "Women don't belong in tech, go back to the kitchen",
]

DESCRIPTION = """
# Hate Speech Detection with a Small Language Model

**TinyLlama-1.1B** fine-tuned with **LoRA** (about 1% of weights trained) on the
[HateXplain](https://github.com/hate-alert/HateXplain) dataset (20k Twitter and Gab posts)
to classify text as **Hate speech**, **Offensive**, or **Normal**.

Type a post or pick an example. The word highlights show which words drove the decision.

[GitHub repo](https://github.com/venkateshmedasani/SLM-based-hate-speech-recognition)
"""

DISCLAIMER = """
<small>Research demo only. The model reflects the biases and labelling choices of HateXplain
and makes mistakes, particularly between *hate* and *offensive*. Inputs are not stored.
Running on a free CPU, so the first request after the Space wakes up can take a little longer.</small>
"""

with gr.Blocks(title="SLM Hate Speech Detector") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Post", lines=4, placeholder="Type a social-media post...")
            with gr.Row():
                clear = gr.ClearButton(value="Clear")
                btn = gr.Button("Classify", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column(scale=1):
            label = gr.Label(label="Prediction", num_top_classes=3)
            highlights = gr.HighlightedText(label="Word importance", show_legend=False)
            info = gr.Markdown()
    gr.Markdown(DISCLAIMER)

    btn.click(classify, inputs=inp, outputs=[label, highlights, info])
    inp.submit(classify, inputs=inp, outputs=[label, highlights, info])
    clear.add([inp, label, highlights, info])

if __name__ == "__main__":
    demo.launch()
