---
title: SLM Hate Speech Detector
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: TinyLlama-1.1B + LoRA hate speech classifier with explanations
models:
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0
datasets:
  - Hate-speech-CNERG/hatexplain
---

# SLM Hate Speech Detector

Interactive demo for [SLM-based hate speech recognition](https://github.com/venkateshmedasani/SLM-based-hate-speech-recognition):
TinyLlama-1.1B fine-tuned with LoRA on HateXplain to classify posts as hate speech, offensive, or normal,
with word-level occlusion explanations.
