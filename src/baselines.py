"""Classical TF-IDF baselines on the same train/val/test split as the SLM.

Usage:
    python src/baselines.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "baselines.json"


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }

    results = {}
    for name, clf in models.items():
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), clf)
        pipe.fit(train.text, train.label)
        pred = pipe.predict(test.text)
        results[name] = {
            "accuracy": round(accuracy_score(test.label, pred), 4),
            "macro_f1": round(f1_score(test.label, pred, average="macro"), 4),
        }
        print(f"{name:20s} acc={results[name]['accuracy']:.4f}  macro-F1={results[name]['macro_f1']:.4f}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
