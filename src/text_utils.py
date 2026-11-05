"""Text normalisation shared by training and the demo.

HateXplain posts are lower-cased, punctuation-stripped and use placeholder
tokens such as <user> and <number>. Raw user input has to be mapped into the
same space, otherwise the model sees a distribution it was never trained on.
"""

import re

LABELS = ["hate", "normal", "offensive"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_APOSTROPHE = re.compile(r"['\u2019`]")
_PUNCT = re.compile(r"[!\"#$%&()*+,\-./:;=?\[\\\]^_{|}~\u201c\u201d\u2014\u2026]")
_PLACEHOLDER = re.compile(r"<([a-z]+)>")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = _URL.sub(" <url> ", text)
    text = _MENTION.sub(" <user> ", text)
    text = _APOSTROPHE.sub("", text)
    text = _NUMBER.sub(" <number> ", text)
    # Protect placeholders from the punctuation pass, which would split "<" and ">".
    text = _PLACEHOLDER.sub(lambda m: f" \x00{m.group(1)}\x01 ", text)
    text = text.replace("<", " ").replace(">", " ")
    text = _PUNCT.sub(" ", text)
    text = text.replace("\x00", "<").replace("\x01", ">")
    return _SPACES.sub(" ", text).strip()
