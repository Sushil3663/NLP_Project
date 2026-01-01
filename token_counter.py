import json
import re
import torch
from tqdm import tqdm
from pathlib import Path
from collections import Counter

TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)

def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_vocab_from_lstm(lstm_ckpt_path):
    ckpt = torch.load(lstm_ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]

    # Case 1: vocab is a Vocab object
    if hasattr(vocab, "stoi"):
        return set(vocab.stoi.keys())

    # Case 2: vocab is a list (itos)
    if isinstance(vocab, (list, tuple)):
        return set(vocab)

    raise ValueError("Unsupported vocab format in lstm.pt")


def count_json_tokens_with_vocab(json_path, lstm_ckpt_path):
    json_path = Path(json_path)

    vocab_set = load_vocab_from_lstm(lstm_ckpt_path)

    in_vocab = Counter()
    oov = Counter()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in tqdm(data):
        texts = []

        if record.get("anchor"):
            texts.append(record["anchor"])

        texts.extend(record.get("positives", []))

        for text in texts:
            for tok in tokenize(text):
                if tok in vocab_set:
                    in_vocab[tok] += 1
                else:
                    oov[tok] += 1

    return {
        "total_tokens": sum(in_vocab.values()) + sum(oov.values()),
        "in_vocab_tokens": sum(in_vocab.values()),
        "oov_tokens": sum(oov.values()),
        "unique_in_vocab": len(in_vocab),
        "unique_oov": len(oov),
        "in_vocab_freqs": in_vocab,
        "oov_freqs": oov,
    }

if __name__=="__main__":
    stats = count_json_tokens_with_vocab(
        json_path="data/contrastive_anchors_positives.json",
        lstm_ckpt_path="models/ms_marco_emb/bilstm_best.pt"
    )

    print("Total tokens:", stats["total_tokens"])
    print("In-vocab tokens:", stats["in_vocab_tokens"])
    print("OOV tokens:", stats["oov_tokens"])
    print("Unique in-vocab:", stats["unique_in_vocab"])
    print("Unique OOV:", stats["unique_oov"])
