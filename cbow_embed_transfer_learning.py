# Takes the mlm_lstm's vocab to train a cbow model
# the embedding trained is transfered to the mlm_lstm (2nd experiment)

import torch
import json
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import re
import os

# 1. Load checkpoint
CKPT_PATH = "models/mlm_bilstm/bilstm_mlm_epoch3.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")

vocab_itos = ckpt["vocab"]
stoi = {t: i for i, t in enumerate(vocab_itos)}

# Special tokens (match training)
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

SPECIAL_TOKENS = [CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, PAD_TOKEN, UNK_TOKEN]
TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def build_sentences(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences = []
    for item in data:
        for t in [item.get("anchor")] + item.get("positives", []):
            if t:
                tokens = tokenize(t)
                tokens = [tok if tok in stoi else UNK_TOKEN for tok in tokens]
                tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
                sentences.append(tokens)
    return sentences

class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.prev_loss = 0.0
        self.epoch = 0
    def on_epoch_end(self, model):
        self.epoch += 1
        curr_loss = model.get_latest_training_loss()
        epoch_loss = curr_loss - self.prev_loss
        self.prev_loss = curr_loss
        print(f"Epoch {self.epoch} — loss: {epoch_loss:.2f}")

def train_cbow_from_bilstm_vocab(
    json_path,
    save_path,
    dim=None,
    window=5,
    min_count=1,
    workers=4,
    epochs=10,
):
    sentences = build_sentences(json_path)

    if dim is None:
        dim = ckpt["config"]["emb_dim"]

    model = Word2Vec(
        vector_size=dim,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=0,   # CBOW
        compute_loss=True,
    )
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
        compute_loss=True,
        callbacks=[LossLogger()],
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Saved model → {save_path}")

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CBOW with MLM LSTM vocab and transfer embeddings")
    parser.add_argument("--json_path", type=str, default="data/contrastive_anchors_positives.json", help="Path to contrastive_anchors_positives.json")
    parser.add_argument("--save_path", type=str, default="models/word2vec_cbow_mlmvocab.model", help="Where to save trained Word2Vec CBOW model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--emb_dim", type=int, default=None)
    args = parser.parse_args()

    train_cbow_from_bilstm_vocab(
        json_path=args.json_path,
        save_path=args.save_path,
        dim=args.emb_dim,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
    )