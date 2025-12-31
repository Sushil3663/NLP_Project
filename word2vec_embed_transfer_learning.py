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
        negative=10, # changed
        window=7 # changed
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
    parser.add_argument("--save_path", type=str, default="models/word2vec_skip_mlmvocab.model", help="Where to save trained Word2Vec CBOW model")
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

# Exp: 1 (cbow)

# Epoch 1 — loss: 5878969.00
# Epoch 2 — loss: 5031063.00
# Epoch 3 — loss: 4840968.00
# Epoch 4 — loss: 4290784.00
# Epoch 5 — loss: 4122540.00
# Epoch 6 — loss: 4163064.00
# Epoch 7 — loss: 4151498.00
# Epoch 8 — loss: 3303742.00
# Epoch 9 — loss: 2986552.00
# Epoch 10 — loss: 3000704.00
# Epoch 11 — loss: 2991648.00
# Epoch 12 — loss: 3057568.00
# Epoch 13 — loss: 3009704.00
# Epoch 14 — loss: 3053768.00
# Epoch 15 — loss: 2990884.00
# Epoch 16 — loss: 3022192.00
# Epoch 17 — loss: 3032496.00
# Epoch 18 — loss: 2999260.00
# Epoch 19 — loss: 1782436.00
# Epoch 20 — loss: 971496.00
# Epoch 21 — loss: 955672.00
# Epoch 22 — loss: 965200.00
# Epoch 23 — loss: 959784.00
# Epoch 24 — loss: 950104.00
# Epoch 25 — loss: 944840.00
# Epoch 26 — loss: 948544.00
# Epoch 27 — loss: 938016.00
# Epoch 28 — loss: 930808.00
# Epoch 29 — loss: 947192.00
# Epoch 30 — loss: 933512.00
# Epoch 31 — loss: 924704.00
# Epoch 32 — loss: 929944.00
# Epoch 33 — loss: 927648.00
# Epoch 34 — loss: 921912.00
# Epoch 35 — loss: 910648.00
# Epoch 36 — loss: 905624.00
# Epoch 37 — loss: 901920.00
# Epoch 38 — loss: 902024.00
# Epoch 39 — loss: 904152.00
# Epoch 40 — loss: 884160.00
# Epoch 41 — loss: 881864.00
# Epoch 42 — loss: 876080.00
# Epoch 43 — loss: 859848.00
# Epoch 44 — loss: 859568.00
# Epoch 45 — loss: 838648.00
# Epoch 46 — loss: 845928.00
# Epoch 47 — loss: 848944.00
# Epoch 48 — loss: 833704.00
# Epoch 49 — loss: 824032.00
# Epoch 50 — loss: 817016.00
# Epoch 51 — loss: 806952.00
# Epoch 52 — loss: 797264.00
# Epoch 53 — loss: 800312.00
# Epoch 54 — loss: 795136.00
# Epoch 55 — loss: 786728.00
# Epoch 56 — loss: 786264.00
# Epoch 57 — loss: 769888.00
# Epoch 58 — loss: 763000.00
# Epoch 59 — loss: 753608.00
# Epoch 60 — loss: 750072.00
# Epoch 61 — loss: 733224.00
# Epoch 62 — loss: 733328.00
# Epoch 63 — loss: 709880.00
# Epoch 64 — loss: 705464.00
# Epoch 65 — loss: 695600.00
# Epoch 66 — loss: 692264.00
# Epoch 67 — loss: 671784.00
# Epoch 68 — loss: 670256.00
# Epoch 69 — loss: 657352.00
# Epoch 70 — loss: 645328.00
# Epoch 71 — loss: 641872.00
# Epoch 72 — loss: 635752.00
# Epoch 73 — loss: 624136.00
# Epoch 74 — loss: 610568.00
# Epoch 75 — loss: 602968.00
# Epoch 76 — loss: 585912.00
# Epoch 77 — loss: 572080.00
# Epoch 78 — loss: 565704.00
# Epoch 79 — loss: 548568.00
# Epoch 80 — loss: 537632.00
# Epoch 81 — loss: 522024.00
# Epoch 82 — loss: 511000.00
# Epoch 83 — loss: 503136.00
# Epoch 84 — loss: 480080.00
# Epoch 85 — loss: 470840.00
# Epoch 86 — loss: 455320.00
# Epoch 87 — loss: 442400.00
# Epoch 88 — loss: 431400.00
# Epoch 89 — loss: 416064.00
# Epoch 90 — loss: 397184.00
# Epoch 91 — loss: 383688.00
# Epoch 92 — loss: 364928.00
# Epoch 93 — loss: 350384.00
# Epoch 94 — loss: 331664.00
# Epoch 95 — loss: 319128.00
# Epoch 96 — loss: 300792.00
# Epoch 97 — loss: 282144.00
# Epoch 98 — loss: 265744.00
# Epoch 99 — loss: 241760.00
# Epoch 100 — loss: 225296.00
# Saved model → models/word2vec_cbow_mlmvocab.model