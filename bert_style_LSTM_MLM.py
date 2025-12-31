# This is for just a base model trained with Self-supervised Learning

import json
import re
import random
import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

PAD_ID = 0
MASK_ID = 1
UNK_ID = 2
CLS_ID = 3
SEP_ID = 4

SPECIAL_TOKENS = [PAD_TOKEN, MASK_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]
TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

class Vocab:
    def __init__(self, counter, max_size=None, min_freq=1):
        items = [t for t, c in counter.items() if c >= min_freq]
        items.sort(key=lambda t: (-counter[t], t))
        if max_size is not None:
            items = items[: max_size - len(SPECIAL_TOKENS)]
        self.itos = SPECIAL_TOKENS + items
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def __len__(self):
        return len(self.itos)
    def encode(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens]
    def decode(self, ids):
        return [self.itos[i] if i < len(self.itos) else UNK_TOKEN for i in ids]
    
def create_pretrained_embedding_matrix(
    lstm_vocab_ckpt_path,
    cbow_model_path,
    emb_dim=None,
    print_missing=True
):
    """
    Loads vocab from LSTM checkpoint file path (lstm_vocab_ckpt_path),
    returns a torch tensor embedding matrix sized (vocab_size, emb_dim),
    using CBOW vectors for matches and random for missing.
    Prints missing tokens if print_missing=True.
    """
    import torch
    import numpy as np
    from gensim.models import Word2Vec

    ckpt = torch.load(lstm_vocab_ckpt_path, map_location="cpu")
    lstm_vocab = ckpt["vocab"]

    w2v_model = Word2Vec.load(cbow_model_path)
    cbow_vocab = set(w2v_model.wv.index_to_key)
    if emb_dim is None:
        emb_dim = w2v_model.wv.vector_size

    embedding_matrix = np.zeros((len(lstm_vocab), emb_dim), dtype=np.float32)
    missing_tokens = []
    for idx, token in enumerate(lstm_vocab):
        if token in cbow_vocab:
            embedding_matrix[idx] = w2v_model.wv[token]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.02, size=emb_dim)
            if print_missing:
                missing_tokens.append(token)
    if print_missing and missing_tokens:
        print("Tokens not in CBOW Vocab (randomly initialized):")
        print(missing_tokens)
    return torch.tensor(embedding_matrix)

class ContrastiveMLMDataset(Dataset):
    def __init__(self, json_path, vocab=None, max_length=256, build_vocab=False, vocab_size=30000):
        self.json_path = Path(json_path)
        self.max_length = max_length
        self.samples = []
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for record in data:
            anchor = record.get('anchor')
            if anchor:
                self.samples.append(anchor)
            positives = record.get('positives') or []
            for p in positives:
                self.samples.append(p)
        if build_vocab:
            counter = Counter()
            for s in self.samples:
                toks = tokenize(s)[: self.max_length - 2]
                counter.update(toks)
            self.vocab = Vocab(counter, max_size=vocab_size)
        else:
            self.vocab = vocab
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = tokenize(text)[: self.max_length - 2]
        tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
        ids = self.vocab.encode(tokens)
        return torch.tensor(ids, dtype=torch.long)

def mask_tokens(inputs, vocab, mask_prob=0.15):
    """
    BERT-style MLM masking:
    - 15% of tokens selected
    - 80% -> [MASK]
    - 10% -> random token (non-special)
    - 10% -> original token
    """
    labels = inputs.clone()

    # Do not mask special tokens
    special_ids = {PAD_ID, CLS_ID, SEP_ID}
    probability_matrix = torch.full(labels.shape, mask_prob, device=inputs.device)

    for sid in special_ids:
        probability_matrix.masked_fill_(inputs == sid, 0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Only compute loss on masked tokens
    labels[~masked_indices] = -100

    masked_inputs = inputs.clone()

    # 80% -> [MASK]
    replace_prob = torch.full(labels.shape, 0.8, device=inputs.device)
    indices_replaced = torch.bernoulli(replace_prob).bool() & masked_indices
    masked_inputs[indices_replaced] = MASK_ID

    # 10% -> random token (non-special)
    random_prob = torch.full(labels.shape, 0.5, device=inputs.device)
    indices_random = (
        torch.bernoulli(random_prob).bool()
        & masked_indices
        & ~indices_replaced
    )

    if indices_random.any():
        # exclude special tokens from random sampling
        valid_ids = torch.arange(len(vocab.itos), device=inputs.device)
        valid_ids = valid_ids[
            ~torch.isin(valid_ids, torch.tensor(list(special_ids), device=inputs.device))
        ]
        random_tokens = valid_ids[
            torch.randint(0, len(valid_ids), size=inputs.shape, device=inputs.device)
        ]
        masked_inputs[indices_random] = random_tokens[indices_random]

    # remaining 10% keep original token
    return masked_inputs, labels

def masked_token_accuracy(logits, labels):
    """
    logits: (B, T, V)
    labels: (B, T) with -100 for unmasked
    """
    with torch.no_grad():
        mask = labels != -100
        if mask.sum() == 0:
            return 0.0

        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct / total

def perplexity_from_loss(loss):
    return torch.exp(torch.tensor(loss)).item()

def collate_fn(batch, pad_id=PAD_ID, max_length=None, vocab=None):
    lengths = [b.size(0) for b in batch]
    if max_length is None:
        max_len = max(lengths)
    else:
        max_len = min(max(lengths), max_length)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        l = min(b.size(0), max_len)
        padded[i, :l] = b[:l]
        attention_mask[i, :l] = 1
    masked_inputs, labels = mask_tokens(padded, vocab)
    return {'input_ids': masked_inputs, 'labels': labels, 'attention_mask': attention_mask, 'orig_input_ids': padded}

class FlexibleLSTMBase(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, bidirectional=True, pad_id=PAD_ID):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.out_dim = hidden_dim * self.num_directions
        self.proj = nn.Linear(self.out_dim, emb_dim)
    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        seq_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return seq_out, (h_n, c_n)
    def encode(self, input_ids, attention_mask, mode='pooled'):
        seq_out, (h_n, c_n) = self.forward(input_ids, attention_mask)
        if mode == 'sequence':
            return seq_out
        if mode == 'last':
            if self.num_directions == 2:
                h_n_combined = torch.cat([h_n.view(self.num_layers, 2, h_n.size(1), -1)[:,0], h_n.view(self.num_layers, 2, h_n.size(1), -1)[:,1]], dim=-1)
                c_n_combined = torch.cat([c_n.view(self.num_layers, 2, c_n.size(1), -1)[:,0], c_n.view(self.num_layers, 2, c_n.size(1), -1)[:,1]], dim=-1)
            else:
                h_n_combined, c_n_combined = h_n, c_n
            return F.normalize(self.proj(h_n_combined[-1]), dim=1), c_n_combined[-1]
        if mode == 'pooled':
            mask = attention_mask.unsqueeze(-1)
            pooled = (seq_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return F.normalize(self.proj(pooled), dim=1)
        raise ValueError(f'Unknown mode: {mode}')

class FlexibleLSTMMLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_layers=2, bidirectional=True, pad_id=PAD_ID):
        super().__init__()
        self.encoder = FlexibleLSTMBase(vocab_size, emb_dim, hidden_dim, num_layers, bidirectional, pad_id)
        self.mlm_head = nn.Linear(self.encoder.out_dim, vocab_size)
    def forward(self, input_ids, attention_mask):
        seq_out, _ = self.encoder.forward(input_ids, attention_mask)
        logits = self.mlm_head(seq_out)
        return logits

def train(json_path, model_save_path, epochs=3, batch_size=32, max_length=256, emb_dim=256, hidden_dim=256, num_layers=2, bidirectional=True, vocab_size=30000, lr=5e-4, device=None, pretrained_embedding=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds_build = ContrastiveMLMDataset(json_path, build_vocab=True, max_length=max_length, vocab_size=vocab_size)
    vocab = ds_build.vocab
    print(f"Built vocab size: {len(vocab)}")
    ds = ContrastiveMLMDataset(json_path, vocab=vocab, max_length=max_length)
    collate = lambda batch: collate_fn(batch, pad_id=PAD_ID, max_length=max_length, vocab=vocab)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = FlexibleLSTMMLM(len(vocab), emb_dim=emb_dim, hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=bidirectional).to(device)
    if pretrained_embedding is not None:
        with torch.no_grad():
            model.encoder.emb.weight.copy_(pretrained_embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        epoch_step = 0
        for batch in dl:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), labels.view(B * T))
            
            acc = masked_token_accuracy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
            epoch_step += 1
            if epoch_step % 100 == 0:
                avg_loss = total_loss / epoch_step
                avg_acc = total_acc / epoch_step
                ppl = perplexity_from_loss(avg_loss)
            
                print(
                    f"Epoch {epoch} step {epoch_step} "
                    f"loss={avg_loss:.4f} ppl={ppl:.2f} acc={avg_acc:.4f}"
                )

        epoch_loss = total_loss / max(1, epoch_step)
        epoch_acc = total_acc / max(1, epoch_step)
        epoch_ppl = perplexity_from_loss(epoch_loss)
        
        print(
            f"=== Epoch {epoch} finished | "
            f"loss={epoch_loss:.4f} "
            f"ppl={epoch_ppl:.2f} "
            f"masked_acc={epoch_acc:.4f} ==="
        )

        ckpt = {
                "model_state": model.state_dict(),
                "vocab": vocab.itos,
                "config": {
                    "emb_dim": emb_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "bidirectional": bidirectional,
                    "max_length": max_length,
                    "vocab_size": vocab_size,
                    }
                }           
        torch.save(ckpt,Path(model_save_path) / f"bilstm_mlm_epoch{epoch}.pt")
    print('Training complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/contrastive_anchors_positives.json')
    parser.add_argument('--out', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--emb-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--vocab-size', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--pretrained_embedding_path', type=str, default="models/word2vec_cbow_mlmvocab.model")  # << add this line
    parser.add_argument('--lstm_vocab_ckpt_path', type=str, default="models/mlm_bilstm/bilstm_mlm_epoch3.pt")  # << add this
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- Insert this block before train() ---------
    pretrained_embedding = None
    if args.pretrained_embedding_path and args.lstm_vocab_ckpt_path:
        pretrained_embedding = create_pretrained_embedding_matrix(
            lstm_vocab_ckpt_path=args.lstm_vocab_ckpt_path,
            cbow_model_path=args.pretrained_embedding_path
        )
        print(f"Loaded pretrained embedding from {args.pretrained_embedding_path}")

    # Modify train function to accept pretrained_embedding and feed to FlexibleLSTMMLM
    train(
        json_path=args.data,
        model_save_path=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        vocab_size=args.vocab_size,
        lr=args.lr,
        device=None,
        pretrained_embedding=pretrained_embedding  # << add this
    )

# Experimemt 1: Self-supervised from scratch

# Built vocab size: 30000
# Epoch 1 step 100 loss=7.1775 ppl=1309.60 acc=0.0622
# Epoch 1 step 200 loss=6.9224 ppl=1014.72 acc=0.0654
# Epoch 1 step 300 loss=6.7557 ppl=858.95 acc=0.0815
# Epoch 1 step 400 loss=6.5869 ppl=725.56 acc=0.0979
# Epoch 1 step 500 loss=6.4304 ppl=620.42 acc=0.1130
# Epoch 1 step 600 loss=6.2886 ppl=538.42 acc=0.1273
# Epoch 1 step 700 loss=6.1707 ppl=478.54 acc=0.1388
# Epoch 1 step 800 loss=6.0655 ppl=430.74 acc=0.1493
# Epoch 1 step 900 loss=5.9719 ppl=392.24 acc=0.1583
# Epoch 1 step 1000 loss=5.8885 ppl=360.88 acc=0.1662
# Epoch 1 step 1100 loss=5.8132 ppl=334.69 acc=0.1735
# Epoch 1 step 1200 loss=5.7459 ppl=312.90 acc=0.1799
# Epoch 1 step 1300 loss=5.6847 ppl=294.33 acc=0.1858
# Epoch 1 step 1400 loss=5.6285 ppl=278.24 acc=0.1911
# Epoch 1 step 1500 loss=5.5758 ppl=263.95 acc=0.1963
# Epoch 1 step 1600 loss=5.5283 ppl=251.71 acc=0.2009
# Epoch 1 step 1700 loss=5.4811 ppl=240.11 acc=0.2053
# Epoch 1 step 1800 loss=5.4389 ppl=230.18 acc=0.2093
# Epoch 1 step 1900 loss=5.3989 ppl=221.16 acc=0.2132
# Epoch 1 step 2000 loss=5.3628 ppl=213.31 acc=0.2166
# Epoch 1 step 2100 loss=5.3274 ppl=205.90 acc=0.2200
# Epoch 1 step 2200 loss=5.2937 ppl=199.07 acc=0.2232
# Epoch 1 step 2300 loss=5.2618 ppl=192.83 acc=0.2262
# Epoch 1 step 2400 loss=5.2312 ppl=187.01 acc=0.2292
# Epoch 1 step 2500 loss=5.2025 ppl=181.72 acc=0.2320
# Epoch 1 step 2600 loss=5.1741 ppl=176.63 acc=0.2347
# Epoch 1 step 2700 loss=5.1490 ppl=172.25 acc=0.2371
# Epoch 1 step 2800 loss=5.1237 ppl=167.96 acc=0.2396
# Epoch 1 step 2900 loss=5.0975 ppl=163.61 acc=0.2422
# Epoch 1 step 3000 loss=5.0728 ppl=159.62 acc=0.2447
# Epoch 1 step 3100 loss=5.0501 ppl=156.04 acc=0.2469
# Epoch 1 step 3200 loss=5.0288 ppl=152.75 acc=0.2490
# Epoch 1 step 3300 loss=5.0072 ppl=149.49 acc=0.2512
# Epoch 1 step 3400 loss=4.9864 ppl=146.41 acc=0.2532
# Epoch 1 step 3500 loss=4.9659 ppl=143.44 acc=0.2553
# Epoch 1 step 3600 loss=4.9464 ppl=140.67 acc=0.2572
# Epoch 1 step 3700 loss=4.9276 ppl=138.05 acc=0.2591
# Epoch 1 step 3800 loss=4.9092 ppl=135.53 acc=0.2609
# Epoch 1 step 3900 loss=4.8904 ppl=133.01 acc=0.2628
# Epoch 1 step 4000 loss=4.8727 ppl=130.67 acc=0.2646
# Epoch 1 step 4100 loss=4.8559 ppl=128.50 acc=0.2663
# Epoch 1 step 4200 loss=4.8404 ppl=126.52 acc=0.2679
# Epoch 1 step 4300 loss=4.8248 ppl=124.57 acc=0.2695
# Epoch 1 step 4400 loss=4.8088 ppl=122.58 acc=0.2711
# Epoch 1 step 4500 loss=4.7941 ppl=120.80 acc=0.2725
# Epoch 1 step 4600 loss=4.7795 ppl=119.05 acc=0.2740
# Epoch 1 step 4700 loss=4.7653 ppl=117.36 acc=0.2755
# Epoch 1 step 4800 loss=4.7512 ppl=115.72 acc=0.2769
# Epoch 1 step 4900 loss=4.7369 ppl=114.08 acc=0.2783
# Epoch 1 step 5000 loss=4.7240 ppl=112.61 acc=0.2796
# Epoch 1 step 5100 loss=4.7110 ppl=111.16 acc=0.2809
# Epoch 1 step 5200 loss=4.6972 ppl=109.64 acc=0.2823
# Epoch 1 step 5300 loss=4.6842 ppl=108.23 acc=0.2836
# Epoch 1 step 5400 loss=4.6722 ppl=106.93 acc=0.2848
# Epoch 1 step 5500 loss=4.6604 ppl=105.67 acc=0.2860
# Epoch 1 step 5600 loss=4.6485 ppl=104.43 acc=0.2872
# Epoch 1 step 5700 loss=4.6370 ppl=103.24 acc=0.2883
# Epoch 1 step 5800 loss=4.6259 ppl=102.10 acc=0.2895
# Epoch 1 step 5900 loss=4.6150 ppl=100.98 acc=0.2906
# === Epoch 1 finished | loss=4.6106 ppl=100.55 masked_acc=0.2911 ===
# Epoch 2 step 100 loss=3.9367 ppl=51.25 acc=0.3574
# Epoch 2 step 200 loss=3.9300 ppl=50.91 acc=0.3595
# Epoch 2 step 300 loss=3.9285 ppl=50.83 acc=0.3599
# Epoch 2 step 400 loss=3.9193 ppl=50.36 acc=0.3607
# Epoch 2 step 500 loss=3.9163 ppl=50.22 acc=0.3609
# Epoch 2 step 600 loss=3.9130 ppl=50.05 acc=0.3614
# Epoch 2 step 700 loss=3.9095 ppl=49.87 acc=0.3616
# Epoch 2 step 800 loss=3.9076 ppl=49.78 acc=0.3620
# Epoch 2 step 900 loss=3.8995 ppl=49.38 acc=0.3628
# Epoch 2 step 1000 loss=3.8966 ppl=49.23 acc=0.3632
# Epoch 2 step 1100 loss=3.8939 ppl=49.10 acc=0.3635
# Epoch 2 step 1200 loss=3.8920 ppl=49.01 acc=0.3637
# Epoch 2 step 1300 loss=3.8882 ppl=48.82 acc=0.3641
# Epoch 2 step 1400 loss=3.8854 ppl=48.68 acc=0.3643
# Epoch 2 step 1500 loss=3.8817 ppl=48.51 acc=0.3646
# Epoch 2 step 1600 loss=3.8789 ppl=48.37 acc=0.3649
# Epoch 2 step 1700 loss=3.8753 ppl=48.20 acc=0.3653
# Epoch 2 step 1800 loss=3.8717 ppl=48.02 acc=0.3656
# Epoch 2 step 1900 loss=3.8663 ppl=47.77 acc=0.3662
# Epoch 2 step 2000 loss=3.8624 ppl=47.58 acc=0.3667
# Epoch 2 step 2100 loss=3.8584 ppl=47.39 acc=0.3671
# Epoch 2 step 2200 loss=3.8539 ppl=47.18 acc=0.3676
# Epoch 2 step 2300 loss=3.8517 ppl=47.07 acc=0.3678
# Epoch 2 step 2400 loss=3.8489 ppl=46.94 acc=0.3680
# Epoch 2 step 2500 loss=3.8473 ppl=46.87 acc=0.3681
# Epoch 2 step 2600 loss=3.8434 ppl=46.69 acc=0.3684
# Epoch 2 step 2700 loss=3.8398 ppl=46.52 acc=0.3688
# Epoch 2 step 2800 loss=3.8366 ppl=46.37 acc=0.3692
# Epoch 2 step 2900 loss=3.8330 ppl=46.20 acc=0.3695
# Epoch 2 step 3000 loss=3.8297 ppl=46.05 acc=0.3698
# Epoch 2 step 3100 loss=3.8262 ppl=45.89 acc=0.3702
# Epoch 2 step 3200 loss=3.8231 ppl=45.74 acc=0.3706
# Epoch 2 step 3300 loss=3.8194 ppl=45.58 acc=0.3710
# Epoch 2 step 3400 loss=3.8161 ppl=45.43 acc=0.3713
# Epoch 2 step 3500 loss=3.8126 ppl=45.27 acc=0.3717
# Epoch 2 step 3600 loss=3.8097 ppl=45.13 acc=0.3720
# Epoch 2 step 3700 loss=3.8063 ppl=44.98 acc=0.3724
# Epoch 2 step 3800 loss=3.8030 ppl=44.83 acc=0.3727
# Epoch 2 step 3900 loss=3.7996 ppl=44.68 acc=0.3731
# Epoch 2 step 4000 loss=3.7959 ppl=44.52 acc=0.3734
# Epoch 2 step 4100 loss=3.7934 ppl=44.41 acc=0.3737
# Epoch 2 step 4200 loss=3.7902 ppl=44.26 acc=0.3740
# Epoch 2 step 4300 loss=3.7870 ppl=44.12 acc=0.3744
# Epoch 2 step 4400 loss=3.7834 ppl=43.96 acc=0.3747
# Epoch 2 step 4500 loss=3.7793 ppl=43.78 acc=0.3751
# Epoch 2 step 4600 loss=3.7770 ppl=43.68 acc=0.3753
# Epoch 2 step 4700 loss=3.7737 ppl=43.54 acc=0.3757
# Epoch 2 step 4800 loss=3.7713 ppl=43.44 acc=0.3759
# Epoch 2 step 4900 loss=3.7679 ppl=43.29 acc=0.3762
# Epoch 2 step 5000 loss=3.7649 ppl=43.16 acc=0.3765
# Epoch 2 step 5100 loss=3.7624 ppl=43.05 acc=0.3767
# Epoch 2 step 5200 loss=3.7596 ppl=42.93 acc=0.3770
# Epoch 2 step 5300 loss=3.7566 ppl=42.80 acc=0.3773
# Epoch 2 step 5400 loss=3.7538 ppl=42.68 acc=0.3776
# Epoch 2 step 5500 loss=3.7519 ppl=42.60 acc=0.3778
# Epoch 2 step 5600 loss=3.7493 ppl=42.49 acc=0.3781
# Epoch 2 step 5700 loss=3.7463 ppl=42.36 acc=0.3784
# Epoch 2 step 5800 loss=3.7433 ppl=42.24 acc=0.3787
# Epoch 2 step 5900 loss=3.7409 ppl=42.14 acc=0.3789
# === Epoch 2 finished | loss=3.7401 ppl=42.10 masked_acc=0.3790 ===
# Epoch 3 step 100 loss=3.5552 ppl=34.99 acc=0.3970
# Epoch 3 step 200 loss=3.5520 ppl=34.88 acc=0.3971
# Epoch 3 step 300 loss=3.5528 ppl=34.91 acc=0.3973
# Epoch 3 step 400 loss=3.5580 ppl=35.09 acc=0.3969
# Epoch 3 step 500 loss=3.5550 ppl=34.99 acc=0.3971
# Epoch 3 step 600 loss=3.5525 ppl=34.90 acc=0.3977
# Epoch 3 step 700 loss=3.5504 ppl=34.83 acc=0.3976
# Epoch 3 step 800 loss=3.5475 ppl=34.73 acc=0.3983
# Epoch 3 step 900 loss=3.5469 ppl=34.71 acc=0.3984
# Epoch 3 step 1000 loss=3.5467 ppl=34.70 acc=0.3983
# Epoch 3 step 1100 loss=3.5451 ppl=34.64 acc=0.3985
# Epoch 3 step 1200 loss=3.5411 ppl=34.50 acc=0.3989
# Epoch 3 step 1300 loss=3.5431 ppl=34.57 acc=0.3986
# Epoch 3 step 1400 loss=3.5423 ppl=34.55 acc=0.3986
# Epoch 3 step 1500 loss=3.5398 ppl=34.46 acc=0.3989
# Epoch 3 step 1600 loss=3.5386 ppl=34.42 acc=0.3990
# Epoch 3 step 1700 loss=3.5369 ppl=34.36 acc=0.3992
# Epoch 3 step 1800 loss=3.5343 ppl=34.27 acc=0.3997
# Epoch 3 step 1900 loss=3.5333 ppl=34.24 acc=0.3999
# Epoch 3 step 2000 loss=3.5320 ppl=34.19 acc=0.3999
# Epoch 3 step 2100 loss=3.5316 ppl=34.18 acc=0.4000
# Epoch 3 step 2200 loss=3.5287 ppl=34.08 acc=0.4002
# Epoch 3 step 2300 loss=3.5278 ppl=34.05 acc=0.4003
# Epoch 3 step 2400 loss=3.5253 ppl=33.96 acc=0.4005
# Epoch 3 step 2500 loss=3.5232 ppl=33.89 acc=0.4007
# Epoch 3 step 2600 loss=3.5225 ppl=33.87 acc=0.4008
# Epoch 3 step 2700 loss=3.5206 ppl=33.80 acc=0.4010
# Epoch 3 step 2800 loss=3.5194 ppl=33.76 acc=0.4011
# Epoch 3 step 2900 loss=3.5178 ppl=33.71 acc=0.4013
# Epoch 3 step 3000 loss=3.5166 ppl=33.67 acc=0.4014
# Epoch 3 step 3100 loss=3.5142 ppl=33.59 acc=0.4017
# Epoch 3 step 3200 loss=3.5121 ppl=33.52 acc=0.4019
# Epoch 3 step 3300 loss=3.5097 ppl=33.44 acc=0.4022
# Epoch 3 step 3400 loss=3.5080 ppl=33.38 acc=0.4024
# Epoch 3 step 3500 loss=3.5067 ppl=33.34 acc=0.4025
# Epoch 3 step 3600 loss=3.5046 ppl=33.27 acc=0.4028
# Epoch 3 step 3700 loss=3.5035 ppl=33.23 acc=0.4029
# Epoch 3 step 3800 loss=3.5005 ppl=33.13 acc=0.4033
# Epoch 3 step 3900 loss=3.4988 ppl=33.08 acc=0.4035
# Epoch 3 step 4000 loss=3.4965 ppl=33.00 acc=0.4038
# Epoch 3 step 4100 loss=3.4938 ppl=32.91 acc=0.4041
# Epoch 3 step 4200 loss=3.4921 ppl=32.86 acc=0.4042
# Epoch 3 step 4300 loss=3.4908 ppl=32.81 acc=0.4043
# Epoch 3 step 4400 loss=3.4889 ppl=32.75 acc=0.4046
# Epoch 3 step 4500 loss=3.4865 ppl=32.67 acc=0.4049
# Epoch 3 step 4600 loss=3.4851 ppl=32.63 acc=0.4050
# Epoch 3 step 4700 loss=3.4834 ppl=32.57 acc=0.4052
# Epoch 3 step 4800 loss=3.4819 ppl=32.52 acc=0.4054
# Epoch 3 step 4900 loss=3.4803 ppl=32.47 acc=0.4055
# Epoch 3 step 5000 loss=3.4788 ppl=32.42 acc=0.4058
# Epoch 3 step 5100 loss=3.4776 ppl=32.38 acc=0.4059
# Epoch 3 step 5200 loss=3.4758 ppl=32.32 acc=0.4061
# Epoch 3 step 5300 loss=3.4746 ppl=32.29 acc=0.4062
# Epoch 3 step 5400 loss=3.4732 ppl=32.24 acc=0.4064
# Epoch 3 step 5500 loss=3.4721 ppl=32.20 acc=0.4065
# Epoch 3 step 5600 loss=3.4709 ppl=32.17 acc=0.4066
# Epoch 3 step 5700 loss=3.4695 ppl=32.12 acc=0.4068
# Epoch 3 step 5800 loss=3.4678 ppl=32.07 acc=0.4069
# Epoch 3 step 5900 loss=3.4664 ppl=32.02 acc=0.4071
# === Epoch 3 finished | loss=3.4656 ppl=32.00 masked_acc=0.4072 ===



# Experimemt: 2 (cbow embedding transfered)

# Tokens not in CBOW Vocab (randomly initialized):
# ['[PAD]', '[MASK]']
# Loaded pretrained embedding from /kaggle/input/base-bilstm-transfer-embed/word2vec_cbow_mlmvocab.model
# Built vocab size: 30000
# Epoch 1 step 100 loss=7.1853 ppl=1319.82 acc=0.0732
# Epoch 1 step 200 loss=6.6514 ppl=773.86 acc=0.1081
# Epoch 1 step 300 loss=6.3021 ppl=545.72 acc=0.1392
# Epoch 1 step 400 loss=6.0546 ppl=426.06 acc=0.1620
# Epoch 1 step 500 loss=5.8658 ppl=352.77 acc=0.1789
# Epoch 1 step 600 loss=5.7214 ppl=305.34 acc=0.1924
# Epoch 1 step 700 loss=5.6008 ppl=270.64 acc=0.2037
# Epoch 1 step 800 loss=5.4953 ppl=243.55 acc=0.2136
# Epoch 1 step 900 loss=5.4040 ppl=222.29 acc=0.2218
# Epoch 1 step 1000 loss=5.3229 ppl=204.99 acc=0.2294
# Epoch 1 step 1100 loss=5.2524 ppl=191.03 acc=0.2361
# Epoch 1 step 1200 loss=5.1903 ppl=179.52 acc=0.2420
# Epoch 1 step 1300 loss=5.1340 ppl=169.69 acc=0.2474
# Epoch 1 step 1400 loss=5.0786 ppl=160.55 acc=0.2525
# Epoch 1 step 1500 loss=5.0310 ppl=153.09 acc=0.2569
# Epoch 1 step 1600 loss=4.9845 ppl=146.13 acc=0.2613
# Epoch 1 step 1700 loss=4.9420 ppl=140.05 acc=0.2654
# Epoch 1 step 1800 loss=4.9029 ppl=134.68 acc=0.2691
# Epoch 1 step 1900 loss=4.8659 ppl=129.79 acc=0.2726
# Epoch 1 step 2000 loss=4.8304 ppl=125.26 acc=0.2760
# Epoch 1 step 2100 loss=4.7971 ppl=121.16 acc=0.2792
# Epoch 1 step 2200 loss=4.7666 ppl=117.52 acc=0.2821
# Epoch 1 step 2300 loss=4.7371 ppl=114.10 acc=0.2849
# Epoch 1 step 2400 loss=4.7079 ppl=110.82 acc=0.2876
# Epoch 1 step 2500 loss=4.6793 ppl=107.69 acc=0.2903
# Epoch 1 step 2600 loss=4.6536 ppl=104.97 acc=0.2929
# Epoch 1 step 2700 loss=4.6283 ppl=102.34 acc=0.2953
# Epoch 1 step 2800 loss=4.6058 ppl=100.06 acc=0.2975
# Epoch 1 step 2900 loss=4.5835 ppl=97.86 acc=0.2997
# Epoch 1 step 3000 loss=4.5612 ppl=95.70 acc=0.3018
# Epoch 1 step 3100 loss=4.5398 ppl=93.68 acc=0.3038
# Epoch 1 step 3200 loss=4.5202 ppl=91.85 acc=0.3056
# Epoch 1 step 3300 loss=4.5013 ppl=90.14 acc=0.3073
# Epoch 1 step 3400 loss=4.4826 ppl=88.46 acc=0.3091
# Epoch 1 step 3500 loss=4.4648 ppl=86.91 acc=0.3108
# Epoch 1 step 3600 loss=4.4476 ppl=85.42 acc=0.3124
# Epoch 1 step 3700 loss=4.4306 ppl=83.98 acc=0.3140
# Epoch 1 step 3800 loss=4.4130 ppl=82.51 acc=0.3158
# Epoch 1 step 3900 loss=4.3968 ppl=81.19 acc=0.3173
# Epoch 1 step 4000 loss=4.3814 ppl=79.95 acc=0.3188
# Epoch 1 step 4100 loss=4.3669 ppl=78.80 acc=0.3201
# Epoch 1 step 4200 loss=4.3521 ppl=77.64 acc=0.3216
# Epoch 1 step 4300 loss=4.3376 ppl=76.53 acc=0.3230
# Epoch 1 step 4400 loss=4.3244 ppl=75.52 acc=0.3242
# Epoch 1 step 4500 loss=4.3110 ppl=74.52 acc=0.3254
# Epoch 1 step 4600 loss=4.2980 ppl=73.56 acc=0.3267
# Epoch 1 step 4700 loss=4.2850 ppl=72.60 acc=0.3279
# Epoch 1 step 4800 loss=4.2723 ppl=71.69 acc=0.3291
# Epoch 1 step 4900 loss=4.2602 ppl=70.82 acc=0.3303
# Epoch 1 step 5000 loss=4.2482 ppl=69.98 acc=0.3315
# Epoch 1 step 5100 loss=4.2371 ppl=69.21 acc=0.3325
# Epoch 1 step 5200 loss=4.2253 ppl=68.40 acc=0.3337
# Epoch 1 step 5300 loss=4.2146 ppl=67.66 acc=0.3347
# Epoch 1 step 5400 loss=4.2037 ppl=66.93 acc=0.3358
# Epoch 1 step 5500 loss=4.1927 ppl=66.20 acc=0.3368
# Epoch 1 step 5600 loss=4.1825 ppl=65.53 acc=0.3378
# Epoch 1 step 5700 loss=4.1722 ppl=64.86 acc=0.3388
# Epoch 1 step 5800 loss=4.1621 ppl=64.21 acc=0.3398
# Epoch 1 step 5900 loss=4.1526 ppl=63.60 acc=0.3407
# === Epoch 1 finished | loss=4.1486 ppl=63.35 masked_acc=0.3411 ===
# Epoch 2 step 100 loss=3.5954 ppl=36.43 acc=0.3930
# Epoch 2 step 200 loss=3.5875 ppl=36.15 acc=0.3938
# Epoch 2 step 300 loss=3.5742 ppl=35.66 acc=0.3952
# Epoch 2 step 400 loss=3.5662 ppl=35.38 acc=0.3958
# Epoch 2 step 500 loss=3.5676 ppl=35.43 acc=0.3956
# Epoch 2 step 600 loss=3.5646 ppl=35.33 acc=0.3959
# Epoch 2 step 700 loss=3.5598 ppl=35.16 acc=0.3966
# Epoch 2 step 800 loss=3.5577 ppl=35.08 acc=0.3968
# Epoch 2 step 900 loss=3.5568 ppl=35.05 acc=0.3971
# Epoch 2 step 1000 loss=3.5525 ppl=34.90 acc=0.3975
# Epoch 2 step 1100 loss=3.5462 ppl=34.68 acc=0.3982
# Epoch 2 step 1200 loss=3.5416 ppl=34.52 acc=0.3986
# Epoch 2 step 1300 loss=3.5375 ppl=34.38 acc=0.3990
# Epoch 2 step 1400 loss=3.5333 ppl=34.24 acc=0.3994
# Epoch 2 step 1500 loss=3.5299 ppl=34.12 acc=0.3998
# Epoch 2 step 1600 loss=3.5274 ppl=34.03 acc=0.4000
# Epoch 2 step 1700 loss=3.5232 ppl=33.89 acc=0.4003
# Epoch 2 step 1800 loss=3.5210 ppl=33.82 acc=0.4004
# Epoch 2 step 1900 loss=3.5170 ppl=33.68 acc=0.4008
# Epoch 2 step 2000 loss=3.5142 ppl=33.59 acc=0.4011
# Epoch 2 step 2100 loss=3.5100 ppl=33.45 acc=0.4016
# Epoch 2 step 2200 loss=3.5076 ppl=33.37 acc=0.4018
# Epoch 2 step 2300 loss=3.5057 ppl=33.30 acc=0.4021
# Epoch 2 step 2400 loss=3.5031 ppl=33.22 acc=0.4023
# Epoch 2 step 2500 loss=3.4995 ppl=33.10 acc=0.4027
# Epoch 2 step 2600 loss=3.4964 ppl=33.00 acc=0.4030
# Epoch 2 step 2700 loss=3.4929 ppl=32.88 acc=0.4033
# Epoch 2 step 2800 loss=3.4903 ppl=32.80 acc=0.4036
# Epoch 2 step 2900 loss=3.4864 ppl=32.67 acc=0.4040
# Epoch 2 step 3000 loss=3.4836 ppl=32.58 acc=0.4043
# Epoch 2 step 3100 loss=3.4806 ppl=32.48 acc=0.4046
# Epoch 2 step 3200 loss=3.4774 ppl=32.38 acc=0.4050
# Epoch 2 step 3300 loss=3.4757 ppl=32.32 acc=0.4051
# Epoch 2 step 3400 loss=3.4730 ppl=32.23 acc=0.4055
# Epoch 2 step 3500 loss=3.4703 ppl=32.15 acc=0.4057
# Epoch 2 step 3600 loss=3.4675 ppl=32.06 acc=0.4059
# Epoch 2 step 3700 loss=3.4653 ppl=31.99 acc=0.4061
# Epoch 2 step 3800 loss=3.4626 ppl=31.90 acc=0.4064
# Epoch 2 step 3900 loss=3.4599 ppl=31.81 acc=0.4067
# Epoch 2 step 4000 loss=3.4573 ppl=31.73 acc=0.4070
# Epoch 2 step 4100 loss=3.4548 ppl=31.65 acc=0.4073
# Epoch 2 step 4200 loss=3.4521 ppl=31.57 acc=0.4075
# Epoch 2 step 4300 loss=3.4496 ppl=31.49 acc=0.4078
# Epoch 2 step 4400 loss=3.4470 ppl=31.41 acc=0.4080
# Epoch 2 step 4500 loss=3.4447 ppl=31.34 acc=0.4083
# Epoch 2 step 4600 loss=3.4423 ppl=31.26 acc=0.4085
# Epoch 2 step 4700 loss=3.4394 ppl=31.17 acc=0.4088
# Epoch 2 step 4800 loss=3.4367 ppl=31.09 acc=0.4091
# Epoch 2 step 4900 loss=3.4345 ppl=31.02 acc=0.4093
# Epoch 2 step 5000 loss=3.4325 ppl=30.95 acc=0.4095
# Epoch 2 step 5100 loss=3.4299 ppl=30.87 acc=0.4098
# Epoch 2 step 5200 loss=3.4275 ppl=30.80 acc=0.4100
# Epoch 2 step 5300 loss=3.4249 ppl=30.72 acc=0.4103
# Epoch 2 step 5400 loss=3.4226 ppl=30.65 acc=0.4106
# Epoch 2 step 5500 loss=3.4206 ppl=30.59 acc=0.4108
# Epoch 2 step 5600 loss=3.4185 ppl=30.52 acc=0.4110
# Epoch 2 step 5700 loss=3.4164 ppl=30.46 acc=0.4112
# Epoch 2 step 5800 loss=3.4141 ppl=30.39 acc=0.4114
# Epoch 2 step 5900 loss=3.4121 ppl=30.33 acc=0.4116
# === Epoch 2 finished | loss=3.4115 ppl=30.31 masked_acc=0.4117 ===
# Epoch 3 step 100 loss=3.2699 ppl=26.31 acc=0.4258
# Epoch 3 step 200 loss=3.2840 ppl=26.68 acc=0.4229
# Epoch 3 step 300 loss=3.2766 ppl=26.49 acc=0.4243
# Epoch 3 step 400 loss=3.2737 ppl=26.41 acc=0.4251
# Epoch 3 step 500 loss=3.2750 ppl=26.44 acc=0.4249
# Epoch 3 step 600 loss=3.2734 ppl=26.40 acc=0.4252
# Epoch 3 step 700 loss=3.2681 ppl=26.26 acc=0.4255
# Epoch 3 step 800 loss=3.2662 ppl=26.21 acc=0.4258
# Epoch 3 step 900 loss=3.2619 ppl=26.10 acc=0.4265
# Epoch 3 step 1000 loss=3.2588 ppl=26.02 acc=0.4267
# Epoch 3 step 1100 loss=3.2566 ppl=25.96 acc=0.4269
# Epoch 3 step 1200 loss=3.2556 ppl=25.94 acc=0.4271
# Epoch 3 step 1300 loss=3.2541 ppl=25.90 acc=0.4272
# Epoch 3 step 1400 loss=3.2541 ppl=25.90 acc=0.4273
# Epoch 3 step 1500 loss=3.2535 ppl=25.88 acc=0.4273
# Epoch 3 step 1600 loss=3.2511 ppl=25.82 acc=0.4277
# Epoch 3 step 1700 loss=3.2487 ppl=25.76 acc=0.4280
# Epoch 3 step 1800 loss=3.2461 ppl=25.69 acc=0.4282
# Epoch 3 step 1900 loss=3.2436 ppl=25.63 acc=0.4285
# Epoch 3 step 2000 loss=3.2429 ppl=25.61 acc=0.4286
# Epoch 3 step 2100 loss=3.2428 ppl=25.61 acc=0.4286
# Epoch 3 step 2200 loss=3.2401 ppl=25.54 acc=0.4290
# Epoch 3 step 2300 loss=3.2377 ppl=25.48 acc=0.4293
# Epoch 3 step 2400 loss=3.2354 ppl=25.42 acc=0.4296
# Epoch 3 step 2500 loss=3.2351 ppl=25.41 acc=0.4296
# Epoch 3 step 2600 loss=3.2332 ppl=25.36 acc=0.4298
# Epoch 3 step 2700 loss=3.2321 ppl=25.33 acc=0.4299
# Epoch 3 step 2800 loss=3.2321 ppl=25.33 acc=0.4299
# Epoch 3 step 2900 loss=3.2315 ppl=25.32 acc=0.4299
# Epoch 3 step 3000 loss=3.2300 ppl=25.28 acc=0.4301
# Epoch 3 step 3100 loss=3.2287 ppl=25.25 acc=0.4302
# Epoch 3 step 3200 loss=3.2281 ppl=25.23 acc=0.4303
# Epoch 3 step 3300 loss=3.2261 ppl=25.18 acc=0.4305
# Epoch 3 step 3400 loss=3.2252 ppl=25.16 acc=0.4306
# Epoch 3 step 3500 loss=3.2240 ppl=25.13 acc=0.4307
# Epoch 3 step 3600 loss=3.2232 ppl=25.11 acc=0.4309
# Epoch 3 step 3700 loss=3.2216 ppl=25.07 acc=0.4311
# Epoch 3 step 3800 loss=3.2205 ppl=25.04 acc=0.4312
# Epoch 3 step 3900 loss=3.2193 ppl=25.01 acc=0.4313
# Epoch 3 step 4400 loss=3.2119 ppl=24.83 acc=0.4321
# Epoch 3 step 4500 loss=3.2105 ppl=24.79 acc=0.4322
# Epoch 3 step 4600 loss=3.2094 ppl=24.77 acc=0.4323
# Epoch 3 step 4700 loss=3.2084 ppl=24.74 acc=0.4324
# Epoch 3 step 4800 loss=3.2073 ppl=24.71 acc=0.4326
# Epoch 3 step 4900 loss=3.2061 ppl=24.68 acc=0.4327
# Epoch 3 step 5000 loss=3.2050 ppl=24.66 acc=0.4328
# Epoch 3 step 5100 loss=3.2034 ppl=24.62 acc=0.4330
# Epoch 3 step 5200 loss=3.2025 ppl=24.59 acc=0.4331
# Epoch 3 step 5300 loss=3.2014 ppl=24.57 acc=0.4332
# Epoch 3 step 5400 loss=3.2007 ppl=24.55 acc=0.4333
# Epoch 3 step 5500 loss=3.1996 ppl=24.52 acc=0.4334
# Epoch 3 step 5600 loss=3.1984 ppl=24.49 acc=0.4336
# Epoch 3 step 5700 loss=3.1973 ppl=24.47 acc=0.4337
# Epoch 3 step 5800 loss=3.1964 ppl=24.44 acc=0.4338
# Epoch 3 step 5900 loss=3.1951 ppl=24.41 acc=0.4340
# === Epoch 3 finished | loss=3.1945 ppl=24.40 masked_acc=0.4341 ===