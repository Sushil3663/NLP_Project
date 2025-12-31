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

def train(json_path, model_save_path, epochs=3, batch_size=32, max_length=256, emb_dim=256, hidden_dim=256, num_layers=2, bidirectional=True, vocab_size=30000, lr=5e-4, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds_build = ContrastiveMLMDataset(json_path, build_vocab=True, max_length=max_length, vocab_size=vocab_size)
    vocab = ds_build.vocab
    print(f"Built vocab size: {len(vocab)}")
    ds = ContrastiveMLMDataset(json_path, vocab=vocab, max_length=max_length)
    collate = lambda batch: collate_fn(batch, pad_id=PAD_ID, max_length=max_length, vocab=vocab)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = FlexibleLSTMMLM(len(vocab), emb_dim=emb_dim, hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=bidirectional).to(device)
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
    parser.add_argument('--data', type=str, default='/kaggle/input/bookdata/contrastive_anchors_positives.json')
    parser.add_argument('--out', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--emb-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--vocab-size', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train(json_path=args.data, model_save_path=out_dir, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=args.bidirectional, vocab_size=args.vocab_size, lr=args.lr)