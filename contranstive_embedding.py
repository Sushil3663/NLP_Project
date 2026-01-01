import re
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np

# ============================================================
# TOKENS — MUST MATCH MLM (DO NOT TOUCH)
# ============================================================
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

TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(text.lower()) if text else []


# ============================================================
# ENCODING — MLM SAFE (keeps vocab & IDs unchanged)
# ============================================================
def encode(text, vocab, max_len):
    toks = tokenize(text)
    toks = [CLS_TOKEN] + toks[: max_len - 2] + [SEP_TOKEN]
    ids = [vocab.get(t, UNK_ID) for t in toks]
    return ids if len(ids) > 0 else [PAD_ID]


# ============================================================
# BiLSTM ENCODER (unchanged semantics)
# ============================================================
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, bidirectional):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(self.out_dim, emb_dim)

    def forward(self, ids, mask):
        # ids: (N, L) long, mask: (N, L) long/float
        x = self.emb(ids)
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        emb = self.proj(pooled)
        return F.normalize(emb, dim=1)


# ============================================================
# MS MARCO QUERY → POS/NEG GROUPED DATASET
# - keep queries that have >=1 positive
# - negatives may be zero (we still allow those queries)
# ============================================================
class MSMarcoGrouped(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.data = []
        self.vocab = vocab
        self.max_len = max_len

        for ex in hf_ds:
            q = ex.get("query")
            passages = ex.get("passages")
            if q is None or passages is None:
                continue

            pos, neg = [], []
            for p, l in zip(
                passages.get("passage_text", []),
                passages.get("is_selected", []),
            ):
                if p is None:
                    continue
                if int(l) == 1:
                    pos.append(p)
                else:
                    neg.append(p)

            # keep queries with at least one positive (negatives may be zero)
            if len(pos) > 0:
                self.data.append((q, pos, neg))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, pos, neg = self.data[idx]
        # return sequences already encoded (lists of ints)
        return (
            encode(q, self.vocab, self.max_len),
            [encode(p, self.vocab, self.max_len) for p in pos],
            [encode(n, self.vocab, self.max_len) for n in neg],
        )


# ============================================================
# COLLATE (produces p_ids as POS+NEG grouped per query, and pos_mask)
# Returns: q_ids, q_mask, p_ids, p_mask, pos_mask
# - pos_mask is float/tensor with 1.0 where passage is positive for that query
# ============================================================
def collate(batch):
    def pad(seqs):
        maxl = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), maxl), PAD_ID, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids, mask

    q_ids_list = []
    p_grouped = []  # flattened list of all passages (positives then negatives per query)
    pos_ranges = []  # list of (start_idx, n_pos) for each query

    for q, pos_list, neg_list in batch:
        q_ids_list.append(q)
        start = len(p_grouped)
        # group positives first, then negatives
        group = pos_list + neg_list
        p_grouped.extend(group)
        pos_ranges.append((start, len(pos_list)))

    # Encode / pad
    q_ids, q_mask = pad(q_ids_list)
    if len(p_grouped) == 0:
        # Edge case: no passages at all (shouldn't happen because each query has >=1 pos),
        # but handle defensively:
        p_ids = torch.full((1, 1), PAD_ID, dtype=torch.long)
        p_mask = torch.zeros_like(p_ids)
        pos_mask = torch.zeros((len(batch), 1), dtype=torch.float)
        return q_ids, q_mask, p_ids, p_mask, pos_mask

    p_ids, p_mask = pad(p_grouped)

    # Build pos_mask: shape (B, P)
    B = len(batch)
    P = p_ids.size(0)
    pos_mask = torch.zeros((B, P), dtype=torch.float)
    for i, (start, npos) in enumerate(pos_ranges):
        if npos > 0:
            pos_mask[i, start : start + npos] = 1.0

    return q_ids, q_mask, p_ids, p_mask, pos_mask


# ============================================================
# MULTI-POSITIVE LOSS (supports multiple positives per query)
# ============================================================
def multi_pos_contrastive_loss(q_emb, p_emb, pos_mask, temp=0.05):
    """
    q_emb: (B, D)
    p_emb: (P, D)
    pos_mask: (B, P) float {0,1}
    """
    logits = (q_emb @ p_emb.T) / temp  # (B, P)
    logp = F.log_softmax(logits, dim=1)
    pos_sum = (logp * pos_mask).sum(dim=1)  # sum log-probs of positive positions
    denom = pos_mask.sum(dim=1).clamp(min=1.0)
    loss_per_query = -pos_sum / denom
    return loss_per_query.mean()


# ============================================================
# METRICS (multi-positive)
# Returns dict of top1, mrr, r@5, r@10 averaged over batch
# ============================================================
def compute_metrics(q_emb, p_emb, pos_mask):
    sims = q_emb @ p_emb.T  # (B, P)
    B, P = sims.shape

    top1 = 0.0
    mrr = 0.0
    r5 = 0.0
    r10 = 0.0

    # compute per-query using best positive score (handles multiple positives)
    for i in range(B):
        pos_indices = (pos_mask[i] > 0.5).nonzero(as_tuple=False).flatten()
        if pos_indices.numel() == 0:
            # no positive (shouldn't happen), skip
            continue
        scores = sims[i]  # (P,)
        pos_scores = scores[pos_indices]
        best_pos_score = pos_scores.max()
        # rank = number of items with strictly greater score
        rank = int((scores > best_pos_score).sum().item())
        # Top1
        if rank == 0:
            top1 += 1.0
        # MRR
        mrr += 1.0 / (rank + 1.0)
        # Recall@K
        if rank < 5:
            r5 += 1.0
        if rank < 10:
            r10 += 1.0

    # normalize by B
    denom = float(B)
    return {
        "top1": top1 / denom,
        "mrr": mrr / denom,
        "r@5": r5 / denom,
        "r@10": r10 / denom,
    }


# ============================================================
# TRAIN / EVAL (supports bf16 autocast; uses GradScaler only for fp16)
# ============================================================
# Choose AMP dtype here. Set to torch.bfloat16 for bf16, or torch.float16 for fp16.
AMP_DTYPE = torch.bfloat16
_SCALER = GradScaler() if AMP_DTYPE is torch.float16 else None


def run_epoch(model, loader, opt=None):
    train = opt is not None
    total_loss = 0.0
    metrics_acc = {"top1": 0.0, "mrr": 0.0, "r@5": 0.0, "r@10": 0.0}
    batch_count = 0

    if train:
        model.train()
    else:
        model.eval()

    for q_ids, q_mask, p_ids, p_mask, pos_mask in tqdm(loader, leave=False):
        q_ids = q_ids.cuda(non_blocking=True)
        q_mask = q_mask.cuda(non_blocking=True)
        p_ids = p_ids.cuda(non_blocking=True)
        p_mask = p_mask.cuda(non_blocking=True)
        pos_mask = pos_mask.cuda(non_blocking=True)

        if train:
            opt.zero_grad()

        # forward w/ autocast
        with autocast(dtype=AMP_DTYPE):
            q_emb = model(q_ids, q_mask)  # (B, D)
            p_emb = model(p_ids, p_mask)  # (P, D)
            loss = multi_pos_contrastive_loss(q_emb, p_emb, pos_mask)

        if train:
            if _SCALER is not None:
                _SCALER.scale(loss).backward()
                _SCALER.step(opt)
                _SCALER.update()
            else:
                loss.backward()
                opt.step()

        total_loss += loss.item()
        batch_metrics = compute_metrics(q_emb.detach(), p_emb.detach(), pos_mask.detach())
        for k in metrics_acc:
            metrics_acc[k] += batch_metrics[k]
        batch_count += 1

    if batch_count == 0:
        return 0.0, {k: 0.0 for k in metrics_acc}

    avg_loss = total_loss / batch_count
    avg_metrics = {k: metrics_acc[k] / batch_count for k in metrics_acc}
    return avg_loss, avg_metrics


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm_ckpt", required=True)
    parser.add_argument("--batch_size", type=int, default=4, help="number of queries per batch")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", default="bilstm_best.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MLM checkpoint (DO NOT CHANGE VOCAB)
    ckpt = torch.load(args.mlm_ckpt, map_location="cpu")
    vocab = {t: i for i, t in enumerate(ckpt["vocab"])}
    cfg = ckpt["config"]

    # Load datasets
    train_ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    val_ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")

    # train_ds = train_ds.select(range(10000))   # first 100 examples
    # val_ds   = val_ds.select(range(2000))      # first 20 examples

    train_set = MSMarcoGrouped(train_ds, vocab, args.max_len)
    val_set = MSMarcoGrouped(val_ds, vocab, args.max_len)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True,
        num_workers=2,
    )

    print(f"Dataset sizes | train={len(train_set)} val={len(val_set)}")
    print(f"Using AMP dtype: {AMP_DTYPE}, using scaler: {_SCALER is not None}")

    # Model
    model = BiLSTMEncoder(
        vocab_size=len(vocab),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
    ).to(device)

    # Load encoder weights from MLM (if present)
    model.load_state_dict(
        {k.replace("encoder.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("encoder.")},
        strict=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop with BEST checkpoint saving
    best_val_loss = float("inf")
    save_path = Path(args.save_path)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_metrics = run_epoch(model, train_loader, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader)

        print(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        print(f"  train_metrics: {train_metrics}")
        print(f"  val_metrics:   {val_metrics}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": ckpt["vocab"],
                    "config": cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )
            print(f"  ✓ Saved best model → {save_path} (val_loss={val_loss:.4f})")

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()