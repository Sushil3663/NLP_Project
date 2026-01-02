#!/usr/bin/env python3
"""
train_with_curation.py

Train LSTM QA on a curated SQuAD v2 dataset using your checkpoint vocab and conversion diagnostics.

Typical usage:
  python train_with_curation.py --ckpt bilstm_best.pt --out_dir results --batch_size 32

If you already have the conversion JSONL produced by realign_squad_wordpos.py,
pass --conversion_jsonl path/to/squad_realign_validation.jsonl to reuse those diagnostics.
If not provided, this script will re-run the simple regex mapping inline.

Outputs:
 - curated_train_indices.json, curated_validation_indices.json
 - curation_summary.json
 - model checkpoints saved as training proceeds
"""

import os
import json
import re
import argparse
from collections import Counter, defaultdict
from datasets import load_dataset
import torch
import unicodedata
from tqdm import tqdm
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =========================
# Constants & tokenization
# =========================
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

def normalize_text(s):
    s = unicodedata.normalize("NFC", s)
    s = " ".join(s.split())
    return s.lower().strip()

def tokenize_with_spans(text, max_tokens=None):
    """Return list of tokens and list of (start_char, end_char) spans using re.finditer."""
    if not text:
        return [], []
    tokens = []
    spans = []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group(0).lower())
        spans.append((m.start(), m.end()))
        if max_tokens is not None and len(tokens) >= max_tokens:
            break
    return tokens, spans

# =========================
# Load vocab from checkpoint
# =========================
def load_vocab_from_checkpoint(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "vocab" not in ckpt:
        raise KeyError("Checkpoint does not contain 'vocab' key")
    vocab_itos = list(ckpt["vocab"])
    stoi = {t: i for i, t in enumerate(vocab_itos)}
    return vocab_itos, stoi, ckpt

# =========================
# Conversion JSONL loader
# =========================
def load_conversion_jsonl(jsonl_path):
    """Return dict: key -> (example_index or id) -> list of per-answer dicts (as in your out jsonl)."""
    mapping = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            # key by SQuAD example id if present, else by example_index
            key = obj.get("id") if obj.get("id") is not None else str(obj.get("example_index"))
            mapping[key].append(obj)
    return mapping

# -------------------------
# Inline re-alignment (fallback)
# -------------------------
def compute_alignment_for_example(ex, stoi, max_ctx_tokens=None):
    """
    Recompute the mapping for a single example using the same tokenization logic as earlier.
    Returns list of per-answer dicts similar to your conversion JSONL outputs.
    """
    context = ex["context"]
    answers = ex["answers"]["text"]
    answer_starts = ex["answers"]["answer_start"]
    outputs = []

    ctx_tokens, ctx_spans = tokenize_with_spans(context, max_tokens=max_ctx_tokens)
    for a_i, (answer_text, answer_start) in enumerate(zip(answers, answer_starts)):
        rec = {
            "id": ex.get("id"),
            "example_index": ex.get("__index__", None),
            "answer_index": a_i,
            "answer_text": answer_text,
            "answer_start_char": answer_start,
            "start_tok": None,
            "end_tok": None,
            "tokens_in_span": None,
            "char_slice": None,
            "reconstructed_from_tokens": None,
            "unk_tokens": [],
            "success_mapped_to_tokens": False,
            "alignment_good_perfect_match": False,
            "reasons": []
        }
        ans_len = len(answer_text)
        # find overlap
        ans_start = answer_start
        ans_end = ans_start + ans_len
        start_tok = None
        end_tok = None
        for i,(s,e) in enumerate(ctx_spans):
            if s < ans_end and e > ans_start:
                if start_tok is None:
                    start_tok = i
                end_tok = i
        if start_tok is None:
            rec["reasons"].append("no_overlapping_tokens")
        else:
            rec["start_tok"] = start_tok
            rec["end_tok"] = end_tok
            rec["tokens_in_span"] = ctx_tokens[start_tok:end_tok+1]
            rec["char_slice"] = context[ctx_spans[start_tok][0]: ctx_spans[end_tok][1]]
            rec["reconstructed_from_tokens"] = " ".join(ctx_tokens[start_tok:end_tok+1])
            for tok in rec["tokens_in_span"]:
                if tok not in stoi:
                    rec["unk_tokens"].append(tok)
            rec["success_mapped_to_tokens"] = True
            norm_answer = normalize_text(answer_text)
            norm_recon = normalize_text(rec["reconstructed_from_tokens"])
            norm_char_slice = normalize_text(rec["char_slice"])
            if norm_recon == norm_answer:
                rec["alignment_good_perfect_match"] = True
            elif norm_char_slice == norm_answer:
                rec["alignment_good_perfect_match"] = True
                rec["reasons"].append("whitespace_or_punct_spacing_difference")
            else:
                # imperfect: diagnose
                if rec["unk_tokens"]:
                    rec["reasons"].append("unknown_tokens_in_span")
                if norm_answer in norm_char_slice or norm_char_slice in norm_answer:
                    rec["reasons"].append("partial_containment_between_char_slice_and_answer")
                else:
                    rec["reasons"].append("reconstructed_mismatch")
        outputs.append(rec)
    return outputs

# =========================
# Curation logic
# =========================
def curate_split(hf_split, conversion_map, stoi, keep_partial=False, allow_unknowns=False, max_unk_in_span=0, max_ctx_tokens=None):
    """
    Return a list of indices to KEEP from hf_split (so training uses only these examples).
    - conversion_map: dict keyed by example id or example_index (string) -> list(per-answer dicts)
    - keep_partial: if True, accept partial containment / whitespace diffs, otherwise require alignment_good_perfect_match
    - allow_unknowns: if True, allow unknown tokens in the span up to max_unk_in_span
    """
    kept_indices = []
    stats = Counter()
    reason_counts = Counter()
    sample_examples = defaultdict(list)
    MAX_SAMPLE_PER_REASON = 5

    for idx, ex in enumerate(tqdm(hf_split, desc="Curating split", leave=False)):
        ex_id = ex.get("id")
        key_by_id = ex_id
        key_by_idx = str(idx)
        answers = ex["answers"]["text"]
        is_impossible = len(answers) == 0

        # get mapping entries (either precomputed or compute inline)
        entries = conversion_map.get(key_by_id) or conversion_map.get(key_by_idx)
        if entries is None:
            # fallback compute
            entries = compute_alignment_for_example(ex, stoi, max_ctx_tokens=max_ctx_tokens)

        # If unanswerable, keep by default
        if is_impossible:
            kept_indices.append(idx)
            stats["kept_unanswerable"] += 1
            continue

        # For answerable: keep if at least one answer passes filters
        any_ok = False
        for rec in entries:
            # if mapping failed to tokens -> skip this answer
            if not rec.get("success_mapped_to_tokens"):
                reason_counts["no_overlapping_tokens"] += 1
                if len(sample_examples["no_overlapping_tokens"]) < MAX_SAMPLE_PER_REASON:
                    sample_examples["no_overlapping_tokens"].append({"id": ex_id, "idx": idx})
                continue

            aln_good = bool(rec.get("alignment_good_perfect_match"))
            unk_count = len(rec.get("unk_tokens", []))
            rec_reasons = rec.get("reasons", [])

            # determine acceptability
            accept = False
            if aln_good:
                accept = True
            else:
                # imperfect
                if keep_partial and ("partial_containment_between_char_slice_and_answer" in rec_reasons or "whitespace_or_punct_spacing_difference" in rec_reasons):
                    accept = True
                elif allow_unknowns and unk_count <= max_unk_in_span:
                    accept = True

            # record reasons
            if accept:
                any_ok = True
                stats["kept_answerable_ok"] += 1
                if unk_count > 0:
                    stats["kept_with_unk"] += 1
                    reason_counts["unknown_tokens_in_span"] += 1
                    if len(sample_examples["unknown_tokens_in_span"]) < MAX_SAMPLE_PER_REASON:
                        sample_examples["unknown_tokens_in_span"].append({"id": ex_id, "idx": idx, "unk": rec.get("unk_tokens")})
                break
            else:
                # collect reasons for failing this answer
                for r in rec_reasons:
                    reason_counts[r] += 1
                    if len(sample_examples[r]) < MAX_SAMPLE_PER_REASON:
                        sample_examples[r].append({"id": ex_id, "idx": idx})

        if any_ok:
            kept_indices.append(idx)
        else:
            stats["filtered_answerable"] += 1

    total = len(hf_split)
    kept = len(kept_indices)
    stats.update({
        "total_examples": total,
        "kept_examples": kept,
        "filtered_examples": total - kept
    })

    summary = {
        "stats": dict(stats),
        "reason_counts": dict(reason_counts),
        "sample_examples_by_reason": {k: v for k, v in sample_examples.items()}
    }
    return kept_indices, summary

# =========================
# Adapted Dataset using curated indices
# =========================
class CuratedSquadV2Dataset(Dataset):
    def __init__(self, hf_ds, kept_indices, stoi, max_ctx=384, max_q=64, verbose=False):
        self.ds = hf_ds
        self.indices = kept_indices
        self.max_ctx = max_ctx
        self.max_q = max_q
        self.stoi = stoi
        self.verbose = verbose

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        ex = self.ds[idx]
        context = ex["context"]
        question = ex["question"]
        answers = ex["answers"]

        ctx_tokens, ctx_spans = tokenize_with_spans(context, max_tokens=self.max_ctx - 1)
        q_tokens, _ = tokenize_with_spans(question, max_tokens=self.max_q)

        ctx_ids = [CLS_ID] + [self.stoi.get(t, UNK_ID) for t in ctx_tokens]
        q_ids = [self.stoi.get(t, UNK_ID) for t in q_tokens]

        start = end = 0
        gold_texts = []
        is_impossible = len(answers["text"]) == 0

        if not is_impossible:
            # Find the first overlapping token span as best effort mapping
            ans_text = answers["text"][0]
            ans_start_char = answers["answer_start"][0]
            ans_end_char = ans_start_char + len(ans_text)
            s_tok = None
            e_tok = None
            for t_i,(s_char,e_char) in enumerate(ctx_spans):
                if s_tok is None and e_char > ans_start_char:
                    s_tok = t_i
                if s_char < ans_end_char:
                    e_tok = t_i
            if s_tok is not None and e_tok is not None:
                start = s_tok + 1
                end = e_tok + 1
                gold_texts = [ans_text.lower()]
        # else remains no-answer at start=end=0

        return {
            "context_ids": ctx_ids,
            "question_ids": q_ids,
            "start": start,
            "end": end,
            "answers": gold_texts,
            "context_tokens": ctx_tokens,
            "is_impossible": is_impossible
        }

# =========================
# Collate / model / training scaffolding
# (Mostly taken from your training script; unchanged semantics)
# =========================

def collate_fn(batch):
    B = len(batch)
    max_c = max(len(x["context_ids"]) for x in batch)
    max_q = max(len(x["question_ids"]) for x in batch)

    ctx = torch.full((B, max_c), PAD_ID, dtype=torch.long)
    q = torch.full((B, max_q), PAD_ID, dtype=torch.long)
    ctx_mask = torch.zeros((B, max_c), dtype=torch.float32)
    q_mask = torch.zeros((B, max_q), dtype=torch.float32)

    start = torch.zeros(B, dtype=torch.long)
    end = torch.zeros(B, dtype=torch.long)

    for i, x in enumerate(batch):
        ctx[i, :len(x["context_ids"])] = torch.tensor(x["context_ids"], dtype=torch.long)
        q[i, :len(x["question_ids"])] = torch.tensor(x["question_ids"], dtype=torch.long)
        ctx_mask[i, :len(x["context_ids"])] = 1
        q_mask[i, :len(x["question_ids"])] = 1
        start[i] = x["start"]
        end[i] = x["end"]

    return ctx, ctx_mask, q, q_mask, start, end, batch

# -------------------------
# Reuse your FlexibleLSTMBase, ContextQuestionAttention, LSTMQA, metrics, evaluate, best_span, etc.
# For brevity we reuse similar classes from your original training file.
# Insert here the exact model classes (omitted in this listing to keep code short).
# In practice copy your FlexibleLSTMBase, ContextQuestionAttention and LSTMQA classes here.
# -------------------------

# For demonstration, implement minimal placeholders (replace with your full model code)
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
    def encode(self, input_ids, attention_mask, mode='sequence'):
        seq_out, (h_n, c_n) = self.forward(input_ids, attention_mask)
        if mode == 'sequence':
            return seq_out
        if mode == 'pooled':
            mask = attention_mask.unsqueeze(-1)
            pooled = (seq_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return F.normalize(self.proj(pooled), dim=1)
        raise ValueError()

class ContextQuestionAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sim = nn.Linear(dim * 3, 1, bias=False)
    def forward(self, C, Q, q_mask):
        B, C_len, H = C.shape
        Q_len = Q.size(1)
        C_exp = C.unsqueeze(2).expand(B, C_len, Q_len, H)
        Q_exp = Q.unsqueeze(1).expand(B, C_len, Q_len, H)
        S = self.sim(torch.cat([C_exp, Q_exp, C_exp * Q_exp], dim=-1)).squeeze(-1)
        S = S.masked_fill(q_mask.unsqueeze(1) == 0, -1e9)
        A = torch.softmax(S, dim=-1) @ Q
        maxS, _ = S.max(dim=-1)
        B_att = torch.softmax(maxS, dim=-1)
        B_vec = (B_att.unsqueeze(-1) * C).sum(dim=1).unsqueeze(1).expand(-1, C_len, -1)
        return torch.cat([C, A, C * A, C * B_vec], dim=-1)

class LSTMQA(nn.Module):
    def __init__(self, ckpt_cfg):
        super().__init__()
        base_kwargs = dict(
            vocab_size=ckpt_cfg["vocab_size"],
            emb_dim=ckpt_cfg["emb_dim"],
            hidden_dim=ckpt_cfg["hidden_dim"],
            num_layers=ckpt_cfg.get("num_layers",1),
            bidirectional=ckpt_cfg.get("bidirectional", True),
            pad_id=PAD_ID,
        )
        self.ctx_enc = FlexibleLSTMBase(**base_kwargs)
        self.q_enc = FlexibleLSTMBase(**base_kwargs)
        H = self.ctx_enc.out_dim
        self.att = ContextQuestionAttention(H)
        self.modeling = nn.LSTM(H * 4, H // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.start = nn.Linear(H, 1)
        self.end = nn.Linear(H, 1)
        self.no_answer = nn.Linear(H, 1)
    def forward(self, ctx, ctx_mask, q, q_mask):
        C = self.ctx_enc.encode(ctx, ctx_mask, mode="sequence")
        Q = self.q_enc.encode(q, q_mask, mode="sequence")
        G = self.att(C, Q, q_mask)
        M, _ = self.modeling(G)
        M = self.dropout(M)
        s = self.start(M).squeeze(-1)
        e = self.end(M).squeeze(-1)
        na = self.no_answer(M[:, 0, :]).squeeze(-1)
        s = s.masked_fill(ctx_mask == 0, -1e9)
        e = e.masked_fill(ctx_mask == 0, -1e9)
        return s, e, na

# Metrics from your training script (exact_match, f1) - copied here:
def normalize_for_metrics(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(pred, gold):
    return normalize_for_metrics(pred) == normalize_for_metrics(gold)

def f1_score(pred, gold):
    p = normalize_for_metrics(pred).split()
    g = normalize_for_metrics(gold).split()
    common = Counter(p) & Counter(g)
    if not p or not g:
        return float(p == g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    return 2 * num_same / (len(p) + len(g))

def best_span(start_logits, end_logits, max_len=30):
    L = start_logits.size(0)
    best_s, best_e, best_score = 0, 0, -1e9
    max_len = min(max_len, L)
    start_logits = start_logits.cpu().tolist()
    end_logits = end_logits.cpu().tolist()
    for s in range(L):
        for e in range(s, min(s + max_len, L)):
            score = start_logits[s] + end_logits[e]
            if score > best_score:
                best_score = score
                best_s, best_e = s, e
    return best_s, best_e, best_score

def evaluate(model, loader, device, max_span_len=30, threshold=0.5):
    model.eval()
    EM = F1 = N = 0
    pred_no_answer = gold_no_answer = correct_no_answer = 0
    has_answer_em = has_answer_f1 = has_answer_count = 0

    with torch.no_grad():
        for ctx, ctx_m, q, q_m, _, _, raw in loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            q, q_m = q.to(device), q_m.to(device)
            ps, pe, na = model(ctx, ctx_m, q, q_m)
            for i, ex in enumerate(raw):
                ps_i = ps[i]
                pe_i = pe[i]
                na_i = na[i]
                s_i, e_i, span_score = best_span(ps_i, pe_i, max_len=max_span_len)
                has_ans_prob = torch.sigmoid(na_i).item()
                golds = ex["answers"]
                is_gold_no_answer = len(golds) == 0
                if is_gold_no_answer:
                    gold_no_answer += 1
                if has_ans_prob < threshold:
                    pred = ""
                    pred_no_answer += 1
                else:
                    if s_i == 0 or e_i < s_i:
                        pred = ""
                        pred_no_answer += 1
                    else:
                        pred = " ".join(ex["context_tokens"][s_i - 1: e_i])
                if is_gold_no_answer:
                    em_score = int(pred == "")
                    f1_s = int(pred == "")
                    if pred == "":
                        correct_no_answer += 1
                else:
                    em_score = max(exact_match(pred, g) for g in golds)
                    f1_s = max(f1_score(pred, g) for g in golds)
                    has_answer_em += em_score
                    has_answer_f1 += f1_s
                    has_answer_count += 1
                EM += em_score
                F1 += f1_s
                N += 1

    overall_em = EM / N * 100 if N>0 else 0.0
    overall_f1 = F1 / N * 100 if N>0 else 0.0
    has_ans_em = has_answer_em / has_answer_count * 100 if has_answer_count > 0 else 0.0
    has_ans_f1 = has_answer_f1 / has_answer_count * 100 if has_answer_count > 0 else 0.0
    no_ans_acc = correct_no_answer / gold_no_answer * 100 if gold_no_answer > 0 else 0.0
    print(f"  Overall: EM={overall_em:.2f}, F1={overall_f1:.2f}")
    print(f"  Has-Answer: EM={has_ans_em:.2f}, F1={has_ans_f1:.2f} (n={has_answer_count})")
    print(f"  No-Answer: Acc={no_ans_acc:.2f}% (n={gold_no_answer})")
    print(f"  Predictions: {pred_no_answer}/{N} no-answer ({pred_no_answer/N*100:.1f}%)")
    return overall_em, overall_f1

# =========================
# Training entrypoint - wiring everything together
# =========================
def train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading checkpoint vocab...")
    vocab_itos, stoi, ckpt = load_vocab_from_checkpoint(args.ckpt)
    cfg = {
        "vocab_size": len(vocab_itos),
        "emb_dim": ckpt["config"]["emb_dim"] if "config" in ckpt and "emb_dim" in ckpt["config"] else 300,
        "hidden_dim": ckpt["config"]["hidden_dim"] if "config" in ckpt and "hidden_dim" in ckpt["config"] else 256,
        "num_layers": ckpt["config"].get("num_layers", 1) if "config" in ckpt else 1,
        "bidirectional": ckpt["config"].get("bidirectional", True) if "config" in ckpt else True
    }

    print("Loading SQuAD v2 dataset from HuggingFace...")
    hf = load_dataset("rajpurkar/squad_v2")
    # Optionally reuse a precomputed conversion JSONL
    conversion_map_train = {}
    conversion_map_val = {}
    if args.conversion_jsonl:
        print(f"Loading conversion diagnostics from {args.conversion_jsonl} ...")
        conv = load_conversion_jsonl(args.conversion_jsonl)
        # conv contains entries for the split the JSONL was made against (likely validation). We'll reuse for the matching split.
        conversion_map_val = conv
    # Curate train split (we re-run mapping for train by default)
    kept_train, summary_train = curate_split(hf["train"], conversion_map_train, stoi,
                                            keep_partial=args.keep_partial,
                                            allow_unknowns=args.allow_unknowns,
                                            max_unk_in_span=args.max_unk_in_span,
                                            max_ctx_tokens=args.max_ctx_tokens)
    kept_val, summary_val = curate_split(hf["validation"], conversion_map_val if conversion_map_val else {}, stoi,
                                         keep_partial=args.keep_partial,
                                         allow_unknowns=args.allow_unknowns,
                                         max_unk_in_span=args.max_unk_in_span,
                                         max_ctx_tokens=args.max_ctx_tokens)

    # Write curated indices and summaries
    with open(os.path.join(args.out_dir, "curated_train_indices.json"), "w", encoding="utf-8") as f:
        json.dump(kept_train, f)
    with open(os.path.join(args.out_dir, "curated_validation_indices.json"), "w", encoding="utf-8") as f:
        json.dump(kept_val, f)
    curation_summary = {
        "train": summary_train,
        "validation": summary_val,
        "vocab_size": len(vocab_itos),
        "total_train_examples": len(hf["train"]),
        "total_validation_examples": len(hf["validation"])
    }
    with open(os.path.join(args.out_dir, "curation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(curation_summary, f, indent=2, ensure_ascii=False)

    print("\nCuration summary (train):")
    print(json.dumps(summary_train["stats"], indent=2))
    print("Curation summary (validation):")
    print(json.dumps(summary_val["stats"], indent=2))

    # Create curated datasets
    train_ds = CuratedSquadV2Dataset(hf["train"], kept_train, stoi, max_ctx=args.max_ctx, max_q=args.max_q, verbose=False)
    val_ds = CuratedSquadV2Dataset(hf["validation"], kept_val, stoi, max_ctx=args.max_ctx, max_q=args.max_q, verbose=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMQA(cfg).to(device)
    # Optionally load pretrained encoder weights if available in ckpt['model_state']
    if "model_state" in ckpt:
        try:
            model.ctx_enc.load_state_dict(ckpt["model_state"], strict=False)
            model.q_enc.load_state_dict(ckpt["model_state"], strict=False)
            print("Loaded encoder weights from ckpt['model_state'] (non-strict).")
        except Exception as e:
            print("Warning: failed to load model_state into encoders:", e)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_span = nn.CrossEntropyLoss()
    loss_na = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for ctx, ctx_m, q, q_m, s, e, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            q, q_m = q.to(device), q_m.to(device)
            s, e = s.to(device), e.to(device)

            optim.zero_grad()
            ps, pe, na = model(ctx, ctx_m, q, q_m)
            loss_s = loss_span(ps, s)
            loss_e = loss_span(pe, e)
            has_answer = (s != 0).float().to(device)
            loss_n = loss_na(na, has_answer)
            loss = loss_s + loss_e + args.na_weight * loss_n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"\nEpoch {epoch+1} | Avg Loss={avg_loss:.4f} | LR={args.lr}")
        em, f1 = evaluate(model, val_loader, device, max_span_len=args.max_span_len, threshold=args.na_threshold)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.out_dir, "lstm_squadv2_best.pt"))
            print(f"  ✓ Best model saved (F1={best_f1:.2f})")

    print("Training complete. Best F1:", best_f1)

# =========================
# CLI
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to pretrained .pt checkpoint containing ckpt['vocab']")
    p.add_argument("--out_dir", default="results", help="Output dir for curated lists, summaries, checkpoints")
    p.add_argument("--conversion_jsonl", default=None, help="Optional conversion jsonl to reuse (from realign_squad_wordpos.py)")
    p.add_argument("--keep_partial", action="store_true", help="Allow partial containment / whitespace differences to be accepted")
    p.add_argument("--allow_unknowns", action="store_true", help="Allow spans with unknown tokens (controlled by --max_unk_in_span)")
    p.add_argument("--max_unk_in_span", type=int, default=0, help="Max unknown tokens allowed in span to accept when --allow_unknowns")
    p.add_argument("--max_ctx_tokens", type=int, default=384, help="Max context tokens for tokenization/truncation")
    p.add_argument("--max_q", type=int, default=64, help="Max question tokens")
    p.add_argument("--max_ctx", type=int, default=384, help="Max context tokens (dataset)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--na_weight", type=float, default=0.3)
    p.add_argument("--max_span_len", type=int, default=30)
    p.add_argument("--na_threshold", type=float, default=0.5)
    args = p.parse_args()
    train(args)
