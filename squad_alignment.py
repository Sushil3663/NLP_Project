#!/usr/bin/env python3
"""
realign_squad_wordpos.py

Converts char-based answer start positions in rajpurkar/squad_v2 to word/token positions
based on a pretrained .pt vocabulary (ckpt['vocab']). Saves per-example diagnostics (jsonl)
and a summary (json). Does not print to stdout by default.

Usage:
    python realign_squad_wordpos.py --ckpt path/to/model.pt --out_dir results --split validation

Requirements:
    pip install datasets torch transformers

Notes:
    - The script uses a simple regex tokenizer (r"\w+|[^\s\w]") to produce tokens and character spans.
    - If your pretrained vocab uses a different tokenization scheme (BPE/WordPiece), provide that tokenizer
      instead and modify the tokenization accordingly (or pass --use_hf_tokenizer <hf-tokenizer-name>)
"""

import os
import json
import re
import argparse
from collections import Counter, defaultdict
from datasets import load_dataset
import torch
import unicodedata

# -------------------------
# Tokenization config
# -------------------------
TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)

def normalize_text(s):
    # NFC normalization + collapse whitespace + lowercase
    s = unicodedata.normalize("NFC", s)
    s = " ".join(s.split())
    return s.lower().strip()

def word_tokenize_with_spans(text, max_tokens=None):
    """Return list of tokens and list of (start_char, end_char) spans."""
    if not text:
        return [], []
    tokens = []
    spans = []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group(0).lower())  # lower to match typical vocab
        spans.append((m.start(), m.end()))
        if max_tokens is not None and len(tokens) >= max_tokens:
            break
    return tokens, spans

# -------------------------
# load vocab from .pt
# -------------------------
def load_vocab_from_checkpoint(checkpoint_path):
    """
    Load checkpoint and return vocab_itos (list) and stoi dict.
    Expect ckpt['vocab'] to be an iterable of token strings.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "vocab" not in ckpt:
        raise KeyError("Checkpoint does not contain 'vocab' key")
    vocab_itos = list(ckpt["vocab"])
    stoi = {t: i for i, t in enumerate(vocab_itos)}
    return vocab_itos, stoi

# -------------------------
# Span mapping helpers
# -------------------------
def find_span_by_offsets(offsets, answer_start, answer_text_len):
    """Return (start_tok, end_tok) indices (inclusive), or (None, None)."""
    ans_start = answer_start
    ans_end = ans_start + answer_text_len
    start_tok = None
    end_tok = None
    for i, (s, e) in enumerate(offsets):
        # overlap check for half-open intervals [s,e) and [ans_start, ans_end)
        if s < ans_end and e > ans_start:
            if start_tok is None:
                start_tok = i
            end_tok = i
    return start_tok, end_tok

# -------------------------
# Main processing
# -------------------------
def process_split(split="validation", checkpoint_path=None, out_dir="results", max_ctx_tokens=None):
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"squad_realign_{split}.jsonl")
    summary_path = os.path.join(out_dir, f"summary_{split}.json")

    # Load dataset
    ds = load_dataset("rajpurkar/squad_v2", split=split)

    # Load vocab
    vocab_itos, stoi = load_vocab_from_checkpoint(checkpoint_path)

    counters = Counter()
    reason_counts = Counter()
    top_reasons_examples = defaultdict(list)  # reason -> list of example ids (small sample)
    MAX_SAMPLE_PER_REASON = 5

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(ds):
            # Example metadata
            ex_id = ex.get("id", None)
            context = ex["context"]
            answers = ex["answers"]["text"]
            answer_starts = ex["answers"]["answer_start"]

            if not answers:
                # unanswerable example
                out = {
                    "id": ex_id,
                    "example_index": idx,
                    "unanswerable": True
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                counters["unanswerable"] += 1
                continue

            # We'll align each annotated answer (loop) — SQuAD usually has multiple answers
            per_example_results = []
            for a_i, (answer_text, answer_start) in enumerate(zip(answers, answer_starts)):
                # Normalize (only for comparison; mapping uses raw characters)
                answer_text_orig = answer_text
                answer_len = len(answer_text_orig)

                # tokenize context using your regex tokenizer
                ctx_tokens, ctx_spans = word_tokenize_with_spans(context, max_tokens=max_ctx_tokens)

                # Map char-based answer -> token indices
                start_tok, end_tok = find_span_by_offsets(ctx_spans, answer_start, answer_len)

                reasons = []
                success = False
                alignment_good = False
                unk_tokens = []
                reconstructed_tokens_text = None
                char_slice = None

                if start_tok is None or end_tok is None:
                    reasons.append("no_overlapping_tokens")
                    # possible causes: truncation, tokenizer mismatch, wrong start
                else:
                    # reconstruct from token list (join with original spacing inferred from spans)
                    span_char_start = ctx_spans[start_tok][0]
                    span_char_end = ctx_spans[end_tok][1]
                    char_slice = context[span_char_start:span_char_end]
                    # Reconstructed as token-joined string (space-joined)
                    reconstructed_tokens_text = " ".join(ctx_tokens[start_tok:end_tok+1])
                    # gather unk tokens relative to your vocab
                    for tok in ctx_tokens[start_tok:end_tok+1]:
                        if tok not in stoi:
                            unk_tokens.append(tok)

                    # Alignment checks
                    norm_answer = normalize_text(answer_text_orig)
                    norm_reconstructed_from_tokens = normalize_text(reconstructed_tokens_text)
                    norm_char_slice = normalize_text(char_slice)

                    # Perfect conversion: if normalized reconstructed token text (or normalized char_slice) equals answer
                    if norm_reconstructed_from_tokens == norm_answer:
                        alignment_good = True
                        success = True
                    elif norm_char_slice == norm_answer:
                        # char slice matches but token-join differs (whitespace/punctuation differences)
                        alignment_good = True
                        success = True
                        reasons.append("whitespace_or_punct_spacing_difference")
                    else:
                        # Not exact match — diagnose reasons
                        success = True  # we still found tokens, but not perfect
                        # If there are unknown tokens relative to your vocab, it's likely a vocab mismatch
                        if unk_tokens:
                            reasons.append("unknown_tokens_in_span")
                        # If reconstructed tokens differ from answer but char_slice contains answer -> tokenization spacing issue
                        if norm_answer in norm_char_slice or norm_char_slice in norm_answer:
                            reasons.append("partial_containment_between_char_slice_and_answer")
                        else:
                            reasons.append("reconstructed_mismatch")

                # pack per-answer output
                out = {
                    "id": ex_id,
                    "example_index": idx,
                    "answer_index": a_i,
                    "answer_text": answer_text_orig,
                    "answer_start_char": answer_start,
                    "start_tok": start_tok,
                    "end_tok": end_tok,
                    "tokens_in_span": ctx_tokens[start_tok:end_tok+1] if (start_tok is not None and end_tok is not None) else None,
                    "char_slice": char_slice,
                    "reconstructed_from_tokens": reconstructed_tokens_text,
                    "unk_tokens": unk_tokens,
                    "success_mapped_to_tokens": bool(start_tok is not None and end_tok is not None),
                    "alignment_good_perfect_match": alignment_good,
                    "reasons": reasons
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

                # update counters
                counters["total_answers_processed"] += 1
                if out["success_mapped_to_tokens"]:
                    counters["mapped_token_spans"] += 1
                else:
                    counters["mapping_failed_no_overlap"] += 1

                if alignment_good:
                    counters["perfect_conversions"] += 1
                else:
                    counters["imperfect_conversions"] += 1
                    for r in reasons:
                        reason_counts[r] += 1
                        if len(top_reasons_examples[r]) < MAX_SAMPLE_PER_REASON:
                            top_reasons_examples[r].append({"id": ex_id, "example_index": idx})

    # Build summary structure
    summary = {
        "counters": dict(counters),
        "fail_reasons_count": dict(reason_counts),
        "sample_examples_per_reason": dict(top_reasons_examples),
        "vocab_size": len(vocab_itos)
    }

    # Save summary
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    # Script intentionally prints nothing. Results written to:
    #   out_jsonl
    #   summary_path
    return out_jsonl, summary_path

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to pretrained .pt checkpoint containing ckpt['vocab']")
    parser.add_argument("--split", default="validation", choices=["train", "validation"], help="Dataset split to process")
    parser.add_argument("--out_dir", default="results", help="Directory to write results")
    parser.add_argument("--max_ctx_tokens", type=int, default=None, help="Optional max tokens to read from context (truncation)")
    args = parser.parse_args()

    # Run (silent)
    process_split(split=args.split, checkpoint_path=args.ckpt, out_dir=args.out_dir, max_ctx_tokens=args.max_ctx_tokens)
