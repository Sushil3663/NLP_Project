import torch
import torch.nn.functional as F
import random
from pathlib import Path

from bert_style_LSTM_MLM import FlexibleLSTMBase
# =========================
# 1. Load checkpoint
# =========================
CKPT_PATH = "models/mlm_bilstm_embed_100/bilstm_mlm_epoch3.pt"

ckpt = torch.load(CKPT_PATH, map_location="cpu")

vocab_itos = ckpt["vocab"]
stoi = {t: i for i, t in enumerate(vocab_itos)}

# Special tokens (MUST match training)
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

CLS_ID = stoi[CLS_TOKEN]
SEP_ID = stoi[SEP_TOKEN]
MASK_ID = stoi[MASK_TOKEN]
PAD_ID = stoi[PAD_TOKEN]
UNK_ID = stoi[UNK_TOKEN]

# =========================
# 2. Build model
# =========================
model = FlexibleLSTMBase(
    vocab_size=len(vocab_itos),
    emb_dim=ckpt["config"]["emb_dim"],
    hidden_dim=ckpt["config"]["hidden_dim"],
    num_layers=ckpt["config"]["num_layers"],
    bidirectional=ckpt["config"]["bidirectional"],
)

model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# =========================
# 3. Token helpers
# =========================
def encode(tokens):
    return [stoi.get(t, UNK_ID) for t in tokens]

def decode(ids):
    return [vocab_itos[i] for i in ids]

# Replace with *your* tokenizer
def tokenize(text):
    return text.lower().split()

# =========================
# 4. Random masking (BERT-style)
# =========================
def random_mask_tokens(
    tokens,
    mask_prob=0.15,
    min_masks=1,
):
    """
    Randomly mask tokens except CLS/SEP
    """
    candidate_positions = [
        i for i, t in enumerate(tokens)
        if t not in (CLS_TOKEN, SEP_TOKEN)
    ]

    num_to_mask = max(
        min_masks,
        int(len(candidate_positions) * mask_prob)
    )

    mask_positions = random.sample(
        candidate_positions,
        min(num_to_mask, len(candidate_positions))
    )

    masked_tokens = tokens.copy()
    for pos in mask_positions:
        masked_tokens[pos] = MASK_TOKEN

    return masked_tokens, mask_positions

# =========================
# 5. MLM prediction
# =========================
def predict_random_masks(
    text,
    mask_prob=0.15,
    top_k=5,
    max_length=256,
):
    tokens = tokenize(text)[: max_length - 2]
    tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]

    original_tokens = tokens.copy()

    masked_tokens, mask_positions = random_mask_tokens(
        tokens,
        mask_prob=mask_prob
    )

    input_ids = torch.tensor(encode(masked_tokens)).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)

    print("\n" + "=" * 80)
    print("ORIGINAL:")
    print(" ".join(original_tokens))

    print("\nMASKED:")
    print(" ".join(masked_tokens))

    print("\nPREDICTIONS:")

    for pos in mask_positions:
        gold_token = original_tokens[pos]

        probs = F.softmax(logits[0, pos], dim=-1)
        topk = torch.topk(probs, top_k)

        print(f"\n[MASK @ position {pos}] (gold: '{gold_token}')")
        for idx, prob in zip(topk.indices, topk.values):
            token = vocab_itos[idx.item()]
            print(f"  {token:<15} {prob.item():.4f}")

# =========================
# 6. Run example
# =========================
random.seed(100)
torch.manual_seed(42)

text = "The declaration of independence was signed in 1776"

predict_random_masks(
    text,
    mask_prob=0.3,
    top_k=7
)