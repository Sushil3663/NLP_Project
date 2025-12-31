import torch
import torch.nn.functional as F
from pathlib import Path

from bert_style_LSTM_MLM import FlexibleLSTMBase
# load checkpoint
ckpt = torch.load("models/mlm_bilstm/bilstm_mlm_epoch3.pt", map_location="cpu")

vocab_itos = ckpt["vocab"]
stoi = {t: i for i, t in enumerate(vocab_itos)}

def encode_tokens(tokens):
    return [stoi.get(t, UNK_ID) for t in tokens]

model = FlexibleLSTMBase(
    vocab_size=len(vocab_itos),
    emb_dim=ckpt["config"]["emb_dim"],
    hidden_dim=ckpt["config"]["hidden_dim"],
    num_layers=ckpt["config"]["num_layers"],
    bidirectional=ckpt["config"]["bidirectional"],
)

model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

def sentence_embedding(text, max_length=256):
    tokens = tokenize(text)[: max_length - 2]
    tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]

    ids = encode_tokens(tokens)
    input_ids = torch.tensor(ids).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        emb = model.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode="pooled"
        )
    return emb.squeeze(0)

def similarity_scores(query, sentences):
    q_emb = sentence_embedding(query)

    scores = []
    for s in sentences:
        s_emb = sentence_embedding(s)
        score = F.cosine_similarity(q_emb, s_emb, dim=0).item()
        scores.append((s, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)

query = "declaration of independence"

sentences = [
    "The unanimous Declaration of the thirteen united States of America",
    "Project Gutenberg released its first ebook in 1971",
    "This recipe uses eggs and flour",
]

results = similarity_scores(query, sentences)

for sent, score in results:
    print(f"{score:.4f} | {sent}")
