import re
import torch
import torch.nn as nn
import torch.nn.functional as F

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
DEVICE = "cpu"

def tokenize(text):
    return TOKEN_RE.findall(text.lower()) if text else []

def encode_tokens(tokens, stoi, unk_id=2):
    return [stoi.get(t, unk_id) for t in tokens]

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

def load_lstm_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab_itos = ckpt["vocab"]
    stoi = {t: i for i, t in enumerate(vocab_itos)}
    config = ckpt["config"]
    model = FlexibleLSTMBase(
        vocab_size=len(vocab_itos),
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    model.to(device)
    return model, vocab_itos, stoi

def batch_sentence_encode(texts, model, stoi, max_length=256, device="cpu"):
    batch_ids = []
    attention_masks = []
    for text in texts:
        tokens = tokenize(text)[: max_length - 2]
        tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
        ids = encode_tokens(tokens, stoi)
        batch_ids.append(torch.tensor(ids, dtype=torch.long))
    max_len = max(len(x) for x in batch_ids)
    padded = torch.full((len(batch_ids), max_len), 0, dtype=torch.long)  # PAD_ID = 0
    attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        l = len(ids)
        padded[i, :l] = ids
        attention_mask[i, :l] = 1
    padded = padded.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        emb = model.encode(
            input_ids=padded,
            attention_mask=attention_mask,
            mode="pooled"
        )
    return emb.cpu()  # (N, emb_dim)

# Usage Example:
if __name__ == "__main__":
    CKPT_PATH = "models/ms_marco_emb/bilstm_best.pt"
    texts = [
        "This is a cat.",
        "The book was on the table.",
        "Artificial intelligence is fascinating."
    ]
    model, vocab_itos, stoi = load_lstm_model(CKPT_PATH)
    batch_emb = batch_sentence_encode(texts, model, stoi)
    print(batch_emb.shape)  # (batch_size, emb_dim)
    print(batch_emb)