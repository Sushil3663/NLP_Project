import os
import json
from tokenizers import ByteLevelBPETokenizer

def train_bpe_tokenizer_books(json_path, output_dir, special_tokens, vocab_size=10000, min_frequency=2):
    """
    Train ByteLevelBPETokenizer on book-style JSON with 'anchor' and 'positives' fields.
    Args:
        json_path (str): Path to JSON file [{'book_id', 'anchor', 'positives'}, ...]
        output_dir (str): Directory to save tokenizer
        special_tokens (list): Special tokens, e.g. ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        vocab_size (int): Vocabulary size
        min_frequency (int): Minimum frequency
    """
    output_dir = output_dir+"_"+vocab_size
    with open(json_path, 'r') as f:
        data = json.load(f)
    texts = []
    for item in data:
        if "anchor" in item:
            texts.append(item["anchor"])
        if "positives" in item and isinstance(item["positives"], list):
            texts.extend(item["positives"])
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    pad_id = tokenizer.token_to_id(special_tokens[0])
    print(f"{special_tokens[0]} token ID: {pad_id}")

if __name__=="__main__":
    train_bpe_tokenizer_books(
        json_path="data/contrastive_anchors_positives.json",
        output_dir="./models/BPEtokenizer_books_an_pos",
        special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<CLS>", "<SEP>"]
    )