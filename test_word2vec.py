import numpy as np
from gensim.models import Word2Vec

# Use the same tokenizer and vocab as your training script
def text_vector(tokens, model):
    vecs = []
    weights = []
    for t in tokens:
        if t not in model.wv:
            continue
        freq = model.wv.get_vecattr(t, "count")
        weight = 1.0 / np.log(freq + 2.0)
        vecs.append(model.wv[t])
        weights.append(weight)
    if not vecs:
        return None
    vecs = np.vstack(vecs)
    weights = np.array(weights)
    return np.sum(vecs * weights[:, None], axis=0) / weights.sum()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def similarity_scores_vocab(
    model_path,
    query,
    words,
    min_token_len=2,
):
    model = Word2Vec.load(model_path)
    # Import tokenizer and vocab from your CBOW script (already defined above in your notebook)
    # Use the same 'tokenize' and 'stoi' from previous cells

    q_tokens = [tok for tok in tokenize(query) if tok in stoi and len(tok) >= min_token_len]
    qv = text_vector(q_tokens, model)
    if qv is None:
        print("Query OOV")
        return

    for w in words:
        w_tokens = [tok for tok in tokenize(w) if tok in stoi and len(tok) >= min_token_len]
        wv = text_vector(w_tokens, model)
        if wv is None:
            print(f"{w}: OOV")
            continue
        sim = cosine(qv, wv)
        print(f"{query} ↔ {w} = {sim:.4f}")

if __name__=="__main__":
    # Example usage:
    model_path = "models/word2vec_cbow_mlmvocab.model"
    query_word = "government"
    compare_words = ["nation", "book", "happiness", "machine", "citizen", "judiciary", "democracy"]
    similarity_scores_vocab(
        model_path=model_path,
        query=query_word,
        words=compare_words
    )