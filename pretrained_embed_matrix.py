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

if __name__=="__main__":
    pretrained_embedding = create_pretrained_embedding_matrix(
        lstm_vocab_ckpt_path="models/mlm_bilstm/bilstm_mlm_epoch3.pt",
        cbow_model_path="models/word2vec_cbow_mlmvocab.model"
    )