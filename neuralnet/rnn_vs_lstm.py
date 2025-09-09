import numpy as np
np.random.seed = 42
sentence = ["this","movie","was","amazing"]
vocab = {
    "this":0,
    "movie":1,
    "was":2,
    "amazing":3
}

embedding_dim = 5
embeddings = np.random.randn(len(vocab))

inputs = [embeddings[vocab[w]] for w in sentence]

