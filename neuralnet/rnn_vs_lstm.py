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
embeddings = np.random.randn(len(vocab), embedding_dim)

inputs = [embeddings[vocab[w]] for w in sentence]


# RNN parameters
hidden_dim = 4
W_x = np.random.randn(hidden_dim, embedding_dim)   # input → hidden
W_h = np.random.randn(hidden_dim, hidden_dim)      # hidden → hidden
b   = np.zeros((hidden_dim, 1))

# Forward pass
h_prev = np.zeros((hidden_dim, 1))
print("RNN forward pass:\n")

for t, x_t in enumerate(inputs):
    x_t = x_t.reshape(-1,1)
    h_prev = np.tanh(W_x @ x_t + W_h @ h_prev + b)
    print(f"Step {t+1} ({sentence[t]}): hidden = {h_prev.ravel()}")




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# LSTM parameters
hidden_dim = 4
concat_dim = hidden_dim + embedding_dim

W_f = np.random.randn(hidden_dim, concat_dim)
W_i = np.random.randn(hidden_dim, concat_dim)
W_c = np.random.randn(hidden_dim, concat_dim)
W_o = np.random.randn(hidden_dim, concat_dim)

b_f = np.zeros((hidden_dim, 1))
b_i = np.zeros((hidden_dim, 1))
b_c = np.zeros((hidden_dim, 1))
b_o = np.zeros((hidden_dim, 1))

# Forward pass
h_prev = np.zeros((hidden_dim, 1))
c_prev = np.zeros((hidden_dim, 1))

print("\nLSTM forward pass:\n")

for t, x_t in enumerate(inputs):
    x_t = x_t.reshape(-1,1)
    concat = np.vstack((h_prev, x_t))

    f_t = sigmoid(W_f @ concat + b_f)
    i_t = sigmoid(W_i @ concat + b_i)
    c_hat = np.tanh(W_c @ concat + b_c)
    o_t = sigmoid(W_o @ concat + b_o)

    c_prev = f_t * c_prev + i_t * c_hat
    h_prev = o_t * np.tanh(c_prev)

    print(f"Step {t+1} ({sentence[t]}): hidden = {h_prev.ravel()}")
