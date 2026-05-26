# RNNs and Transformers from Scratch

Sequence modeling implemented entirely in NumPy.
RNN cells, attention mechanisms, language modeling, 
and sentiment classification — no framework, no autograd.

## What's Implemented

### RNN Cells (with full BPTT)

**VanillaRNN**  
`h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)`  
Backward: gradient through tanh, accumulates dW_xh, 
dW_hh, db_h across timesteps.

**LSTM**  
Forget, input, candidate, output gates with sigmoid 
and tanh activations. Backward routes gradients through 
`dc_total = dh * o * (1 - tanh²(c)) + dc` and 
independently through each gate's activation derivative.

**GRU**  
Reset and update gates. Backward correctly handles 
the `r * h_prev` term in the candidate state by 
routing `dtanh @ W_hh.T * h_prev` through the reset gate.

### Attention

**Scaled Dot-Product Attention**  
`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V`  
Numerically stable via row-wise max subtraction. 
Supports arbitrary masks (1 = mask out).

**Multi-Head Self-Attention**  
Projects input to Q, K, V, splits into `num_heads` 
heads of dimension `d_k = d_model / num_heads`, 
computes attention in parallel across all heads 
via vectorized matmul, concatenates and projects back.

**Sinusoidal Positional Encoding**  
`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`  
`PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

### Models

**RNNLanguageModel**  
Supports vanilla, LSTM, and GRU cells as drop-in 
backends. Includes:
- Temperature sampling for text generation
- Beam search decoding with configurable beam width
- Full BPTT with gradient clipping

**BiLSTMClassifier**  
Bidirectional LSTM (separate forward and backward 
LSTMCell instances) with optional multi-head 
self-attention pooling over the concatenated 
hidden states.

### Training Utilities
- Truncated BPTT with configurable k1 (forward) 
  and k2 (backward) windows
- Gradient clipping by global L2 norm
- LR reduction on plateau
- Perplexity evaluation
- BLEU score (up to 4-gram with brevity penalty)
- Character-level and word-level TextProcessor 
  with PAD, UNK, SOS, EOS tokens

## Tasks

| Task | Model | Dataset |
|---|---|---|
| Text generation | LSTM Language Model | Shakespeare |
| Sentiment classification | BiLSTM + Attention | Movie reviews |

## Run

```bash
# Requires shakespeare.txt and reviews.csv in working directory
python rnn.py
```

Generates text samples every 3 epochs.  
Prints confusion matrix after classifier training.

## Key Design Notes

**GRU reset gate gradient** — the candidate state 
`h_tilde = tanh(W_xh * x + W_hh * (r * h_prev) + b_h)` 
means the gradient through W_hh must be accumulated 
as `(r * h_prev).T @ dtanh`, and the gradient to 
`h_prev` picks up an additional `dtanh @ W_hh.T * r` term.

**LSTM cell state highway** — gradients flow 
through `dc_prev = dc_total * f` with minimal 
attenuation when the forget gate is near 1, 
which is why LSTMs mitigate vanishing gradients 
compared to vanilla RNNs.

**Beam search** — maintains `beam_width` candidate 
sequences ranked by cumulative log-probability. 
At each step expands all beams by `beam_width` 
candidates and retains the top-k by score.
