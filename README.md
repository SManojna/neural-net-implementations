# Deep Learning from Scratch — Pure NumPy

Deep learning implementations built entirely 
in NumPy without any ML frameworks. Every forward pass, 
backward pass, and optimizer step is hand-coded.

## What's Implemented

### 1-  Neural Network Fundamentals
`mnist-classifier/`

Built a modular neural network engine from scratch:
- Dense, ReLU, Sigmoid, Softmax, Dropout layers with 
  full forward and backward passes
- Gradient checking via central finite differences 
  to verify analytical gradients
- SGD and SGD with Momentum optimizers
- MSE and Cross-Entropy loss with numerically stable 
  implementations
- LR scheduler with linear warmup + cosine decay
- Gradient clipping by global L2 norm
- Dead ReLU tracking across training
- CSV logging of training metrics per batch

Trained on MNIST digit classification.

### 2 — CNN Architectures
`cnn-image-recognition/`

Implemented a full CNN training framework from scratch:
- Conv2D forward via `sliding_window_view` + `tensordot` 
  (no im2col loops)
- Conv2D backward via exact transposed convolution 
  (upsample → pad → contract with flipped kernels)
- MaxPool2D with argmax caching for gradient routing
- BatchNorm2D with training/eval modes and frozen 
  layer support for transfer learning
- Spatial Dropout2D for CNNs
- Adam optimizer with bias correction, weight decay, 
  and global-norm clipping
- LeNet5 and MiniVGG architectures on CIFAR-10
- Transfer learning framework with layer freezing
- Data augmentation (flip, rotation, random crop)
- Guided backpropagation saliency maps
- Per-class accuracy and confusion matrix

### 3 — RNNs and Transformers
`RNN-and-LSTM/`

Sequence modeling from scratch:
- VanillaRNN, LSTM, and GRU cells with full 
  BPTT backward passes
- Multi-head self-attention with scaled 
  dot-product attention
- Sinusoidal positional encoding
- RNN language model with temperature sampling 
  and beam search decoding
- Bidirectional LSTM classifier with attention
- Truncated BPTT
- Perplexity and BLEU score evaluation
- Character and word-level text processors

Trained on Shakespeare text generation and 
movie review sentiment classification.

---

## Why Pure NumPy

Every operation — convolution, attention, batch norm, 
backprop — is implemented using only NumPy array 
operations. No autograd, no framework abstractions. 
This forces a precise understanding of what happens 
at the mathematical level inside modern deep learning.

## Stack

`Python` `NumPy` `Matplotlib`
