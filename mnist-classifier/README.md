# Neural Network Fundamentals from Scratch

A modular neural network engine built entirely in NumPy.
No PyTorch, no autograd — every gradient computed by hand.

## What's Implemented

**Layers**
- Dense (Xavier, He, normal initialization)
- ReLU with dead neuron tracking (per-batch and running average)
- Sigmoid with numerical stability via clipping
- Softmax with fused cross-entropy backward mode
- Dropout with inverted scaling

**Loss Functions**
- Cross-Entropy with numerically stable softmax fusion
- MSE with correct batch normalization

**Optimizers**
- SGD
- SGD with Momentum

**Training Infrastructure**
- LR scheduler: linear warmup followed by cosine decay
- Gradient clipping by global L2 norm
- Gradient checking via central finite differences 
  (numerical vs analytical verification)
- CSV logging of loss, accuracy, LR, gradient norms, 
  dead ReLU percentage per batch
- Timestamped run directories with checkpoints and plots

## Architecture Trained
MNIST (784) → Dense(128, He) → ReLU → Dropout(0.2) → Dense(64, He)  → ReLU → Dropout(0.3) → Dense(10, Xavier) → Softmax
