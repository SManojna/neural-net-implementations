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


Optimizer: Momentum (lr=0.17, momentum=0.9)  
Gradient clip norm: 1.0  
LR: warmup 1 epoch → cosine decay to 0.034

## Gradient Checking

Before training, analytical gradients from backprop 
are verified against numerical gradients computed via 
central finite differences across all Dense layers.
Max relative error threshold: 1e-4.

```bash
layer0.W  relative error = 3.2e-07
layer0.b  relative error = 1.1e-07
layer3.W  relative error = 4.8e-07
Gradient check passed.
```

## Run

```bash
python neural_net.py
```

MNIST downloads automatically to `./data`.
Training logs saved to `./runs/<timestamp>/`.

## Key Design Decisions

**Softmax + CrossEntropy fusion** — when CrossEntropyLoss 
is compiled with a Softmax output layer, the backward mode 
is set to `softmax_cross_entropy` so the combined gradient 
`(p - y) / N` passes through directly, avoiding the full 
Jacobian multiply.

**Inverted dropout** — activations scaled by `1 / keep_prob` 
at training time so inference requires no adjustment.

**Cosine LR decay** — `lr = lr_min + 0.5 * (lr_max - lr_min) 
* (1 + cos(π * t))` where t is normalized progress through 
decay phase after warmup.
