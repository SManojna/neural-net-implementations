#!/usr/bin/env python3
"""
CS5720 Assignment 1: Neural Network Fundamentals
Student Name: Sai Manojna Velagala

Starter Code - Build a Neural Network from Scratch
"""

import numpy as np
import struct
import gzip
import csv, json, time, math
import matplotlib.pyplot as plt



# ============================================================================
# Data Loading Utilities
# ============================================================================
import os, urllib.request

def load_mnist(path='data/'):
    """
    Load MNIST dataset from files or download if not present.

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    import os
    import urllib.request

    # Create data directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # MNIST file information
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    # Download files if not present
    base_url = 'https://raw.githubusercontent.com/fgnt/mnist/master/'
    for file in files.values():
        filepath = os.path.join(path, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)

    # Load data
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
            return images / 255.0  # Normalize to [0, 1]

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    X_train = load_images(os.path.join(path, files['train_images']))
    y_train = load_labels(os.path.join(path, files['train_labels']))
    X_test = load_images(os.path.join(path, files['test_images']))
    y_test = load_labels(os.path.join(path, files['test_labels']))

    return X_train, y_train, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def gradient_norms(layers) -> tuple[float, float]:
    """Return (global_L2, global_Linf) across all layer grads named dW/db."""
    # global_L2 is the sqrt of sum of squares over all grads (dW and db).
    # global_Linf is the maximum absolute gradient value seen anywhere.
    l2_sq = 0.0
    linf = 0.0
    for layer in layers:
        for name in ("dW", "db"):
            if hasattr(layer, name):
                analytical_gradient = getattr(layer, name)
                if analytical_gradient is None:
                    continue
                analytical_gradient = np.asarray(analytical_gradient)
                if analytical_gradient.size == 0:
                    continue
                # Flatten to a vector so norms are easy to compute
                flat = analytical_gradient.ravel()
                # Accumulate L2^2 so we can sqrt at the end
                l2_sq += float(np.dot(flat, flat))
                #absolute gradient value for Linf
                linf = max(linf, float(np.max(np.abs(flat))))
    return (l2_sq ** 0.5, linf)

def grad_clipping(layers, max_norm: float = None, epsilon = 1e-6):
    # only l2 norm is required
    g_l2, _ = gradient_norms(layers)

    if (max_norm is None or max_norm <= 0) or (not np.isfinite(g_l2)) or (g_l2 <= max_norm):
        return g_l2, 1.0

    # uniform scale to shrink all grads so that new gradient will be between -max_norm and + max_norm
    scale = float(max_norm) / (g_l2 + epsilon)
    for layer in layers:
        for name in ("dW", "db"):
            if hasattr(layer, name):
                g = getattr(layer, name)
                if g is None:
                    continue
                setattr(layer, name, np.asarray(g, dtype=np.float64) * scale)
    # return original l2 norm and scale
    return g_l2, scale

def avg_relu_dead(layers, is_running:bool = False):
    # track what percent of ReLU activations are <= 0 (dead)
    # If is_running is True, use the running average; else use last batch
    vals = []
    for l in layers:
        if hasattr(l, 'dead_percentage_last'):
            vals.append(float(l.dead_percentage_running() if is_running else l.dead_percentage_last()))
    # Return mean of ReLU layers
    return float(np.mean(vals)) if vals else 0.0


class lr_scheduler:
    #LR scheduler with warmup followed by cosine decay.
    # total_steps: steps in the whole training run
    # warmup_steps: initial steps to linearly ramp LR from lr_min to lr_max
    # lr_max ot lr_min: LR bounds
    def __init__(self, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float):
        self.total_steps = int(total_steps)
        self.warmup_steps = int(max(0,warmup_steps)) # making sure warmup steps are non negative
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)

    def __call__(self, step)-> float:
        # warm up phase
        if step < self.warmup_steps:
            #calculating multiplier for interplaotion between lr_max and lr_min
            multiplier = step / max(1,self.warmup_steps)
            # increasing the lr from lr_min
            return self.lr_min+ (multiplier * (self.lr_max - self.lr_min))

        # attributing cosine decay to lr

        # calculating how far we are into the decay process
        # progress = how many steps since warmup ended / total steps available for cosine decay
        progress = (step - self.warmup_steps) / max(1,(self.total_steps - self.warmup_steps))
        # normalizing progress between 0 and 1
        normalized_progress = min(max(progress, 0.), 1.)
        # cos(0) = 1 ( max lr), cos (Pi) = -1 (min lr)
        # cosine_decay = 0.5*(1.0 + cos(pi*normalized progress)
        cosine_decay = 0.5*(1.0 + math.cos(math.pi * normalized_progress))

        # scaling the decay between lr_min and lr_max
        return self.lr_min +  ((self.lr_max - self.lr_min)* cosine_decay)

class logs:
    def __init__(self, root = 'runs'):
        t_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(root, t_stamp)
        self.logs = os.path.join(self.dir, 'logs')
        self.plots = os.path.join(self.dir, 'plots')
        self.checkpoints = os.path.join(self.dir, 'checkpoints')
        os.makedirs(self.logs, exist_ok = True)
        os.makedirs(self.plots, exist_ok = True)
        os.makedirs(self.checkpoints, exist_ok = True)
        self.training_csv = os.path.join(self.logs, 'training.csv')
        self.validation_csv = os.path.join(self.logs, 'validation.csv')

    def save_config(self, config):
        with open(os.path.join(self.logs, 'config.json'), 'w') as f:
            json.dump(config, f)

class csvHandler:
    training_header = ["epoch","step","split","loss","acc","lr","grad_l2","grad_inf","dead_relu"]
    validation_header = ["epoch","split","loss","acc"]

    def __init__(self, training_path, validation_path):
        with open(training_path, 'w', newline="") as f:
            csv.DictWriter(f, csvHandler.training_header).writeheader()
        with open(validation_path, 'w', newline="") as f:
            csv.DictWriter(f, csvHandler.validation_header).writeheader()
        self.training_path, self.validation_path = training_path, validation_path

    def log_training(self, **row):
        with open(self.training_path, 'a', newline="") as f:
            csv.DictWriter(f, csvHandler.training_header).writerow(row)

    def log_validation(self, **row):
        with open(self.validation_path, 'a', newline="") as f:
            csv.DictWriter(f, csvHandler.validation_header).writerow(row)


# ============================================================================
# Layer Implementations
# ============================================================================

class Layer:
    """Base class for all layers."""

    def forward(self, X):
        raise RuntimeError("Layer.forward must be overridden by a subclass.")

    def backward(self, dL_dY):
        raise RuntimeError("Layer.backward must be overridden by a subclass.")

    def get_params(self):
        return {}

    def get_grads(self):
        return {}

    def set_params(self, params):
        pass


class Dense(Layer):
    """
    Fully connected (dense) layer.

    Parameters:
        input_dim: Number of input features
        output_dim: Number of output features
        weight_init: Weight initialization method ('xavier', 'he', 'normal')
    """

    def __init__(self, input_dim, output_dim, weight_init='xavier'):
        self.input_dim = input_dim
        self.output_dim = output_dim

        #  Initialize weights and biases
        # Hint: Use different initialization strategies:
        # - 'xavier': sqrt(2 / (input_dim + output_dim))
        # - 'he': sqrt(2 / input_dim)
        # - 'normal': standard normal * 0.01
        if weight_init == 'xavier': wt = np.sqrt(2/ (input_dim + output_dim))
        elif weight_init == 'he': wt = np.sqrt(2/input_dim)
        else : wt = 0.01

        Z = np.random.randn(input_dim, output_dim) # matrix with mean 0, std 1
        self.W = Z*wt # Shape: (input_dim, output_dim)
        self.b = np.zeros(output_dim)  # Shape: (output_dim,)

        # Storage for backward pass
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        Forward pass: Y = XW + b

        Args:
            X: Input data, shape (batch_size, input_dim)

        Returns:
            Y: Output data, shape (batch_size, output_dim)
        """
        #  Implement forward pass
        # Store X for backward pass
        self.X = X.copy()
        Y = X.dot(self.W) + self.b
        return Y

    def backward(self, dL_dY):
        """
        Backward pass: compute gradients.

        Args:
            dL_dY: Gradient of loss w.r.t. output, shape (batch_size, output_dim)

        Returns:
            dL_dX: Gradient of loss w.r.t. input, shape (batch_size, input_dim)
        """
        #  Compute gradients
        # dL_dW = X.T @ dL_dY
        # dL_db = sum(dL_dY, axis=0)
        # dL_dX = dL_dY @ W.T

        self.dW = np.dot(self.X.T, dL_dY)
        self.db = np.sum(dL_dY, axis=0)
        dL_dX = np.dot(dL_dY, self.W.T)
        return dL_dX

    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def get_grads(self):
        return {'W': self.dW, 'b': self.db}

    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']

# Dropout class

class Dropout(Layer):
    def __init__(self, drop_rate=0.2, seed = 123):
        self.drop_rate = float(drop_rate)
        self.keep = 1.0 - self.drop_rate
        self.training = True
        self.rng = np.random.default_rng(seed)
        self.mask = None


    def set_training(self, is_training:bool=True):
        self.training = bool(is_training)



    def forward(self, X):
        # if training, then return input*mask / (1-dropout_rate)
        if not self.training or self.drop_rate <= 0.0:
            self.mask = None
            return X
        self.mask = (self.rng.random(X.shape) < self.keep).astype(X.dtype)
        return (X * self.mask) / self.keep


    def backward(self, dL_dY):
        '''
        dL_dY: gradient from loss w.r.t. input, shape (batch_size, output_dim)
        returns: gradient with dropout applied
        '''
        if not self.training or self.mask is None:
            return dL_dY
        return (dL_dY * self.mask) / self.keep




# ============================================================================
# Activation Functions
# ============================================================================

class Activation(Layer):
    """Base class for activation functions."""

    def __init__(self):
        self.cache = None


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    def __init__(self):
        super().__init__()
        self._last_total = 0
        self._last_dead = 0
        self._cum_total = 0
        self._cum_dead = 0

    def forward(self, X):
        """
        Forward pass: f(x) = max(0, x)

        Args:
            X: Input data

        Returns:
            Output after applying ReLU
        """
        #  Implement ReLU forward pass
        # Store input for backward pass
        self.cache = X
        self._last_total = X.size
        self._last_dead = int(np.count_nonzero(X<=0))
        self._cum_total += self._last_total
        self._cum_dead += self._last_dead
        return np.maximum(0, X)

    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = 1 if x > 0 else 0

        Args:
            dL_dY: Gradient of loss w.r.t. output

        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        #  Implement ReLU backward pass
        dL_dX = np.multiply(dL_dY, self.cache>0)
        return dL_dX

    def dead_percentage_last(self):
        return 0.0 if self._last_total == 0 else 100.0* self._last_dead / self._last_total

    def dead_percentage_running(self):
        return 0.0 if self._cum_total == 0 else 100.0 * self._cum_dead / self._cum_total

class Sigmoid(Activation):
    """Sigmoid activation function."""

    def forward(self, X):
        """
        Forward pass: f(x) = 1 / (1 + exp(-x))

        Args:
            X: Input data

        Returns:
            Output after applying sigmoid
        """
        #  Implement sigmoid forward pass
        # Store output for backward pass
        Clipped_X = np.clip(X, -200, 200)
        f_x = 1/ (1 + np.exp(-Clipped_X))
        self.cache = f_x
        return f_x

    def backward(self, dL_dY):
        """
        Backward pass: f'(x) = f(x) * (1 - f(x))

        Args:
            dL_dY: Gradient of loss w.r.t. output

        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        #  Implement sigmoid backward pass
        output_f_x = self.cache
        d_f_x = output_f_x*(1-output_f_x)
        dL_dX = np.multiply(dL_dY,d_f_x)
        return dL_dX


class Softmax(Activation):
    """Softmax activation function."""

    def __init__(self, backward_mode="auto"):
        super().__init__()
        # "softmax_cross_entropy" and "softmax_and_cross_entropy" mean:
        #   we are using the fused gradient (dL/dz = p - y), so backward just returns dL/dY.
        # "probability" means: do full Jacobian multiply when loss is not CE.
        # "auto" is a fallback (treated like non-fused).
        self.backward_mode = backward_mode

    def set_backward_mode(self, mode: str):
        # accept only known modes
        valid = {"auto", "softmax_cross_entropy", "probability", "softmax_and_cross_entropy"}
        self.backward_mode = mode if mode in valid else "auto"

    def forward(self, X):
        """
        Forward pass: f(x_i) = exp(x_i) / sum(exp(x))

        Args:
            X: Input data, shape (batch_size, num_classes)

        Returns:
            Output probabilities, shape (batch_size, num_classes)
        """
        #  Implement softmax forward pass
        #subtract row-wise max before exp
        row_max = np.max(X, axis=1, keepdims = True)
        # Hint: Subtract max for numerical stability
        Z = (X - row_max).astype(np.float64)
        z_exp = np.exp(Z)
        f_x_i = z_exp/np.sum(z_exp, axis=1, keepdims = True)
        # Cache probabilities for backward() when not fused
        self.cache = f_x_i
        return f_x_i

    def backward(self, dL_dY):
        """
        Backward pass for softmax.

        Args:
            dL_dY: Gradient of loss w.r.t. output

        Returns:
            dL_dX: Gradient of loss w.r.t. input
        """
        #  Implement softmax backward pass
        mode = self.backward_mode
        # Fused path with cross-entropy: upstream gradient is already (p - y)/N
        # so we simply pass it through. No explicit Jacobian multiply needed.
        if mode in ("softmax_cross_entropy", "softmax_and_cross_entropy"):
            return dL_dY
        # Full Jacobian path (for non-fused setups)
        # Non-fused path: multiply by the softmax Jacobian J.
        # For each row: dL/dz = J^T * dL/dp, and J = diag(p) - p p^T.
        # Efficient row-wise formula: dL/dz = p * (dL/dp - <dL/dp, p>)
        f_x_i = self.cache
        row_dot = np.sum(dL_dY * f_x_i, axis=1, keepdims=True)
        return f_x_i * (dL_dY - row_dot)


# ============================================================================
# Loss Functions
# ============================================================================

class Loss:
    """Base class for loss functions."""

    def compute(self, y_pred, y_true):
        raise RuntimeError("Loss.compute must be overridden by a subclass.")

    def gradient(self, y_pred, y_true):
        raise RuntimeError("Loss.gradient must be overridden by a subclass.")


class MSELoss(Loss):
    """Mean Squared Error loss."""

    def compute(self, y_pred, y_true):
        """
        Compute MSE loss: L = 0.5 * mean((y_pred - y_true)^2)

        Args:
            y_pred: Predictions, shape (batch_size, num_features)
            y_true: True values, shape (batch_size, num_features)

        Returns:
            Scalar loss value
        """
        #  Implement MSE loss
        loss = 0.5 * np.mean((y_pred - y_true)**2)
        return loss

    def gradient(self, y_pred, y_true):
        """
        Compute gradient of MSE loss.

        Args:
            y_pred: Predictions
            y_true: True values

        Returns:
            Gradient w.r.t. predictions
        """
        #  Implement MSE gradient
        # dL/dy_pred = (y_pred - y_true) / batch_size
        batch_size = y_pred.shape[0]
        gradient = (y_pred - y_true)/ batch_size
        return gradient


class CrossEntropyLoss(Loss):
    """Cross-entropy loss for classification."""
    def __init__(self, mode = "softmax_and_cross_entropy"):
        self.epsilon = float(1e-12)
        self.mode = mode

    def compute(self, y_pred, y_true):
        """
        Compute cross-entropy loss: L = -mean(sum(y_true * log(y_pred)))

        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: True labels (one-hot), shape (batch_size, num_classes)

        Returns:
            Scalar loss value
        """
        #  Implement cross-entropy loss
        # Add small epsilon to prevent log(0)

        # clipping y_pred for numerical stability
        y_pred = np.clip(y_pred.astype(np.float64), self.epsilon, 1 - self.epsilon)
        y_true = y_true.astype(np.float64)
        # sum of y_true * log(y_pred) and mean of this full sum
        cross_entropy_loss = -np.mean(np.sum(y_true*np.log(y_pred), axis = 1))
        return float(cross_entropy_loss)

    def gradient(self, y_pred, y_true):
        """
        Compute gradient of cross-entropy loss.

        Args:
            y_pred: Predicted probabilities
            y_true: True labels (one-hot)

        Returns:
            Gradient w.r.t. predictions
        """
        #  Implement cross-entropy gradient
        # For softmax + cross-entropy: gradient = (y_pred - y_true) / batch_size
        y = y_true.astype(np.float64)
        N = y.shape[0]
        return (y_pred - y_true) / N


# ============================================================================
# Optimizers
# ============================================================================

class Optimizer:
    """Base class for optimizers."""

    def update(self, params, grads):
        raise RuntimeError("Optimizer.Update must be overridden by a subclass.")


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        """
        Update parameters using vanilla SGD.

        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        #  Implement SGD update rule
        # params = params - learning_rate * grads
        for i in params:
            # type casting to float
            params[i] = params[i].astype(np.float64)
            # applying SGD update rule
            params[i] -= self.lr * grads[i]


class Momentum(Optimizer):
    """SGD with momentum optimizer."""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        # to store volcity of each param
        self.velocity = {}

    def update(self, params, grads):
        """
        Update parameters using SGD with momentum.

        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        for name, p in params.items():
            gradient = grads.get(name, None)

            if gradient is None:
                continue

            # converting param and gradient to float
            if not np.issubdtype(p.dtype, np.floating):
                p = p.astype(np.float64)
            if not np.issubdtype(gradient.dtype, np.floating):
                gradient = gradient.astype(p.dtype)

            # get previous velocity or initialize to zero if prev is not available
            v = self.velocity.get(name, None)
            if (v is None) or (v.shape != p.shape) or (not np.issubdtype(v.dtype, np.floating)):
                v = np.zeros_like(p, dtype=p.dtype)

            # calculating new_velocity
            # new_velocity = momentun * old_velocity - lr * gradient
            v = self.momentum * v - self.lr * gradient

            # updating params
            p = p + v

            # saving to original velocity and param
            self.velocity[name] = v
            params[name] = p





# ============================================================================
# Neural Network Class
# ============================================================================

class NeuralNetwork:
    """
    Modular neural network implementation.

    Example usage:
        model = NeuralNetwork()
        model.add(Dense(784, 128))
        model.add(ReLU())
        model.add(Dense(128, 10))
        model.add(Softmax())
        model.compile(loss=CrossEntropyLoss(), optimizer=SGD(0.01))
        model.fit(X_train, y_train, epochs=10, batch_size=32)
    """

    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """Configure the model for training."""
        self.loss_fn = loss
        self.optimizer = optimizer
        for layer in self.layers:
            if isinstance(layer, Softmax):
                if isinstance(loss, CrossEntropyLoss):
                    layer.set_backward_mode("softmax_cross_entropy")
                else:
                    layer.set_backward_mode("probability")


    def forward(self, X):
        """
        Forward propagation through all layers.

        Args:
            X: Input data

        Returns:
            Output of the network
        """
        #  Implement forward pass through all layers
        y_pred = X
        for l in self.layers:
            # passing the output sequentially through each layer
            y_pred = l.forward(y_pred)
        return y_pred

    def backward(self, dL_dY):
        """
        Backward propagation through all layers.

        Args:
            dL_dY: Gradient of loss w.r.t. network output
        """
        #  Implement backward pass through all layers in reverse order
        gradient = dL_dY
        for l in reversed(self.layers):
            # propagating gradients backward
            gradient = l.backward(gradient)
        return gradient

    def update_params(self):
        """Update parameters of all trainable layers using the optimizer."""
        #  Collect parameters and gradients from all layers
        # Use optimizer to update parameters
        for i,l in enumerate(self.layers):
            if not (hasattr(l, 'get_params') and (hasattr(l,'get_grads'))):
                # skip layers without params and grads
                continue
            params = l.get_params()
            if params == {}:
                # skipping empty params
                continue
            gradients = l.get_grads()
            self.optimizer.update(params, gradients)
            # writing the updated params
            l.set_params(params)



    def fit(self, X_train, y_train, epochs, batch_size,
            X_val=None, y_val=None, verbose=True, grad_clip = None):
        """
        Train the neural network.

        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress

        Returns:
            Dictionary containing training history
        """
        history = {'train_loss': [], 'train_acc': [],
                   'val_loss': [], 'val_acc': [],
                   'grad_l2_epoch_mean_pre': [],
                   'grad_l2_epoch_mean_post': [],
                   'grad_inf_epoch_max_pre': []
                   }

        n_samples = X_train.shape[0]
        steps_per_epoch = int(np.ceil(n_samples / batch_size))
        total_steps = int(epochs) * steps_per_epoch
        warmup_steps = steps_per_epoch
        rng = np.random.default_rng(42)

        run = logs(root = 'runs')
        base_lr = float(getattr(self.optimizer, 'lr', 0.01))

        # setting configurations for reusing
        config = dict(
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr_max=float(base_lr),
            lr_min=float(max(base_lr * 0.2, 0.01)),
            warmup_epochs=1,
            optimizer=self.optimizer.__class__.__name__
        )

        run.save_config(config)

        logger = csvHandler(run.training_csv, run.validation_csv)

        # scheduling the learning rate to adapt to the data
        scheduled_lr = lr_scheduler(total_steps = total_steps,warmup_steps = warmup_steps,
                                    lr_max=config['lr_max'], lr_min = config['lr_min'])

        global_step = 0

        for epoch in range(epochs):
            #  Implement training loop
            # 1. Shuffle training data
            idx = rng.permutation(n_samples)
            X_tr, y_tr = X_train[idx], y_train[idx]

            l2_sum_pre, l2_sum_post, inf_max_pre, nbatches = 0.0, 0.0, 0.0, 0
            batch_loss, batch_acc = [], []

            # 2. Process mini-batches
            for b in range(steps_per_epoch):
                start = b * batch_size
                end = min(start+batch_size, n_samples)
                if start>= end:
                    break
                X_batch, y_batch = X_tr[start:end], y_tr[start:end]

                # update the learning rate from the scheduler
                self.optimizer.lr = float(scheduled_lr(global_step))

                # 3. Forward pass
                y_pred = self.forward(X_batch)

                # 4. Compute loss
                loss = self.loss_fn.compute(y_pred, y_batch)
                batch_loss.append(loss)

                accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(y_batch, 1)).astype(float)
                batch_acc.append(accuracy)

                # 5. Backward pass
                gradient = self.loss_fn.gradient(y_pred, y_batch)
                self.backward(gradient)
                # grad befor clip
                g_l2_pre, g_inf_pre = gradient_norms(self.layers)

                # clip grads by global L2 norm
                if grad_clip is not None and grad_clip > 0:
                    _, _ = grad_clipping(self.layers, max_norm=grad_clip)

                # gradients after clipping
                g_l2_post, _ = gradient_norms(self.layers)

                l2_sum_pre += g_l2_pre
                l2_sum_post += g_l2_post
                inf_max_pre = max(inf_max_pre, g_inf_pre)
                nbatches += 1


                 # 6. Update parameters
                self.update_params()
                global_step += 1

            # 7. Track metrics
            epoch_loss = float(np.mean(batch_loss)) if batch_loss else float('nan')
            epoch_acc = float(np.mean(batch_acc)) if batch_acc else float('nan')
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            # Evaluating on validation data
            if X_val is not None and y_val is not None:
                self._toggle_eval_mode(True)  # turns off dropout
                try:
                    v_pred = self.forward(X_val)
                finally:
                    self._toggle_eval_mode(False)
                v_loss = float(self.loss_fn.compute(v_pred, y_val))
                v_acc = float((np.argmax(v_pred, 1) == np.argmax(y_val, 1)).mean())
                history['val_loss'].append(v_loss)
                history['val_acc'].append(v_acc)
            else:
                history['val_loss'].append(None)
                history['val_acc'].append(None)

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - loss {epoch_loss:.4f} - acc {epoch_acc:.4f}"
                if history['val_loss'][-1] is not None:
                    msg += f" - val_loss {history['val_loss'][-1]:.4f} - val_acc {history['val_acc'][-1]:.4f}"
                msg += f" - lr {float(self.optimizer.lr):.6f}"
                print(msg)

            if nbatches > 0:
                history['grad_l2_epoch_mean_pre'].append(float(l2_sum_pre / nbatches))
                history['grad_l2_epoch_mean_post'].append(float(l2_sum_post / nbatches))
                history['grad_inf_epoch_max_pre'].append(float(inf_max_pre))
            else:
                history['grad_l2_epoch_mean_pre'].append(float('nan'))
                history['grad_l2_epoch_mean_post'].append(float('nan'))
                history['grad_inf_epoch_max_pre'].append(float('nan'))

        return history

    def predict(self, X):
        """
        Make predictions on input data.

        Args:
            X: Input data

        Returns:
            Predictions (class indices for classification)
        """
        #  Forward pass and return predictions
        self._toggle_eval_mode(True)
        out = np.argmax(self.forward(X), axis=1)
        self._toggle_eval_mode(False)
        return out

    def _toggle_eval_mode(self, is_eval: bool):
        # enable and disable for layers that behave differently during training/ evaluation
        for layer in self.layers:
            if hasattr(layer, "set_training"):
                layer.set_training(not is_eval)

    def evaluate(self, X, y):
        """
        Evaluate model performance.

        Args:
            X: Input data
            y: True labels

        Returns:
            loss, accuracy
        """
        #  Compute loss and accuracy
        self._toggle_eval_mode(True)
        y_pred = self.forward(X)
        loss = self.loss_fn.compute(y_pred, y)
        true_label = np.argmax(y, axis=1)
        pred_label = np.argmax(y_pred, axis=1)
        accuracy = float((true_label == pred_label).mean())
        self._toggle_eval_mode(False)
        return float(loss), accuracy

    def save_weights(self, filename):
        """Save model weights to file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                weights[f'layer_{i}'] = layer.get_params()
        np.savez(filename, **weights)

    def load_weights(self, filename):
        """Load model weights from file."""
        weights = np.load(filename)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_params') and f'layer_{i}' in weights:
                layer.set_params(weights[f'layer_{i}'])


# ============================================================================
# Gradient Checking
# ============================================================================

def gradient_check(model, X, y, epsilon=1e-5):
    """
    Verify gradients using finite differences.

    This function:
    1. Temporarily disables stochastic layers like Dropout (for deterministic output).
    2. Runs forward & backward pass to compute analytical gradients.
    3. Approximates numerical gradients by perturbing parameters with ±epsilon.
    4. Compares analytical vs numerical gradients and reports worst relative error.

    Args:
        model: NeuralNetwork instance
        X: Input mini-batch (numpy array)
        y: One-hot encoded labels for the mini-batch
        epsilon: Small finite difference step size

    Returns:
        Dictionary mapping "layerN.param" -> relative error (float)
    """
    dropout_layers = []
    for layer in model.layers:
        if hasattr(layer, "set_training"):
            dropout_layers.append(layer)
            layer.set_training(False)

    # Ensure float64 precision to reduce numerical noise
    X, y = X.astype(np.float64), y.astype(np.float64)
    maxchecks = 100  # limit number of indices checked per param tensor

    # 1. Forward pass
    y_pred = model.forward(X)

    # 2. Compute analytical gradients using backprop
    gradient = model.loss_fn.gradient(y_pred, y)
    model.backward(gradient)

    scaling_factor = y.shape[1] if isinstance(model.loss_fn, MSELoss) else 1
    errors = {}

    # 3. Loop through all layers with parameters
    for i, l in enumerate(model.layers):
        if not (hasattr(l, "get_params") and hasattr(l, "get_grads")):
            continue

        params, grads = l.get_params(), l.get_grads()
        if not params:
            continue

        for param_name, p in params.items():
            ana_gradient = grads[param_name]
            k = int(min(maxchecks, p.size))
            idx_list = np.linspace(0, p.size - 1, k).astype(int)
            worst_err = 0.0

            for idx in idx_list:
                idx = np.unravel_index(idx, p.shape)
                orig_val = p[idx]

                # Compute loss with param + epsilon
                p[idx] = orig_val + epsilon
                loss_plus = model.loss_fn.compute(model.forward(X), y)

                # Compute loss with param - epsilon
                p[idx] = orig_val - epsilon
                loss_minus = model.loss_fn.compute(model.forward(X), y)

                # Reset parameter to original
                p[idx] = orig_val

                # Numerical gradient via central difference
                numeric_gradient = (loss_plus - loss_minus) / (2 * epsilon)

                # Analytical gradient at this index
                analytic_gradient = ana_gradient[idx] / scaling_factor

                # Relative error
                rel_error = abs(analytic_gradient - numeric_gradient) / (
                    abs(analytic_gradient) + abs(numeric_gradient) + epsilon
                )
                worst_err = max(worst_err, rel_error)

            errors[f"layer{i}.{param_name}"] = float(worst_err)

    # --- Re-enable Dropout layers back to training mode ---
    for layer in dropout_layers:
        layer.set_training(True)

    return errors


def plot_gradient_history(history,show_best=True, savepath=None):
    def arr(key):
        x = history.get(key, [])
        return np.array([np.nan if v is None else float(v) for v in x], dtype=float)

    tl = arr('train_loss');  vl = arr('val_loss')
    ta = arr('train_acc');   va = arr('val_acc')
    gpre  = arr('grad_l2_epoch_mean_pre')
    gpost = arr('grad_l2_epoch_mean_post')
    epochs = np.arange(1, len(tl) + 1)

    fig, ax1 = plt.subplots(figsize=(9,5))
    ax2 = ax1.twinx()

    # Left axis: losses + gradients (
    ax1.plot(epochs, tl,  lw=2.0, marker='o', label="train loss")
    if np.isfinite(vl).any():
        ax1.plot(epochs, vl,  lw=2.0, ls='--', marker='s', label="val loss")
    if gpre.size and np.isfinite(gpre).any():
        ax1.plot(epochs, gpre,  lw=2.0, ls=':',  marker='^', alpha=0.9, label="grad L2 (pre)")
    if gpost.size and np.isfinite(gpost).any():
        ax1.plot(epochs, gpost, lw=2.0, ls='-.', marker='v', alpha=0.9, label="grad L2 (post)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / Gradient L2")

    # Right axis: accuracies
    ax2.plot(epochs, ta, lw=2.0, marker='D', label="train acc")
    if np.isfinite(va).any():
        ax2.plot(epochs, va, lw=2.0, ls='--', marker='x', label="val acc")
    ax2.set_ylabel("Accuracy")

    # Mark best val-loss epoch
    if show_best and np.isfinite(vl).any():
        best_e = int(np.nanargmin(vl) + 1)
        ax1.axvline(best_e, ls='--', alpha=0.4)
        ax1.text(best_e + 0.1, ax1.get_ylim()[1]*0.97, f"{best_e}",
                 fontsize=9, alpha=0.7)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2)

    ax1.grid(True, alpha=0.15)
    plt.title("gradients vs accuracy ")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.17

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Convert labels to one-hot encoding
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)

    # Create model
    print("Building neural network...")

    model = NeuralNetwork()

    #  Build the network architecture
    # Input (784) → Dense (128) → ReLU → Dense (64) → ReLU → Dense (10) → Softmax
    model.add(Dense(784,128, "he"))
    model.add(ReLU())
    model.add(Dropout(0.2))
    model.add(Dense(128,64, "he"))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Dense(64,10, "xavier"))
    model.add(Softmax())

    #  Compile model with CrossEntropyLoss and SGD optimizer
    model.compile(loss = CrossEntropyLoss(), optimizer=SGD(learning_rate=LR))

    # gradient check
    print("performing gradient check...")
    X_check = X_train[:4]
    y_check = y_train[:4]
    y_check_oh = one_hot_encode(y_check)
    # Make sure Softmax is in fused backward mode
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Softmax):
            print("Softmax mode:", layer.backward_mode)  # should print softmax_cross_entropy

    errors = gradient_check(model, X_check, y_check_oh)
    for name, err in errors.items():
        print(f"{name:20s} relative error = {err:.2e}")

    max_err = max(errors.values()) if errors else 0.0
    if max_err > 1e-4:
        print(f"Warning: Large gradient check error detected (max {max_err:.2e}).")
        print("Please debug backprop before training.")
    else:
        print(f"Gradient check passed (max error {max_err:.2e}).")

    #  compiling the model
    print("Compiling the model.....")
    model.compile(loss = CrossEntropyLoss(), optimizer=Momentum(learning_rate=LR, momentum=0.9))


    n = X_train.shape[0]
    val_size = 10000
    X_val, y_val = X_train[-val_size:], y_train_oh[-val_size:]
    X_trn, y_trn = X_train[:-val_size], y_train_oh[:-val_size]
    print("Training model...")
    # history = model.fit(...)
    history = model.fit(X_trn, y_trn, batch_size=BATCH_SIZE, epochs=EPOCHS, grad_clip=1.0,X_val=X_val, y_val=y_val)
    plot_gradient_history(history)
    #  Evaluate on test set
    print("Evaluating model...")
    # test_loss, test_acc = model.evaluate(X_test, y_test_oh)
    test_loss, test_acc = model.evaluate(X_test, y_test_oh)
    print("test acc:", test_acc, "test loss:", test_loss)

    #  Save model weights
    model.save_weights('model_weights.npz')

    #  Save sample predictions
    predictions = model.predict(X_test[:100])
    np.savetxt('predictions_sample.txt', predictions, fmt='%d')



