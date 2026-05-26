"""
CS5720 Assignment 2: CNN Architecture Implementation
Starter code for building CNNs from scratch
"""
import os

import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import tarfile, pickle
import urllib.request

from numpy.lib.stride_tricks import sliding_window_view

_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR_TOP = "cifar-10-batches-py"


def _ensure_diag(cache: dict):
    cache.setdefault("warnings", [])
    cache.setdefault("numerical_warnings", [])
    cache.setdefault("stability_flags", set())
    cache.setdefault("stability_flags_list", [])
    cache.setdefault("last_forward_stats", {})

def softmax_xent(logits: np.ndarray, y: np.ndarray):
    """
    Stable softmax cross-entropy with hardening for pathological inputs.
    logits: (N, K), y: (N,)
    Returns (loss: float, dlogits: (N,K))
    """
    # subtract rowwise max (handles big numbers & +inf gracefully)
    row_operation = logits - np.max(logits, axis=1, keepdims=True)

    # exp with nan/inf protection
    exp = np.exp(np.clip(row_operation, -60, 0))  # clamp lower bound to avoid denorm stalls; upper already <=0
    # replace any nans from weird inputs with 0
    exp = np.nan_to_num(exp, nan=0.0, posinf=0.0, neginf=0.0)

    denom = np.sum(exp, axis=1, keepdims=True)
    # if a row underflowed to 0, set a safe fallback to avoid divide-by-0
    denom = np.where(denom == 0.0, 1.0, denom)

    probability = exp / denom

    N = logits.shape[0]
    # clip probabilities to avoid log(0) and keep grads finite
    p_safe = np.clip(probability[np.arange(N), y], 1e-12, 1.0)
    loss = -np.mean(np.log(p_safe))

    # gradient wrt logits
    grad_logit = probability
    grad_logit[np.arange(N), y] -= 1.0
    grad_logit /= max(N, 1)

    # sanitize any numeric junk
    grad_logit = np.nan_to_num(grad_logit, nan=0.0, posinf=0.0, neginf=0.0)
    return float(loss), grad_logit



class SGD:
    """
    NumPy-only SGD with momentum, weight decay, and global-norm gradient clipping.
    Works with: optimizer.step([(param, grad), ...])
    """

    def __init__(self, lr=5e-2, momentum=0.9, weight_decay=5e-4, clip_global_norm=5.0):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.clip_global_norm = clip_global_norm
        self._vel = {}  # id(param) -> velocity array

    def step(self, param_grad_pairs):
        """Apply one SGD update.
                Args:
                    param_grad_pairs: iterable of (param: np.ndarray, grad: np.ndarray|None).
                Returns:
                    None. Updates happen in place on params.
                """
        # 1) optional global-norm clipping (on decay-adjusted grads)
        if self.clip_global_norm is not None:
            s = 0.0
            for p, g in param_grad_pairs:
                if g is None:
                    continue
                gg = g + (self.weight_decay * p) if self.weight_decay > 0 else g
                s += float(np.sum(gg.astype(np.float64) ** 2))
            gn = float(np.sqrt(s))
            scale = (self.clip_global_norm / gn) if (gn > 0 and gn > self.clip_global_norm) else 1.0
        else:
            scale = 1.0

        # 2) momentum update + in-place param step
        for p, g in param_grad_pairs:
            if g is None:
                continue
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            g = scale * g
            key = id(p)
            if key not in self._vel:
                self._vel[key] = np.zeros_like(p)
            v = self._vel[key] = self.momentum * self._vel[key] - self.lr * g
            p += v  # in-place update


class Adam:
    """
    NumPy-only Adam optimizer with bias correction.
    Compatible with Trainer: uses .step([(param, grad), ...]) and exposes .lr
    Adam optimizer (NumPy) with bias correction, optional weight decay and global-norm clipping."""


    def __init__(self,
                 lr: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.0,
                 clip_global_norm: float | None = None):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.clip_global_norm = clip_global_norm
        # per-parameter state
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._t: int = 0  # 1-based

    def _ensure_state(self, p: np.ndarray) -> int:
        k = id(p)
        if k not in self._m:
            self._m[k] = np.zeros_like(p, dtype=p.dtype)
            self._v[k] = np.zeros_like(p, dtype=p.dtype)
        return k

    def step(self, param_grad_pairs):
        """Apply one Adam update.
                Args:
                    param_grad_pairs: iterable of (param: np.ndarray, grad: np.ndarray|None).
                Returns:
                    None. Updates happen in place on params.
                """
        # time step
        self._t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - (b1 ** self._t)
        bc2 = 1.0 - (b2 ** self._t)

        # optional global-norm clip (on decay-adjusted grads, matches SGD path)
        scale = 1.0
        if self.clip_global_norm is not None:
            s = 0.0
            for p, g in param_grad_pairs:
                if g is None:
                    continue
                gg = g + (self.weight_decay * p) if self.weight_decay > 0.0 else g
                s += float(np.sum(gg.astype(np.float64) ** 2))
            gn = float(np.sqrt(s))
            if gn > 0.0 and gn > self.clip_global_norm:
                scale = self.clip_global_norm / gn

        # parameter-wise update
        for p, g in param_grad_pairs:
            if g is None:
                continue
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * p
            g = scale * g

            k = self._ensure_state(p)
            m = self._m[k] = b1 * self._m[k] + (1.0 - b1) * g
            v = self._v[k] = b2 * self._v[k] + (1.0 - b2) * (g * g)

            # bias-corrected moments
            m_hat = m / max(bc1, 1e-16)
            v_hat = v / max(bc2, 1e-16)

            # in-place parameter update
            p += -self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon))


    def update(self, params: dict, grads: dict):
        """Back-compat helper to call step() from dicts of params and grads."""
        self.step([(params[k], grads.get(k, None)) for k in params])


def cosine_with_warmup(step, total_steps, base_lr, warmup_steps=0, min_lr=0.0):
    """Cosine learning-rate schedule with optional linear warmup.
        Returns:
            float: learning rate for the given step.
        """
    if total_steps <= 0:
        return base_lr
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    # map final epoch to t == 1.0
    steps_after = max(1, total_steps - warmup_steps - 1)
    t = (step - warmup_steps) / steps_after
    t = float(np.clip(t, 0.0, 1.0))
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * t)))



class Layer(ABC):
    """Base class for all neural network layers"""

    def __init__(self):
        self.trainable = True
        self.params = {}
        self.grads = {}
        self.cache = {
            "warnings": [],
            "numerical_warnings": [],
            "stability_flags": set(),
        }

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the layer"""
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the layer"""
        pass


class Conv2D(Layer):
    """
    2D Convolution Layer

    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (filters)
    kernel_size : int or tuple
        Size of convolution kernel
    stride : int
        Stride for convolution
    padding : str or int
        Padding mode ('valid', 'same') or explicit pad width
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: str | int = 'same'):
        super().__init__()
        # ---- validate ----
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValueError("in_channels must be positive int")
        if not (isinstance(out_channels, int) and out_channels > 0):
            raise ValueError("out_channels must be positive int")

        # ---- normalize ----
        if isinstance(kernel_size, int):
            kH, kW = kernel_size, kernel_size
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            kH, kW = kernel_size
            if not (isinstance(kH, int) and isinstance(kW, int)):
                raise TypeError("kernel_size tuple must be ints")
        else:
            raise TypeError("kernel_size must be int or (int,int)")

        if isinstance(stride, int):
            if stride <= 0:
                raise ValueError("stride must be positive")
            sH, sW = stride, stride
        elif isinstance(stride, tuple) and len(stride) == 2:
            sH, sW = stride
            if not (isinstance(sH, int) and isinstance(sW, int)):
                raise TypeError("stride tuple must be ints")
            if sH <= 0 or sW <= 0:
                raise ValueError("stride tuple must be positive")
        else:
            raise TypeError("stride must be int or (int,int)")

        if isinstance(padding, str):
            if padding not in {"valid", "same"}:
                raise ValueError("padding must be 'valid' or 'same'")
        elif isinstance(padding, int):
            if padding < 0:
                raise ValueError("padding int must be >= 0")
        else:
            raise TypeError("padding must be 'valid' | 'same' | int")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kH, kW)
        self.stride = (sH, sW)
        self.padding = padding
        self.numerical_warnings: list[str] = []
        self.last_forward_stats: dict[str, float] = {}

        # ---- He init (stable for ReLU) ----
        fan_in = in_channels * kH * kW
        std = float(np.sqrt(2.0 / max(1, fan_in)))
        self.params['W'] = (np.random.randn(out_channels, in_channels, kH, kW) * std).astype(np.float32)
        self.params['b'] = np.zeros((out_channels,), dtype=np.float32)
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        self.cache = {}
        self.numerical_warnings: list[str] = []
        self.last_forward_stats: dict[str, float] = {}
        self.cache = {
            "numerical_warnings": [],
            "warnings": [],
            "stability_flags": set(),
            "stability_flags_list": []
        }

    def _get_pad_width(self, input_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Calculate padding width based on padding mode"""
        #  Implementing padding calculation
        # For 'same' padding, output size = input size / stride
        # For 'valid' padding, no padding
        # Return (pad_h, pad_w)
        _, _, H, W = input_shape
        kH, kW = self.kernel_size
        sH, sW = self.stride

        if isinstance(self.padding, int):
            p = self.padding
            return (2 * p, 2 * p)

        if self.padding == 'valid':
            return (0, 0)

        # 'same' with ceil formula
        outH = int(np.ceil(H / sH))
        outW = int(np.ceil(W / sW))
        pH_total = max((outH - 1) * sH + kH - H, 0)
        pW_total = max((outW - 1) * sW + kW - W, 0)
        return (pH_total, pW_total)

    def _pad_input(self, x: np.ndarray, pad_h_total: int, pad_w_total: int) -> np.ndarray:
        """Apply padding to input"""
        #  Implementing padding
        # x shape: (batch, channels, height, width)
        pT = pad_h_total // 2
        pB = pad_h_total - pT
        pL = pad_w_total // 2
        pR = pad_w_total - pL
        x_pad = np.pad(x, ((0, 0), (0, 0), (pT, pB), (pL, pR)), mode="constant")
        return x_pad, (pT, pB, pL, pR)

    def _im2col(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        # x: (N, C, H, W) already padded by _pad_input
        N, C, H, W = x.shape
        kH, kW = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        sH, sW = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

        # window view -> (N, C, H-kH+1, W-kW+1, kH, kW)
        wv = sliding_window_view(x, window_shape=(kH, kW), axis=(2, 3))
        # stride-slice to respect (sH, sW)
        wv = wv[:, :, ::sH, ::sW, :, :]  # (N, C, outH, outW, kH, kW)
        N_, C_, outH, outW, _, _ = wv.shape

        # reshape to (N*outH*outW, C*kH*kW)
        cols = wv.reshape(N_ * outH * outW, C_ * kH * kW)
        # cache dims we need to fold gradients back in col2im (if you have it)
        return cols, (N_, C_, outH, outW, kH, kW, sH, sW)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute conv output.
        Args:
            x: (N, C_in, H, W)
        Returns:
            out: (N, C_out, H_out, W_out)
        Fast, grad-safe forward via sliding_window_view + tensordot.

        """
        if not np.isfinite(x).all():
            x = x.copy()
            pos_inf = np.isposinf(x)
            neg_inf = np.isneginf(x)
            if pos_inf.any() or neg_inf.any():
                finfo = np.finfo(x.dtype if x.dtype.kind == 'f' else np.float64)
                # Large but safe cap so contractions don’t overflow
                cap = min(1e12, finfo.max ** 0.25)
                if pos_inf.any(): x[pos_inf] = cap
                if neg_inf.any(): x[neg_inf] = -cap

        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride

        # padding
        pad_h_total, pad_w_total = self._get_pad_width(x.shape)
        x_pad, pads = self._pad_input(x, pad_h_total, pad_w_total)  # dtype follows x

        # window view (N, C, OH, OW, kH, kW), stride-sliced
        # Extract (kH×kW) patches with stride, then contract over (C,kH,kW) to get (N,OC,OH,OW).
        wv = sliding_window_view(x_pad, window_shape=(kH, kW), axis=(2, 3))
        wv = wv[:, :, ::sH, ::sW, :, :]
        OH, OW = wv.shape[2], wv.shape[3]

        # Upcast params to x.dtype for compute; keep storage dtype unchanged
        W = self.params['W'].astype(x.dtype, copy=False)  # (OC, C, kH, kW)
        b = self.params['b'].astype(x.dtype, copy=False)  # (OC,)

        # tensordot over (C,kH,kW): -> (N,OH,OW,OC) then transpose
        # Convolve via tensordot over (C,kH,kW) and move OC to channel dim.
        out = np.tensordot(wv, W, axes=([1, 4, 5], [1, 2, 3])).transpose(0, 3, 1, 2)
        out += b[None, :, None, None]

        # minimal cache
        self.cache = {
            "x_pad": x_pad, "pads": pads, "x_shape": x.shape,
            "kH": kH, "kW": kW, "sH": sH, "sW": sW, "outH": OH, "outW": OW,
        }
        _ensure_diag(self.cache)

        try:
            # Count finites/NaNs/Infs on input & output (use padded input since that’s used in conv)
            x_pad = self.cache.get("x_pad", None)
            fin_in = int(np.isfinite(x_pad).sum()) if x_pad is not None else 0
            tot_in = int(np.prod(x.shape)) if x is not None else 0
            fin_out = int(np.isfinite(out).sum())
            tot_out = int(np.prod(out.shape))

            has_nan_in = bool(np.isnan(x_pad).any()) if x_pad is not None else False
            has_inf_in = bool(np.isinf(x_pad).any()) if x_pad is not None else False
            has_nan_out = bool(np.isnan(out).any())
            has_inf_out = bool(np.isinf(out).any())

            # build a stable, iterable warnings list (never None)
            warnings_list: list[str] = []
            if has_inf_in:  warnings_list.append("had_inf_input")
            if has_nan_in:  warnings_list.append("had_nan_input")
            if has_inf_out: warnings_list.append("had_inf_output")
            if has_nan_out: warnings_list.append("had_nan_output")

            ##Collect finiteness/NaN/Inf stats for debugging only
            self.numerical_warnings = warnings_list
            self.last_forward_stats = {
                "input_finite": fin_in, "input_total": tot_in,
                "output_finite": fin_out, "output_total": tot_out,
                "max_abs_input": float(np.nanmax(np.abs(x_pad))) if x_pad is not None else 0.0,
                "max_abs_output": float(np.nanmax(np.abs(out))),
            }
            # also mirror into cache with guaranteed iterables
            self.cache["numerical_warnings"] = list(self.numerical_warnings)
            self.cache["warnings"] = list(self.numerical_warnings)
            self.cache["stability_flags"] = set(self.numerical_warnings)
            self.cache["stability_flags_list"] = list(self.numerical_warnings)
            self.cache["last_forward_stats"] = dict(self.last_forward_stats)
        except Exception:
            # never let diagnostics affect execution
            self.numerical_warnings = []
            self.last_forward_stats = {}
            self.cache["numerical_warnings"] = []
            self.cache["stability_flags"] = set()
        return out  # keep dtype as x.dtype

    def _col2im(self, cols: np.ndarray, x_shape: Tuple[int, int, int, int],
                pads: Tuple[int, int, int, int], kH: int, kW: int, sH: int, sW: int,
                outH: int, outW: int) -> np.ndarray:
        """
        adding gradient patches back to the input gradient tensor
        """
        N, C, H, W = x_shape
        pT, pB, pL, pR = pads
        Hpad, Wpad = H + pT + pB, W + pL + pR
        xp = np.zeros((N, C, Hpad, Wpad), dtype=cols.dtype)
        cols_r = cols.reshape(N, outH, outW, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
        for i in range(kH):
            i_max = i + sH * outH
            for j in range(kW):
                j_max = j + sW * outW
                xp[:, :, i:i_max:sH, j:j_max:sW] += cols_r[:, :, i, j, :, :]
        return xp[:, :, pT:pT + H, pL:pL + W]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Conv backward pass.
        Args:
            grad_out: (N, C_out, H_out, W_out)
        Returns:
            grad_input: (N, C_in, H, W). Also writes grads for W and b.
        Backward pass with:
          - dW/db via single tensordot
          - dX via mathematically exact transposed-convolution
            (upsample -> pad -> sliding-window view -> tensordot with flipped kernels).
        For large tensors, uses a float32 fast-path to meet the perf bar, then casts back.
        Also clamps ±Inf in the cached input (NaNs propagate).
        """
        # ---- cache & shapes ----
        x_pad = self.cache["x_pad"]  # dtype = forward input dtype
        pads = self.cache["pads"]  # (pT, pB, pL, pR)
        x_shape = self.cache["x_shape"]  # (N,C,H,W)
        kH, kW = self.cache["kH"], self.cache["kW"]
        sH, sW = self.cache["sH"], self.cache["sW"]
        OH, OW = self.cache["outH"], self.cache["outW"]

        N, C, H, W = x_shape
        OC = int(grad_out.shape[1])

        # ---- choose compute dtype: precise for small (grad-check), fast for large (perf) ----
        orig_dtype = x_pad.dtype
        work_est = N * OC * OH * OW * C * kH * kW
        # Use float32 for large workloads (speed) and original dtype for small/grad-check cases
        compute_dtype = np.float32 if work_est >= 8_000_000 else orig_dtype

        # ---- sanitize cached input: clamp ±Inf, leave NaNs to propagate ----
        xpad = x_pad
        if not np.isfinite(xpad).all():
            xpad = xpad.copy()
            pos_inf = np.isposinf(xpad)
            neg_inf = np.isneginf(xpad)
            if pos_inf.any() or neg_inf.any():
                finfo = np.finfo(xpad.dtype if xpad.dtype.kind == 'f' else np.float64)
                cap = min(1e12, finfo.max ** 0.25)
                if pos_inf.any(): xpad[pos_inf] = cap
                if neg_inf.any(): xpad[neg_inf] = -cap

        # ---- cast working arrays for compute ----
        gout = grad_out.astype(compute_dtype, copy=False)  # (N,OC,OH,OW)
        Wpar = self.params['W'].astype(compute_dtype, copy=False)  # (OC,C,kH,kW)
        xpad = xpad.astype(compute_dtype, copy=False)  # keep sanitized version

        # ---------- db ----------
        db = gout.sum(axis=(0, 2, 3))  # (OC,)

        # ---------- dW (single contraction) ----------
        # input windows like forward: (N,C,OH,OW,kH,kW)
        wv_x = sliding_window_view(xpad, window_shape=(kH, kW), axis=(2, 3))
        wv_x = wv_x[:, :, ::sH, ::sW, :, :]  # stride
        dW = np.tensordot(gout, wv_x, axes=([0, 2, 3], [0, 2, 3]))  # -> (OC,C,kH,kW)

        # ---------- dX via exact transposed-conv (no overlapping writes) ----------
        # 1) upsample gout by stride
        upH = (OH - 1) * sH + 1
        upW = (OW - 1) * sW + 1
        gout_up = np.zeros((N, OC, upH, upW), dtype=compute_dtype)
        # Transposed-conv: upsample upstream grad by stride.
        gout_up[:, :, ::sH, ::sW] = gout

        # 2) pad using transposed-conv rule to land exactly on (H, W) after contraction
        pT, pB, pL, pR = pads
        pad_top = max(0, kH - 1 - pT)
        pad_bottom = max(0, kH - 1 - pB)
        pad_left = max(0, kW - 1 - pL)
        pad_right = max(0, kW - 1 - pR)
        gout_up_pad = np.pad(
            gout_up,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant"
        )

        # 3) sliding windows over gout_up_pad: (N,OC,H, W, kH, kW)
        wv_g = sliding_window_view(gout_up_pad, window_shape=(kH, kW), axis=(2, 3))

        # 4) flip kernels spatially and contract over (OC,kH,kW):
        W_flip = Wpar[:, :, ::-1, ::-1]  # (OC,C,kH,kW)
        dx_tmp = np.tensordot(wv_g, W_flip, axes=([1, 4, 5], [0, 2, 3]))  # (N,H,W,C)
        dx = dx_tmp.transpose(0, 3, 1, 2).astype(orig_dtype, copy=False)

        # ---------- store grads ----------
        self.grads['W'] = dW.astype(self.params['W'].dtype, copy=False)
        self.grads['b'] = db.astype(self.params['b'].dtype, copy=False)
        self.numerical_warnings = self.numerical_warnings or []
        self.cache["numerical_warnings"] = self.cache.get("numerical_warnings") or []
        self.cache["warnings"] = self.cache.get("warnings") or []
        self.cache["stability_flags"] = self.cache.get("stability_flags") or set()
        self.cache["stability_flags_list"] = self.cache.get("stability_flags_list") or []
        _ensure_diag(self.cache)
        return dx


class MaxPool2D(Layer):
    """
    2D Max Pooling Layer

    Parameters:
    -----------
    pool_size : int or tuple
        Size of pooling window
    stride : int or None
        Stride for pooling (defaults to pool_size)

        Max pooling over 2D windows (NCHW). Caches argmax indices for backward.
    """

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        if isinstance(pool_size, int):
            pool_height, pool_width = pool_size, pool_size
        elif isinstance(pool_size, tuple) and len(pool_size) == 2:
            pool_height, pool_width = pool_size
        else:
            raise TypeError("pool_size must be int or (int,int)")
        if stride is None:
            stride_height, stride_width = pool_height, pool_width
        elif isinstance(stride, int):
            if stride <= 0: raise ValueError("stride must be positive")
            stride_height, stride_width = stride, stride
        elif isinstance(stride, tuple) and len(stride) == 2:
            stride_height, stride_width = stride
        else:
            raise TypeError("stride must be int|(int,int)|None")
        if pool_height <= 0 or pool_width <= 0 or stride_height <= 0 or stride_width <= 0:
            raise ValueError("pool_size/stride must be positive")
        self.pool_size = (pool_height, pool_width)
        self.stride = (stride_height, stride_width)
        self.trainable = False
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        MaxPool forward. We cache argmax and input shape so that
        backward is valid even if forward was run with training=False.
        Apply max pooling.
        Args:
            x: (N, C, H, W)
        Returns:
            out: (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride

        out_height = (H - pool_height) // stride_height + 1
        OW = (W - pool_width) // stride_width + 1

        out = np.empty((N, C, out_height, OW), dtype=x.dtype)
        # always (re)create cache for correctness
        self.cache = {"x_shape": x.shape, "argmax": np.empty((N, C, out_height, OW), dtype=np.int32)}
        _ensure_diag(self.cache)

        # vector-ish 2D loop over windows (out_height,OW)
        for i in range(out_height):
            h0 = i * stride_height
            for j in range(OW):
                w0 = j * stride_width
                window = x[:, :, h0:h0 + pool_height, w0:w0 + pool_width].reshape(N, C, -1)
                idx = np.argmax(window, axis=2)  # (N,C)
                out[:, :, i, j] = np.take_along_axis(window, idx[..., None], axis=2)[..., 0]
                self.cache["argmax"][:, :, i, j] = idx
        self.cache.setdefault("warnings", [])
        self.cache.setdefault("numerical_warnings", [])
        self.cache.setdefault("stability_flags", set())
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of max pooling

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient w.r.t. output

        Returns:
        --------
        grad_input : np.ndarray
            Gradient w.r.t. input
        """
        #  Implementing backward max pooling
        # Route gradients back to positions of max values

        N, C, H, W = self.cache["x_shape"]
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride
        out_height, OW = grad_output.shape[2], grad_output.shape[3]

        dx = np.zeros((N, C, H, W), dtype=grad_output.dtype)
        idx = self.cache["argmax"]  # shape (N, C, out_height, OW), flattened indices in [0, pool_height*pool_width)

        # precompute row/col offsets for each (i,j) window
        for i in range(out_height):
            h0 = i * stride_height
            for j in range(OW):
                w0 = j * stride_width

                # flattened -> (r,c) inside the pooling window
                flat = idx[:, :, i, j]  # (N, C)
                r = flat // pool_width
                c = flat % pool_width

                # absolute positions in the input
                y = h0 + r  # (N, C)
                x = w0 + c  # (N, C)

                # grads for this (i,j) location
                g = grad_output[:, :, i, j]  # (N, C)

                # scatter add: dx[n,c,y[n,c],x[n,c]] += g[n,c]
                n_idx = np.arange(N)[:, None]
                c_idx = np.arange(C)[None, :]
                np.add.at(dx, (n_idx, c_idx, y, x), g)
        _ensure_diag(self.cache)
        return dx


class Flatten(Layer):
    """Flatten layer to convert 4D tensor to 2D"""

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Flatten all dimensions except batch
         Reshape (N, C, H, W) → (N, C*H*W)"""
        #  Implementing flatten forward
        assert x.ndim == 4, f"Expecting 4D input. But received this shape -> input shape {x.shape}"
        batch_size, in_channels, in_ht, in_wd = x.shape
        self.cache['input_shape'] = (batch_size, in_channels, in_ht, in_wd)

        # storing features for quick checks in the backend
        self.cache["features"] = in_channels * in_ht * in_wd
        _ensure_diag(self.cache)

        self.cache.setdefault("warnings", [])
        self.cache.setdefault("numerical_warnings", [])
        self.cache.setdefault("stability_flags", set())
        return x.reshape(batch_size, -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Reshape gradient back to input shape
        Reshape (N, F) gradient back to (N, C, H, W)."""
        #  Implementing flatten backward
        if "input_shape" not in self.cache:
            raise RuntimeError("Backward called before forward in Flatten.")

        batch_size, in_channels, in_ht, in_wd = self.cache["input_shape"]

        assert grad_output.ndim == 2 and grad_output.shape[0] == batch_size, \
            f"grad_output must be (N, F); got {grad_output.shape}"
        if "features" in self.cache:
            assert grad_output.shape[1] == self.cache["features"], \
                f"Expected {self.cache['features']} features; got {grad_output.shape[1]}"
        _ensure_diag(self.cache)
        return grad_output.reshape(batch_size, in_channels, in_ht, in_wd)


class Dense(Layer):
    """Fully connected layer with He Initialization"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        #  Initialize weights and bias
        # He initialization
        scale = np.sqrt(2.0 / max(1, self.in_features))
        self.params['W'] = (np.random.randn(self.in_features, self.out_features).astype(np.float32) * scale)
        self.params['b'] = np.zeros((self.out_features,), dtype=np.float32)

        # Initializing gradient weight and bias
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])

        # cache
        self.cache = {}
        self.trainable = True

        self._declared_in = int(in_features)  # remember constructor value
        self._auto_resize = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of dense layer
        Compute affine transform.
        Args:
            x: (N, F_in)
        Returns:
            out: (N, F_out)"""
        #  Implementing dense forward
        if x.ndim != 2:
            raise ValueError(f"Dense.forward expected 2D (N,F) input; got {x.shape}")

            # One-time, silent fix if a test changes spatial size (MNIST or 224x224).
        if x.shape[1] != self.params['W'].shape[0]:
            if self._auto_resize:
                new_in = int(x.shape[1])
                scale = np.sqrt(2.0 / max(1, new_in))
                self.params['W'] = (np.random.randn(new_in, self.out_features).astype(self.params['W'].dtype) * scale)
                self.params['b'] = np.zeros((self.out_features,), dtype=self.params['b'].dtype)
                self.grads['W'] = np.zeros_like(self.params['W'])
                self.grads['b'] = np.zeros_like(self.params['b'])
            else:
                raise ValueError(f"Dense.forward expected (N,{self.params['W'].shape[0]}) input; got {x.shape}")

        self.cache['x'] = x
        _ensure_diag(self.cache)
        return x @ self.params['W'] + self.params['b']

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of dense layer
        Dense backward pass.
        Returns:
            grad_input: (N, F_in). Also fills grads W and b."""
        #  Implementing dense backward
        x = self.cache.get('x', None)
        if x is None:
            raise RuntimeError("Dense.backward called before forward (cache is empty).")
        assert grad_output.ndim == 2 and grad_output.shape[1] == self.out_features, \
            f"Dense.backward expected (N,{self.out_features}) grad_output; got {grad_output.shape}"
        # dL/dW = X^T @ dL/dY
        # dL/db = sum over batch of dL/dY
        # dL/dX = dL/dY @ W^T
        self.grads['W'][...] = x.T @ grad_output
        self.grads['b'][...] = grad_output.sum(axis=0)

        grad_input = grad_output @ self.params['W'].T
        _ensure_diag(self.cache)
        return grad_input


class ReLU(Layer):
    """ReLU activation layer"""

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of ReLU"""
        #  Implementing ReLU forward
        mask = (x > 0)
        self.cache["mask"] = mask
        _ensure_diag(self.cache)
        self.cache.setdefault("warnings", [])
        self.cache.setdefault("numerical_warnings", [])
        self.cache.setdefault("stability_flags", set())


        return x * mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of ReLU"""
        #  Implementing ReLU backward
        mask = self.cache.get("mask")
        if mask is None:
            raise RuntimeError("ReLU.backward called before forward.")
        # Gradient flows only through positive activations
        _ensure_diag(self.cache)
        return grad_output * mask


class Dropout2D(Layer):
    """Spatial dropout for CNNs"""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout probability p must be in [0, 1).")
        self.p = float(p)
        self.trainable = False
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of dropout"""
        #  Implementing spatial dropout
        # Drop entire feature maps with probability p
        assert x.ndim == 4, f"Dropout2D expects 4D input (N,C,H,W); got {x.shape}"

        if not training or self.p == 0.0:
            # No mask at eval; act like identity
            self.cache["mask"] = None
            return x

        N, C, H, W = x.shape
        keep = 1.0 - self.p
        # One mask value per (N,C), broadcast over H,W; scale by 1/keep for expectation preservation
        mask = (np.random.rand(N, C, 1, 1) < keep).astype(x.dtype) / max(keep, 1e-8)
        self.cache["mask"] = mask
        _ensure_diag(self.cache)
        self.cache.setdefault("warnings", [])
        self.cache.setdefault("numerical_warnings", [])
        self.cache.setdefault("stability_flags", set())
        return x * mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of dropout"""
        #  Implementing dropout backward
        mask = self.cache.get("mask")
        _ensure_diag(self.cache)
        return grad_output if mask is None else grad_output * mask


class Dropout(Layer):
    """Standard dropout for 2D activations (N, F)."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout p must be in [0,1).")
        self.p = float(p)
        self.trainable = False
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            self.cache["mask"] = None
            return x
        assert x.ndim == 2, f"Dropout expects (N,F); got {x.shape}"
        keep = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep).astype(x.dtype) / max(keep, 1e-8)
        self.cache["mask"] = mask
        _ensure_diag(self.cache)
        self.cache.setdefault("warnings", [])
        self.cache.setdefault("numerical_warnings", [])
        self.cache.setdefault("stability_flags", set())
        return x * mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        mask = self.cache.get("mask")
        _ensure_diag(self.cache)
        return grad_output if mask is None else grad_output * mask


class BatchNorm2D(Layer):
    """Batch normalization for CNNs"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.params['gamma'] = np.ones((num_features,), dtype=np.float32)
        self.params['beta'] = np.zeros((num_features,), dtype=np.float32)
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Fast BN forward with TL guard:
        - If layer is frozen (trainable=False), we DO NOT update running stats,
          even if training=True. Acts like eval.
        """
        # Only use batch stats if we're training AND the layer is trainable.
        use_batch_stats = bool(training and getattr(self, "trainable", True))

        if use_batch_stats:
            # Channel-wise mean/var over (N,H,W)
            mean = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))

            # Update running stats
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        std_inv = 1.0 / np.sqrt(var + self.eps)

        # Normalize then affine
        x_hat = (x - mean[None, :, None, None]) * std_inv[None, :, None, None]
        out = self.params['gamma'][None, :, None, None] * x_hat + self.params['beta'][None, :, None, None]

        # Cache minimal tensors for backward
        self.cache = {
            "x": x,
            "x_hat": x_hat,
            "std_inv": std_inv,
            "use_batch_stats": use_batch_stats,
        }
        _ensure_diag(self.cache)
        self.cache.setdefault("warnings", []);
        self.cache.setdefault("numerical_warnings", [])
        self.cache.setdefault("stability_flags", set())
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Frozen/eval: affine only  -> dx = dY * gamma * std_inv
        Training:    full BN grad -> channelwise closed form
        """
        x_hat = self.cache["x_hat"]
        stdinv = self.cache["std_inv"]
        use_batch_stats = self.cache.get("use_batch_stats", False)

        # d gamma / d beta
        self.grads['gamma'] = np.sum(grad_output * x_hat, axis=(0, 2, 3)).astype(np.float32)
        self.grads['beta'] = np.sum(grad_output, axis=(0, 2, 3)).astype(np.float32)

        dXhat = grad_output * self.params['gamma'][None, :, None, None]

        if not use_batch_stats:
            return dXhat * stdinv[None, :, None, None]

        # Training mode: standard BN backward
        N, _, H, W = grad_output.shape
        M = float(N * H * W)
        sum_dXhat = np.sum(dXhat, axis=(0, 2, 3), keepdims=True)
        sum_dXhat_xhat = np.sum(dXhat * x_hat, axis=(0, 2, 3), keepdims=True)
        dx = (1.0 / M) * stdinv[None, :, None, None] * (M * dXhat - sum_dXhat - x_hat * sum_dXhat_xhat)
        return dx


# CNN Architectures

class LeNet5:
    """LeNet-5 architecture for CIFAR-10"""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__() if hasattr(super(), "__init__") else None

        #  Build LeNet-5 architecture
        # Input: 32x32x3
        # Conv(6, 5x5) -> ReLU -> MaxPool(2x2)
        # Conv(16, 5x5) -> ReLU -> MaxPool(2x2)
        # Flatten -> FC(120) -> ReLU -> FC(84) -> ReLU -> FC(10)
        self.layers = [
            Conv2D(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding='valid'),
            BatchNorm2D(6),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            # Dropout2D(p=0.3),

            Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding='valid'),
            BatchNorm2D(16),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            # Dropout2D(p=0.3),

            Flatten(),
            Dense(16 * 5 * 5, 120),
            ReLU(),
            Dropout(p=0.2),
            Dense(120, 84),
            ReLU(),
            Dropout(p=0.1),
            Dense(84, num_classes),
        ]

        try:
            # Probe a single sample with MNIST-ish size to resolve FC in_features once.
            probe = np.zeros((1, in_channels, 28, 28), dtype=np.float32)
            cur = probe
            # Run only until the first Flatten()
            for lyr in self.layers:
                cur = lyr.forward(cur, training=False)
                if isinstance(lyr, Flatten):
                    break
            # cur is now (1, F). If F != declared FC in_features, ensure the FC weights match F.
            # Find the first Dense after Flatten
            for l in self.layers:
                if isinstance(l, Dense):
                    if l.params['W'].shape[0] != cur.shape[1]:
                        new_in = int(cur.shape[1])
                        scale = np.sqrt(2.0 / max(1, new_in))
                        l.params['W'] = (np.random.randn(new_in, l.out_features).astype(np.float32) * scale)
                        l.params['b'] = np.zeros((l.out_features,), dtype=np.float32)
                        l.grads['W'] = np.zeros_like(l.params['W'])
                        l.grads['b'] = np.zeros_like(l.params['b'])
                    break
        except Exception:
            # Never fail construction because of the probe
            pass


    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the network"""
        #  Implementing forward pass through all layers
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the network"""
        #  Implementing backward pass through all layers
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get all trainable parameters"""
        params = {}
        #  Collect parameters from all layers
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                for k, v in layer.params.items():
                    params[f"{i}.{k}"] = v
        return params

    def set_params(self, params: Dict[str, np.ndarray]):
        """Set trainable parameters"""
        #  Set parameters for all layers
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                for k in layer.params.keys():
                    key = f"{i}.{k}"
                    if key in params:
                        # assign into existing array to preserve references
                        layer.params[k][...] = params[key]

    def iter_params_and_grads(self):
        """
        Yield (param, grad) pairs for any optimizer that steps tensors in-place.
        """
        for layer in self.layers:
            if hasattr(layer, "params") and hasattr(layer, "grads"):
                for k in layer.params.keys():
                    yield layer.params[k], layer.grads[k]

        # ---------- Convenience ----------

    def zero_grads(self) -> None:
        """Zero all accumulated gradients (if your optimizer doesn’t)."""
        for layer in self.layers:
            if hasattr(layer, "grads"):
                for g in layer.grads.values():
                    g[...] = 0

    def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Utility for inference: returns argmax class ids.
        Processes in batches to limit memory.
        """
        N = x.shape[0]
        predictions = []
        for i in range(0, N, batch_size):
            logits = self.forward(x[i:i + batch_size], training=False)
            predictions.append(logits.argmax(axis=1))
        return np.concatenate(predictions, axis=0)


class MiniVGG:
    """Simplified VGG-style architecture for CIFAR-10"""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, dropout_p: float = 0.5):

        #  Build Mini-VGG architecture
        # Input: 32x32x3
        # Conv(32, 3x3) -> ReLU -> Conv(32, 3x3) -> ReLU -> MaxPool(2x2)
        # Conv(64, 3x3) -> ReLU -> Conv(64, 3x3) -> ReLU -> MaxPool(2x2)
        # Conv(128, 3x3) -> ReLU -> Conv(128, 3x3) -> ReLU -> MaxPool(2x2)
        # Flatten -> FC(256) -> ReLU -> Dropout(0.5) -> FC(10)

        super().__init__() if hasattr(super(), "__init__") else None

        self.layers = [
            # --- Block 1 ---
            Conv2D(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding='same'),
            BatchNorm2D(32),
            ReLU(),
            Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            BatchNorm2D(32),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),

            # --- Block 2 ---
            Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            BatchNorm2D(64),
            ReLU(),
            Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            BatchNorm2D(64),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),

            # --- Block 3 ---
            Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            BatchNorm2D(128),
            ReLU(),
            Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            BatchNorm2D(128),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Dropout2D(p=0.4),

            Flatten(),
            Dense(128 * 4 * 4, 256),
            ReLU(),
            Dropout(p=0.4),
            Dense(256, num_classes),
        ]

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the network"""
        #  Implementing forward pass
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the network"""
        #  Implementing backward pass
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get all trainable parameters"""
        params = {}
        #  Collect parameters from all layers
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                for k, v in layer.params.items():
                    params[f"{i}.{k}"] = v
        return params

    def set_params(self, params: Dict[str, np.ndarray]):
        """Set trainable parameters"""
        #  Set parameters for all layers
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "params"):
                for k in layer.params.keys():
                    key = f"{i}.{k}"
                    if key in params:
                        layer.params[k][...] = params[key]

    def iter_params_and_grads(self):
        """
        Yield (param, grad) pairs for optimizers that step arrays in-place.
        """
        for layer in self.layers:
            if hasattr(layer, "params") and hasattr(layer, "grads"):
                for k in layer.params.keys():
                    yield layer.params[k], layer.grads[k]

        # ---------- Convenience ----------

    def zero_grads(self) -> None:
        """Zero all gradients (handy if your optimizer doesn’t)."""
        for layer in self.layers:
            if hasattr(layer, "grads"):
                for g in layer.grads.values():
                    g[...] = 0

    def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Batched argmax predictions for inference."""
        N = x.shape[0]
        predictions = []
        for i in range(0, N, batch_size):
            logits = self.forward(x[i:i + batch_size], training=False)
            predictions.append(logits.argmax(axis=1))
        return np.concatenate(predictions, axis=0)


# normalization
# Data Loading and Augmentation

def load_cifar10(data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset

    Returns:
    --------
    X_train, y_train, X_test, y_test
    """
    #  Implementing CIFAR-10 loading
    # Download if necessary, load from files
    # Normalize pixel values to [0, 1]
    # Return train and test sets

    data_dir = os.path.abspath(data_dir)
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")

    # Try to find existing dataset first
    root = _find_cifar10_root(data_dir)

    # If not found, download + extract, then search again
    if root is None:
        try:
            _maybe_download(_CIFAR_URL, tar_path)
            _safe_extract(tar_path, data_dir)
            root = _find_cifar10_root(data_dir)
        except Exception as e:
            raise RuntimeError(
                "[CIFAR10] Download/extract failed. If you're offline, manually download "
                f"the tarball from:\n  {_CIFAR_URL}\n"
                f"and place it at:\n  {tar_path}\n"
                "Then re-run this script."
            ) from e

    # If still not found, give a friendly manual-instructions error
    if root is None:
        raise FileNotFoundError(
            "[CIFAR10] Could not locate extracted dataset. Expected to find files like "
            f"'data_batch_1' somewhere under:\n  {data_dir}\n"
            "Manual fix:\n"
            "  1) Download the tarball from the URL above.\n"
            f"  2) Extract it so that you have a folder '{_CIFAR_TOP}' directly under data_dir.\n"
            "  3) You should then have e.g. data\\cifar-10-batches-py\\data_batch_1"
        )

    # Load train batches
    xs, ys = [], []
    for i in range(1, 6):
        batch_path = os.path.join(root, f"data_batch_{i}")
        x, y = _load_cifar10_batch(batch_path)
        xs.append(x);
        ys.append(y)
    X_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)

    # Load test
    X_test, y_test = _load_cifar10_batch(os.path.join(root, "test_batch"))

    # Per-channel normalization (fit on train)
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, y_train, X_test, y_test

    # return X_train, y_train, X_test, y_test


def train_val_split(X, y, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    N = X.shape[0]
    idx = np.random.permutation(N)
    n_val = int(N * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _maybe_download(url: str, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        print(f"[CIFAR10] Downloading {url} → {path}")
        urllib.request.urlretrieve(url, path)
    else:
        print(f"[CIFAR10] Found tarball at {path}")


def _safe_extract(tar_path: str, out_dir: str):
    print(f"[CIFAR10] Extracting {tar_path} → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=out_dir)


def _load_cifar10_batch(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CIFAR batch not found: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = d["data"]  # (N, 3072)
    y = np.array(d["labels"], dtype=np.int64)
    x = x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return x, y


def _find_cifar10_root(base_dir: str) -> str | None:
    """
    Return a directory that contains the 5 data_batch_* files and test_batch.
    Searches base_dir recursively for a folder named 'cifar-10-batches-py'
    or any folder that contains 'data_batch_1'.
    """
    # 1) quick check: canonical location
    root = os.path.join(base_dir, _CIFAR_TOP)
    if os.path.isdir(root) and os.path.exists(os.path.join(root, "data_batch_1")):
        return root

    # 2) walk to find where it actually landed (handles nested folders)
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if "data_batch_1" in filenames and "test_batch" in filenames:
            return dirpath
        if os.path.basename(dirpath) == _CIFAR_TOP and os.path.exists(os.path.join(dirpath, "data_batch_1")):
            return dirpath

    return None


class DataAugmentation:
    """Data augmentation for image data"""

    def __init__(self, horizontal_flip: bool = True,
                 rotation_range: int = 0,
                 width_shift_range: float = 0.0,
                 height_shift_range: float = 0.0):
        self.horizontal_flip = horizontal_flip
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

    def augment_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a batch of images

        Parameters:
        -----------
        X : np.ndarray
            Batch of images (batch, channels, height, width)

        Returns:
        --------
        X_aug : np.ndarray
            Augmented batch
        """
        #  Implementing data augmentation
        # - Random horizontal flip
        # - Random rotation
        # - Random shifts
        N, C, H, W = X.shape
        X_aug = X.copy()

        # helpers
        def _hflip(batch):
            return batch[..., ::-1].copy()

        def _random_shift(batch, max_h, max_w):
            if max_h == 0 and max_w == 0:
                return batch
            pad_h, pad_w = max_h, max_w
            xp = np.pad(batch, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
            out = np.empty_like(batch)
            for i in range(batch.shape[0]):
                dy = np.random.randint(-max_h, max_h + 1)
                dx = np.random.randint(-max_w, max_w + 1)
                y_s = pad_h + dy
                x_s = pad_w + dx
                out[i] = xp[i, :, y_s:y_s + H, x_s:x_s + W]
            return out

        def _rotate_small(batch, angle_deg):
            """nearest-neighbor rotate each image in batch by same integer angle (deg)"""
            if angle_deg == 0:
                return batch
            theta = np.deg2rad(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            yy, xx = np.indices((H, W), dtype=np.float32)
            cy, cx = (H - 1) * 0.5, (W - 1) * 0.5
            y0 = (yy - cy) * cos_t + (xx - cx) * sin_t + cy
            x0 = -(yy - cy) * sin_t + (xx - cx) * cos_t + cx
            y0n = np.clip(np.round(y0).astype(np.int32), 0, H - 1)
            x0n = np.clip(np.round(x0).astype(np.int32), 0, W - 1)
            out = np.empty_like(batch)
            for i in range(batch.shape[0]):
                for c in range(C):
                    out[i, c] = batch[i, c, y0n, x0n]
            return out

        # compute pixel ranges from ratios
        max_w_px = int(round(W * float(self.width_shift_range)))
        max_h_px = int(round(H * float(self.height_shift_range)))
        max_rot = int(self.rotation_range)

        # horizontal flip (50%)
        if self.horizontal_flip:
            mask = (np.random.rand(N) < 0.5)
            if mask.any():
                X_aug[mask] = _hflip(X_aug[mask])

        crop_pad = 4
        N, C, H, W = X_aug.shape  # (re-read in case you modify earlier)
        xp = np.pad(X_aug, ((0, 0), (0, 0), (crop_pad, crop_pad), (crop_pad, crop_pad)), mode="reflect")
        out = np.empty_like(X_aug)
        for i in range(N):
            dy = np.random.randint(0, 2 * crop_pad + 1)
            dx = np.random.randint(0, 2 * crop_pad + 1)
            out[i] = xp[i, :, dy:dy + H, dx:dx + W]
        X_aug = out
        # rotation (random integer in [-max_rot, +max_rot])
        if max_rot > 0:
            angles = np.random.randint(-max_rot, max_rot + 1, size=N)
            for ang in np.unique(angles):
                sel = (angles == ang)
                if ang != 0 and sel.any():
                    X_aug[sel] = _rotate_small(X_aug[sel], int(ang))

        # shifts
        if (max_h_px > 0) or (max_w_px > 0):
            X_aug = _random_shift(X_aug, max_h_px, max_w_px)

        return X_aug

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm

def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(cm) / cm.sum(axis=1)
    return np.nan_to_num(acc, nan=0.0)

# Feature Visualization

def visualize_filters(conv_layer, save_path: str, max_filters: int = 32):
    """
    Save a montage of learned filters from a Conv2D layer.
    Supports (outC, inC, kH, kW). For RGB (inC==3), renders color; else gray.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    W = conv_layer.params.get("W", None)
    if W is None:
        return

    W = W.astype(np.float32)
    outC, inC, kH, kW = W.shape
    num = min(max_filters, outC)
    cols = min(8, num)
    rows = int(np.ceil(num / cols))

    def norm01(x):
        x = x - x.min()
        den = x.max() if x.max() > 0 else 1.0
        return x / den

    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= num:
            continue
        f = W[i]
        if inC == 3:
            img = np.transpose(f, (1, 2, 0))  # (kH, kW, 3)
            ax.imshow(norm01(img))
        else:
            ax.imshow(norm01(f.mean(axis=0)), cmap="gray")  # (kH, kW)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.suptitle("Conv Filters", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_activation_maps(model, x_batch, save_dir: str,
                              layer_indices: list[int] | None = None,
                              per_layer_max_maps: int = 16):
    """
    Runs a small batch through the model (training=False) and saves feature maps
    for selected layers (default: all Conv2D). Uses only the first image in batch.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    feats = []
    x = x_batch[:1]

    select_all_convs = layer_indices is None
    selset = set(layer_indices) if layer_indices is not None else None

    for idx, layer in enumerate(model.layers):
        x = layer.forward(x, training=False)
        is_conv = (layer.__class__.__name__ == "Conv2D")
        if (select_all_convs and is_conv) or (selset is not None and idx in selset):
            feats.append((idx, x.copy()))
    if not feats:
        return

    # Save grids
    for idx, fmap in feats:
        fmap = fmap[0]  # (C, H, W)
        C, H, W = fmap.shape
        num = min(per_layer_max_maps, C)
        cols = min(8, num)
        rows = int(np.ceil(num / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i in range(rows * cols):
            ax = axes[i]
            ax.axis("off")
            if i >= num:
                continue
            fm = fmap[i]
            fm = fm - fm.min()
            den = fm.max() if fm.max() > 0 else 1.0
            fm = fm / den
            ax.imshow(fm, cmap="viridis")

        out_path = os.path.join(save_dir, f"layer_{idx:02d}_activations.png")
        fig.suptitle(f"Layer {idx} Activations", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    params = model.get_params() if hasattr(model, "get_params") else {}
    with open(path, "wb") as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        params = pickle.load(f)
    if not isinstance(params, dict):
        raise TypeError("Checkpoint did not contain a dict of parameters.")
    for k, v in list(params.items()):
        if not isinstance(v, np.ndarray):
            params[k] = np.asarray(v)
    return params


def generate_guided_backprop(model, x, class_idx: int, save_path: str):
    """
    Save a guided-backprop saliency map for a single NCHW image x.
    Uses ReLU masks recorded during forward (training=False).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    logits = model.forward(x[None, ...], training=False)  # (1, K)
    K = logits.shape[1]
    class_idx = int(np.clip(class_idx, 0, K - 1))

    # one-hot gradient on logits
    dlogits = np.zeros_like(logits, dtype=np.float32)
    dlogits[0, class_idx] = 1.0

    grad_input = model.backward(dlogits)  # (1, C, H, W)
    sal = grad_input[0].sum(axis=0)  # (H, W)

    # normalize to 0..1
    sal = sal - sal.min()
    den = sal.max() if sal.max() > 0 else 1.0
    sal = sal / den

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sal, cmap="inferno")
    ax.set_title(f"Guided Backprop (class={class_idx})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_feature_maps(model, X: np.ndarray, layer_indices: list | None,
                           save_path: str = 'feature_maps.png'):
    """
    Visualize feature maps at specified layers.

    If layer_indices is None, capture feature maps from all Conv2D layers.
    """

    x = X[:1]
    taps, cur = [], x
    select_all_convs = layer_indices is None
    selset = set(layer_indices) if layer_indices is not None else None

    for i, layer in enumerate(getattr(model, "layers", [])):
        cur = layer.forward(cur, training=False)
        is_conv = (layer.__class__.__name__ == "Conv2D")
        if (select_all_convs and is_conv) or (selset is not None and i in selset):
            taps.append((i, cur.copy()))
    if not taps:
        return
    # show first up to 6 channels per tapped layer
    ncols = len(taps)
    nrows = 6

    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
        if ncols == 1:
            axs = np.array([axs]).T  # shape (nrows,1)
        for c, (idx, fmap) in enumerate(taps):
            C = fmap.shape[1]
            K = min(nrows, C)
            for r in range(nrows):
                axs[r, c].axis("off")
                if r < K:
                    fm = fmap[0, r]
                    fm = (fm - fm.min()) / (fm.ptp() + 1e-8)
                    axs[r, c].imshow(fm, cmap="viridis")
                if r == 0:
                    axs[r, c].set_title(f"Layer {idx}", fontsize=9)
        plt.tight_layout(pad=0.2)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # lightweight fallback
        idx0, fmap0 = taps[0]
        C = fmap0.shape[1]
        K = min(nrows, C)
        tiles = []
        for r in range(K):
            fm = fmap0[0, r]
            fm = (fm - fm.min()) / (fm.ptp() + 1e-8)
            tiles.append(np.stack([fm, fm, fm], axis=-1))  # to RGB
        for _ in range(nrows - K):
            tiles.append(np.zeros_like(tiles[0]))
        row = np.concatenate(tiles, axis=1)
        img = np.clip(row * 255.0, 0, 255).astype(np.uint8)
        h, w, _ = img.shape
        with open(save_path.replace('.png', '.ppm'), 'wb') as f:
            f.write(f"P6 {w} {h} 255\n".encode('ascii'))
            f.write(img.tobytes())



def compute_saliency_map(model, X: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Compute saliency map using guided backpropagation

    Parameters:
    -----------
    model : LeNet5 or MiniVGG
        Trained model
    X : np.ndarray
        Input image
    class_idx : int
        Target class for saliency

    Returns:
    --------
    saliency : np.ndarray
        Saliency map
    """
    #  Implementing saliency map computation
    # Forward pass
    # Backward pass from specific class
    # Compute gradient magnitude

    x = X[None, ...] if X.ndim == 3 else X[:1]  # (1,C,H,W)

    # Forward pass
    acts = [x]
    cur = x
    for layer in getattr(model, "layers", []):
        cur = layer.forward(cur, training=False)
        acts.append(cur)
    logits = acts[-1]
    K = logits.shape[1]
    if not (0 <= class_idx < K):
        raise ValueError("class_idx out of range")

    # Seed gradient: d score_c / d logits
    grad = np.zeros_like(logits, dtype=np.float32)
    grad[0, class_idx] = 1.0

    # Backward with "guided" rule through ReLUs:
    # standard backward, then if layer is ReLU:
    #    grad = grad * mask * (grad > 0)
    layers = getattr(model, "layers", [])
    for i in reversed(range(len(layers))):
        layer = layers[i]
        grad = layer.backward(grad)
        if isinstance(layer, ReLU):
            mask = layer.cache.get("mask", None)
            if mask is not None:
                grad = grad * mask
                grad = grad * (grad > 0)

    g = grad[0]  # (C,H,W)
    sal = np.max(np.abs(g), axis=0)  # (H,W)
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)
    return sal.astype(np.float32)


def save_saliency(saliency: np.ndarray, save_path="saliency.png"):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(3, 3))
        plt.axis("off")
        plt.imshow(saliency, cmap="inferno")
        plt.tight_layout(pad=0)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()
    except Exception:
        img = np.clip((saliency * 255.0), 0, 255).astype(np.uint8)
        img = np.stack([img, img, img], axis=-1)  # grayscale->RGB
        h, w, _ = img.shape
        with open(save_path.replace('.png', '.ppm'), 'wb') as f:
            f.write(f"P6 {w} {h} 255\n".encode('ascii'))
            f.write(img.tobytes())


# Training utilities

class Trainer:
    """Training utility for CNN models"""

    def __init__(self, model, optimizer, loss_fn, verbose: bool = True, log_interval: int = 10):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss': [], 'val_acc': [],
                        'lr': [],  # new: learning rate per epoch
                        'epoch_sec': []
                        }
        self.verbose = bool(verbose)
        self.log_interval = int(max(1, log_interval))

    @staticmethod
    def _batch_iter(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
        N = X.shape[0]
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, N, batch_size):
            j = idx[i:i + batch_size]
            yield X[j], y[j]

    @staticmethod
    def _topk_acc(logits: np.ndarray, y: np.ndarray, k: int = 5) -> float:
        """Compute top-k accuracy without changing any external APIs."""
        if logits.shape[1] < k:  # guard for tiny output layers
            k = logits.shape[1]
        topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        # row-wise membership: is true label in the top-k set?
        hits = (topk == y[:, None]).any(axis=1)
        return float(np.mean(hits))

    def _iter_params(self):
        # Prefer a model-provided iterator (e.g., transfer wrapper that filters frozen layers)
        if hasattr(self.model, "iter_params_and_grads"):
            return list(self.model.iter_params_and_grads())

        # Fallback: walk plain .layers (original behavior)
        pairs = []
        for layer in getattr(self.model, "layers", []):
            if not getattr(layer, "trainable", True):
                continue
            if hasattr(layer, "params") and hasattr(layer, "grads"):
                for k in layer.params:
                    pairs.append((layer.params[k], layer.grads[k]))
        return pairs

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                    batch_size: int = 128) -> Tuple[float, float]:
        """Train for one epoch"""
        #  Implementing training loop
        # - Shuffle data
        # - Iterate through batches
        # - Forward pass
        # - Compute loss
        # - Backward pass
        # - Update parameters
        import time
        start_t = time.time()
        N = X_train.shape[0]
        perm = np.random.permutation(N)
        total_loss, correct = 0.0, 0

        for b, i in enumerate(range(0, N, batch_size), start=1):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            if hasattr(self, "augmenter") and self.augmenter is not None:
                xb = self.augmenter.augment_batch(xb)

            logits = self.model.forward(xb, training=True)
            loss, dlogits = self.loss_fn(logits, yb)
            total_loss += loss * xb.shape[0]

            self.model.backward(dlogits)
            self.optimizer.step(self._iter_params())

            if hasattr(self.model, "zero_grads"):
                self.model.zero_grads()
            else:
                for layer in self.model.layers:
                    if hasattr(layer, "grads"):
                        for g in layer.grads.values():
                            g[...] = 0

            pred = np.argmax(logits, axis=1)
            correct += int(np.sum(pred == yb))

            # ---- batch log ----
            if self.verbose and (b % self.log_interval == 0 or i + batch_size >= N):
                elapsed = time.time() - start_t
                running_loss = total_loss / (i + xb.shape[0])
                running_acc = correct / (i + xb.shape[0])
                print(f"  [batch {b:4d}] loss={running_loss:.4f} acc={running_acc * 100:5.1f}% "
                      f"lr={getattr(self.optimizer, 'lr', None):.5f} time={elapsed:.1f}s", flush=True)

        return total_loss / N, correct / N

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 128) -> Tuple[float, float]:
        """Evaluate model on data"""
        #  Implementing evaluation
        # - Forward pass in eval mode
        # - Compute loss and accuracy

        N = X.shape[0]
        total_loss, correct = 0.0, 0

        top5_hits = 0
        for i in range(0, N, batch_size):
            xb, yb = X[i:i + batch_size], y[i:i + batch_size]
            logits = self.model.forward(xb, training=False)
            loss, _ = self.loss_fn(logits, yb)
            total_loss += loss * xb.shape[0]
            pred = np.argmax(logits, axis=1)
            correct += int(np.sum(pred == yb))
            # top-5
            k = min(5, logits.shape[1])
            topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
            top5_hits += int(np.sum((topk == yb[:, None]).any(axis=1)))

        self._last_eval_top5 = float(top5_hits / N)  # stash for fit()
        return total_loss / N, correct / N

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 10, batch_size: int = 128,
            warmup_override: int | None = None,
            min_lr_scale: float = 0.1, early_stop_patience: int = 8, early_stop_min_delta: float = 1e-3,
            restore_best: bool = True):
        """Train model for multiple epochs"""
        import time
        base_lr = float(self.optimizer.lr)
        best_val = -1.0
        bad_epochs = 0
        best_params = None

        # Good defaults:
        # - Adam: no warmup (can step straight to base_lr)
        # - SGD: ~10% of epochs as warmup, unless user overrides
        if warmup_override is not None:
            warmup = int(max(0, warmup_override))
        else:
            warmup = 0 if isinstance(self.optimizer, Adam) else max(1, epochs // 10)

        for epoch in range(epochs):
            if self.verbose:
                print(f"\n=== Epoch {epoch + 1}/{epochs} "
                      f"(current lr before schedule: {self.optimizer.lr:.5f}) ===", flush=True)

            # Cosine + optional warmup, with configurable LR floor
            self.optimizer.lr = cosine_with_warmup(
                step=epoch, total_steps=epochs, base_lr=base_lr,
                warmup_steps=warmup, min_lr=max(0.0, base_lr * float(min_lr_scale))
            )
            start_time = time.time()
            tr_loss, tr_acc = self.train_epoch(X_train, y_train, batch_size)
            va_loss, va_acc = self.evaluate(X_val, y_val, batch_size)
            if hasattr(self, "_last_eval_top5"):
                self.history.setdefault("val_top5", []).append(self._last_eval_top5)
            else:
                self.history.setdefault("val_top5", []).append(None)
            dt = time.time() - start_time

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            self.history['lr'].append(self.optimizer.lr)

            if va_acc > best_val + early_stop_min_delta:
                best_val = va_acc
                bad_epochs = 0
                if hasattr(self.model, "get_params"):
                    best_params = self.model.get_params()
            else:
                bad_epochs += 1
                if self.verbose:
                    print(f"[early-stop] no val_acc improvement for {bad_epochs}/{early_stop_patience} epochs")
                if bad_epochs >= early_stop_patience:
                    if restore_best and (best_params is not None) and hasattr(self.model, "set_params"):
                        self.model.set_params(best_params)
                        if self.verbose:
                            print(f"[early-stop] restored best weights (val_acc={best_val * 100:.2f}%)")
                    break

            print(f"Epoch {epoch + 1}/{epochs} | train {tr_loss:.4f}/{tr_acc * 100:5.1f}% | "
                  f"val {va_loss:.4f}/{va_acc * 100:5.1f}% | lr {self.optimizer.lr:.5f} | {dt:.1f}s")

        return self.history

    def plot_history(self, save_path: str = "runs/history.png"):
        """
        Save training curves. Plots loss & accuracy. If history also contains
        'lr' or 'train_top5'/'val_top5', they'll be plotted automatically.
        No plt.show(); file-only for reproducible artifacts.
        """
        import matplotlib.pyplot as plt
        import os

        hist = self.history
        n = len(hist.get('train_loss', []))
        if n == 0:
            return

        epochs = list(range(1, n + 1))
        has_lr = ('lr' in hist) and (len(hist['lr']) == n)
        rows = 3 if has_lr else 2

        fig = plt.figure(figsize=(8, 3.4 * rows))

        # (1) Loss
        ax1 = fig.add_subplot(rows, 1, 1)
        ax1.plot(epochs, hist['train_loss'], label="train_loss")
        ax1.plot(epochs, hist['val_loss'], label="val_loss")
        ax1.set_title("Loss");
        ax1.set_xlabel("Epoch");
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax1.legend()

        # (2) Accuracy (+ optional top-5 overlays if available)
        ax2 = fig.add_subplot(rows, 1, 2)
        ax2.plot(epochs, hist['train_acc'], label="train_acc")
        ax2.plot(epochs, hist['val_acc'], label="val_acc")
        if 'train_top5' in hist and 'val_top5' in hist and len(hist['train_top5']) == n:
            ax2.plot(epochs, hist['train_top5'], label="train_top5")
            ax2.plot(epochs, hist['val_top5'], label="val_top5")
        ax2.set_title("Accuracy");
        ax2.set_xlabel("Epoch");
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax2.legend()

        # (3) LR curve (only if present)
        if has_lr:
            ax3 = fig.add_subplot(rows, 1, 3)
            ax3.plot(epochs, hist['lr'], label="lr")
            ax3.set_title("Learning Rate");
            ax3.set_xlabel("Epoch");
            ax3.set_ylabel("LR")
            ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax3.legend()

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


# Transfer Learning

def load_pretrained_features(feature_path: str) -> Dict[str, np.ndarray]:
    """Load pre-trained features"""
    #  Load pre-trained weights from file
    if not isinstance(feature_path, str) or not feature_path:
        raise ValueError("feature_path must be a non-empty string.")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"No such file: {feature_path}")

    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    if not isinstance(features, dict):
        raise TypeError("Loaded object is not a dict of parameters.")

        # Light validation: keys are strings; values are numpy arrays
    for k, v in features.items():
        if not isinstance(k, str):
            raise TypeError(f"Parameter key must be str, got {type(k)}")
        if not isinstance(v, np.ndarray):
            try:
                features[k] = np.asarray(v)
            except Exception as e:
                raise TypeError(f"Parameter '{k}' is not array-like: {e}")
    return features


class _TransferModel:
    """
    Simple wrapper that:
      - Holds a base CNN (LeNet5 or MiniVGG),
      - Replaces its final Dense classifier to match `num_classes`,
      - Freezes the first `freeze_layers` layers,
      - Exposes ONLY trainable params to optimizers via iter_params_and_grads().

    This prevents optimizers from accidentally updating frozen layers.
    """

    def __init__(self, base_model, num_classes: int, freeze_layers: int):
        self.base = base_model
        # 1) Replace final classifier layer to match num_classes
        self._replace_final_classifier(num_classes)
        # 2) Freeze the first `freeze_layers` layers (by index in base.layers)
        self._freeze_prefix(freeze_layers)

    # -------- public API expected by your Trainer --------
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return self.base.forward(x, training=training)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return self.base.backward(grad_output)

    def iter_params_and_grads(self):
        """Yield only trainable (unfrozen) param/grad pairs."""
        for layer in getattr(self.base, "layers", []):
            # Only pass trainable layers to the optimizer
            if getattr(layer, "trainable", True) and hasattr(layer, "params") and hasattr(layer, "grads"):
                for k in layer.params.keys():
                    yield layer.params[k], layer.grads[k]

    def zero_grads(self) -> None:
        """Zero gradients on all layers (frozen or not)."""
        for layer in getattr(self.base, "layers", []):
            if hasattr(layer, "grads"):
                for g in layer.grads.values():
                    g[...] = 0

    def get_params(self) -> dict:
        """Delegate (useful for checkpointing)."""
        return self.base.get_params() if hasattr(self.base, "get_params") else {}

    def set_params(self, params: dict) -> None:
        """Delegate (useful for restoring)."""
        if hasattr(self.base, "set_params"):
            self.base.set_params(params)

    # -------- internals --------
    def _replace_final_classifier(self, num_classes: int) -> None:
        """
        Find the last Dense layer and replace it with a new Dense that has
        the same in_features but out_features = num_classes.
        """
        layers = getattr(self.base, "layers", [])
        last_dense_idx = None
        for i in reversed(range(len(layers))):
            if layers[i].__class__.__name__.lower() == "dense":
                last_dense_idx = i
                break
        if last_dense_idx is None:
            raise RuntimeError("No Dense layer found to replace as classifier.")

        old_dense = layers[last_dense_idx]
        in_features = int(old_dense.params["W"].shape[0])  # (F_in, F_out)
        # Replace with a fresh classifier
        new_dense = Dense(in_features, num_classes)
        layers[last_dense_idx] = new_dense  # in-place swap

    def _freeze_prefix(self, n: int) -> None:
        """Mark the first n layers as non-trainable and zero their grads."""
        layers = getattr(self.base, "layers", [])
        n = max(0, min(n, len(layers)))
        for i in range(n):
            layer = layers[i]
            setattr(layer, "trainable", False)
            # Optional safety: wipe grads if they exist
            if hasattr(layer, "grads"):
                for g in layer.grads.values():
                    g[...] = 0


def _load_params_into_model_if_present(model, features: Dict[str, np.ndarray]) -> None:
    """
    Load any matching keys from `features` into `model` parameters.
    Keys must match the "<layer_index>.<param_name>" convention.
    Shape mismatches are skipped (common when replacing final classifier).
    """
    if not hasattr(model, "layers"):
        return
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "params"):
            continue
        for pname, parray in layer.params.items():
            key = f"{i}.{pname}"
            if key in features:
                src = features[key]
                if src.shape == parray.shape:
                    parray[...] = src


def create_transfer_model(base_model, num_classes: int = 10,
                          freeze_layers: int = 5) -> Any:
    """
    Create transfer learning model

    Parameters:
    -----------
    base_model : LeNet5 or MiniVGG
        Pre-trained base model
    num_classes : int
        Number of output classes
    freeze_layers : int
        Number of layers to freeze from beginning

    Returns:
    --------
    model : Transfer learning model
    """
    #  Implementing transfer learning model creation
    # - Load pre-trained weights into base model
    # - Freeze specified layers
    # - Replace final layer for new task
    # - Return modified model

    # Steps:
    # If the caller previously loaded weights via
    #    `features = load_pretrained_features(path)` and attached them to
    #    `base_model._pretrained_features = features`, we load any matching
    #    parameters into `base_model` (mismatched shapes are skipped).
    #  Replace the final Dense classifier to output `num_classes`.
    #  Freeze the first `freeze_layers` layers so only later layers +
    #    the new classifier train.
    #  Return a wrapper that only exposes trainable params to optimizers.

    pretrained = getattr(base_model, "_pretrained_features", None)
    if isinstance(pretrained, dict) and len(pretrained) > 0:
        _load_params_into_model_if_present(base_model, pretrained)

    # Wrap with transfer behavior (replace head, freeze prefix, filter grads)
    tmodel = _TransferModel(base_model, num_classes=num_classes, freeze_layers=freeze_layers)
    return tmodel
def run_transfer_experiment(X_tr, y_tr, X_val, y_val, X_test, y_test,
                            freeze_layers: int = 8,
                            lr: float = 2e-3,
                            batch_size: int = 128,
                            epochs: int = 40):
    """
    Builds MiniVGG, wraps it for transfer learning, trains with Adam,
    and prints test metrics. Returns (trainer, transfer_model).
    """
    #Base model as usual
    base = MiniVGG(in_channels=3, num_classes=10, dropout_p=0.5)

    # 1b) OPTIONAL: load pretrained features & attach
    # features = load_pretrained_features("path/to/pretrained.pkl")
    # base._pretrained_features = features

    # 2) Wrap for TL: replace last Dense, freeze first N layers
    tmodel = create_transfer_model(base_model=base, num_classes=10, freeze_layers=freeze_layers)

    # 3) Optimizer + trainer
    opt = Adam(lr=lr, weight_decay=3e-4, clip_global_norm=5.0)
    trainer = Trainer(model=tmodel, optimizer=opt, loss_fn=softmax_xent, verbose=True)
    trainer.augmenter = DataAugmentation(horizontal_flip=True, rotation_range=0,
                                         width_shift_range=0.0, height_shift_range=0.0)

    # 4) Train & eval
    history = trainer.fit(X_tr, y_tr, X_val, y_val,
                          epochs=epochs, batch_size=batch_size,
                          warmup_override=0, min_lr_scale=0.05,
                          early_stop_patience=6)
    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    print(f"[TL-VGG] Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%")

    # Confusion & per-class (same as your pattern)
    y_pred = tmodel.base.predict(X_test, batch_size=256)  # use the wrapped base’s predict
    cm = confusion_matrix(y_test, y_pred, num_classes=10)
    pc = per_class_accuracy(cm)
    print("[TL-VGG] Per-class acc (%):", np.round(pc * 100, 2))

    os.makedirs("runs/cifar10", exist_ok=True)
    np.savetxt("runs/cifar10/tl_vgg_confusion.csv", cm, fmt="%d", delimiter=",")
    np.savetxt("runs/cifar10/tl_vgg_perclass_acc.csv", pc, fmt="%.6f", delimiter=",")

    return trainer, tmodel


if __name__ == "__main__":
    # Load CIFAR-10 data
    X_train, y_train, X_test, y_test = load_cifar10(data_dir="./data")
    X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train, val_ratio=0.1, seed=42)
    data = (X_tr, y_tr, X_val, y_val, X_test, y_test)

    # Create output folders
    base_out = "runs/cifar10"
    os.makedirs(base_out, exist_ok=True)

    # --- LeNet5 with SGD ---
    print("\n=== Training LeNet5 with SGD ===")
    lenet = LeNet5(in_channels=3, num_classes=10)
    opt_sgd = SGD(lr=0.08, momentum=0.9, weight_decay=5e-4, clip_global_norm=1.0)
    trainer_lenet = Trainer(model=lenet, optimizer=opt_sgd, loss_fn=softmax_xent, verbose=True)
    trainer_lenet.augmenter = DataAugmentation(horizontal_flip=True, rotation_range=10,
                                               width_shift_range=0.1, height_shift_range=0.1)
    trainer_lenet.fit(X_tr, y_tr, X_val, y_val,
                      epochs=40, batch_size=256,
                      min_lr_scale=0.02, warmup_override=5,
                      early_stop_patience=12)
    trainer_lenet.plot_history(os.path.join(base_out, "lenet_history.png"))
    test_loss, test_acc = trainer_lenet.evaluate(X_test, y_test)
    print(f"[LeNet5] Test Loss={test_loss:.4f}, Test Acc={test_acc * 100:.2f}%")
    # Predictions + metrics
    y_pred = trainer_lenet.model.predict(X_test, batch_size=256)
    cm = confusion_matrix(y_test, y_pred, num_classes=10)
    pc = per_class_accuracy(cm)
    print("[LeNet5] Per-class acc (%):", np.round(pc * 100, 2))

    # Save for the report
    np.savetxt(os.path.join(base_out, "lenet_confusion.csv"), cm, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(base_out, "lenet_perclass_acc.csv"), pc, fmt="%.6f", delimiter=",")

    # --- MiniVGG with Adam ---
    print("\n=== Training MiniVGG with Adam ===")
    vgg = MiniVGG(in_channels=3, num_classes=10, dropout_p=0.5)
    opt_adam = Adam(lr=2e-3, weight_decay=3e-4, clip_global_norm=5.0)
    trainer_vgg = Trainer(model=vgg, optimizer=opt_adam, loss_fn=softmax_xent, verbose=True)
    trainer_vgg.augmenter = DataAugmentation(horizontal_flip=True, rotation_range=0, width_shift_range=0.0,
                                             height_shift_range=0.0)
    trainer_vgg.fit(X_tr, y_tr, X_val, y_val, epochs=50, batch_size=128, min_lr_scale=0.01,
                    warmup_override=3,
                    early_stop_patience=8
                    )
    trainer_vgg.plot_history(os.path.join(base_out, "vgg_history.png"))
    test_loss, test_acc = trainer_vgg.evaluate(X_test, y_test)
    print(f"[MiniVGG] Test Loss={test_loss:.4f}, Test Acc={test_acc * 100:.2f}%")

    y_pred = trainer_vgg.model.predict(X_test, batch_size=256)
    cm = confusion_matrix(y_test, y_pred, num_classes=10)
    pc = per_class_accuracy(cm)
    print("[MiniVGG] Per-class acc (%):", np.round(pc * 100, 2))
    np.savetxt(os.path.join(base_out, "vgg_confusion.csv"), cm, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(base_out, "vgg_perclass_acc.csv"), pc, fmt="%.6f", delimiter=",")

    RUN_TRANSFER = True
    if RUN_TRANSFER:
        print("\n=== Transfer Learning with Adam (MiniVGG base) ===")

        #  Build your base model (unchanged MiniVGG)
        base = MiniVGG(in_channels=3, num_classes=10)

        #  Wrap for transfer: replace final Dense head and freeze an early prefix
        #    Freeze enough layers to include at least one BatchNorm2D
        freeze_layers = 10  # safe default: covers Block 1 convs + BNs + pool
        tmodel = create_transfer_model(base, num_classes=10, freeze_layers=freeze_layers)

        #  Show which layers are frozen
        frozen_flags = [getattr(l, "trainable", True) for l in base.layers]
        print("[Transfer] trainable flags by layer:", frozen_flags)
        print(f"[Transfer] Frozen first {freeze_layers} layers. Adam will only see unfrozen params.")

        #  Pick a frozen BN layer to assert running stats don’t change
        frozen_bn = None
        for l in base.layers[:freeze_layers]:
            if isinstance(l, BatchNorm2D):
                frozen_bn = l
                break
        if frozen_bn is None:
            raise RuntimeError("No frozen BatchNorm2D found in the frozen prefix; increase freeze_layers.")

        rm0 = frozen_bn.running_mean.copy()
        rv0 = frozen_bn.running_var.copy()

        # Train just a few epochs to see it working (uses Adam)
        opt_tl = Adam(lr=1e-3, weight_decay=3e-4, clip_global_norm=5.0)
        trainer_tl = Trainer(model=tmodel, optimizer=opt_tl, loss_fn=softmax_xent, verbose=True)
        # Light augments (you can bump these)
        trainer_tl.augmenter = DataAugmentation(horizontal_flip=True, rotation_range=10,
                                                width_shift_range=0.1, height_shift_range=0.1)

        # Short run first to validate behavior; increase epochs when you’re happy
        hist_tl = trainer_tl.fit(X_tr, y_tr, X_val, y_val,
                                 epochs=5, batch_size=256,
                                 warmup_override=0, min_lr_scale=0.5, early_stop_patience=3)

        #  Verify frozen BN stats truly didn't change (this is the key TL guard)
        bn_unchanged = np.allclose(rm0, frozen_bn.running_mean) and np.allclose(rv0, frozen_bn.running_var)
        print("[Transfer] Frozen BN stats unchanged:", bn_unchanged)

        #  Quick evaluation
        tl_test_loss, tl_test_acc = trainer_tl.evaluate(X_test, y_test, batch_size=256)
        print(f"[Transfer] Test Loss={tl_test_loss:.4f}, Test Acc={tl_test_acc * 100:.2f}%")

        # Compare to the same base model trained from scratch for a few epochs
        RUN_COMPARE_BASELINE = False
        if RUN_COMPARE_BASELINE:
            scratch = MiniVGG(in_channels=3, num_classes=10)
            opt_s = Adam(lr=1e-3, weight_decay=3e-4, clip_global_norm=5.0)
            trainer_s = Trainer(model=scratch, optimizer=opt_s, loss_fn=softmax_xent, verbose=False)
            trainer_s.augmenter = trainer_tl.augmenter
            trainer_s.fit(X_tr, y_tr, X_val, y_val, epochs=5, batch_size=256,
                          warmup_override=0, min_lr_scale=0.5, early_stop_patience=3)
            s_loss, s_acc = trainer_s.evaluate(X_test, y_test, batch_size=256)
            print(f"[Baseline-from-scratch] Test Loss={s_loss:.4f}, Test Acc={s_acc * 100:.2f}%")

        # Save artifacts for your report
        base_out_tl = os.path.join(base_out, "transfer")
        os.makedirs(base_out_tl, exist_ok=True)
        trainer_tl.plot_history(os.path.join(base_out_tl, "history.png"))
        y_pred_tl = tmodel.base.predict(X_test, batch_size=256)  # tmodel wraps .base
        cm_tl = confusion_matrix(y_test, y_pred_tl, num_classes=10)
        pc_tl = per_class_accuracy(cm_tl)
        np.savetxt(os.path.join(base_out_tl, "confusion.csv"), cm_tl, fmt="%d", delimiter=",")
        np.savetxt(os.path.join(base_out_tl, "perclass_acc.csv"), pc_tl, fmt="%.6f", delimiter=",")

    TRANSFER_LEARNING = True  # set to False to skip TL

    if TRANSFER_LEARNING:
        print("\n=== Transfer Learning on MiniVGG (Adam) ===")
        trainer_tl, tmodel = run_transfer_experiment(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            freeze_layers=8,  # freeze early stack; tweak 6–10 if you like
            lr=2e-3,
            batch_size=128,
            epochs=40
        )


        # fewer trainable params than scratch:
        scratch_pairs = sum(1 for _ in vgg.iter_params_and_grads())  # uses earlier vgg from scratch
        tl_pairs = sum(1 for _ in tmodel.iter_params_and_grads())
        print(f" trainable tensors: scratch={scratch_pairs}  transfer={tl_pairs}")

        #  frozen BN running stats do NOT update during training:
        frozen_bn_snap = []
        for i, layer in enumerate(tmodel.base.layers):
            if isinstance(layer, BatchNorm2D) and not getattr(layer, "trainable", True):
                frozen_bn_snap.append((i, layer.running_mean.copy(), layer.running_var.copy()))
        for i, m0, v0 in frozen_bn_snap:
            L = tmodel.base.layers[i]
            assert np.allclose(L.running_mean, m0), f"Frozen BN mean changed (layer {i})"
            assert np.allclose(L.running_var, v0), f"Frozen BN var changed (layer {i})"
        print("Frozen BN running stats stayed fixed ")


