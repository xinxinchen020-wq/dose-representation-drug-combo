"""Dose encoders from the frozen benchmark implementation."""
import hashlib

import numpy as np
import torch
import torch.nn as nn


class DoseEncNoDose(nn.Module):
    """D0: No dose input (zero-dimensional representation)."""

    out_dim = 0

    def forward(self, x):
        return None


class DoseEncRaw(nn.Module):
    """Direct scalar log-dose input without trainable transformation."""

    out_dim = 1

    def forward(self, x):
        return x.unsqueeze(-1)


class DoseEncFixedSin(nn.Module):
    """D1: Fixed sinusoidal features (0 trainable parameters).

    Uses geometrically spaced frequencies f_i = 10^(-4i/64), alternating
    sin (even i) and cos (odd i). Inputs use log10(uM), where 39 of 64
    channels span the active frequency range of the screens.
    """

    def __init__(self, d=64):
        super().__init__()
        self.d = d
        self.out_dim = d
        i = torch.arange(d, dtype=torch.float32)
        self.register_buffer("freq", 10.0 ** (-4.0 * i / d))

    def _feat(self, x):
        a = x.unsqueeze(1) * self.freq.unsqueeze(0)
        out = torch.empty_like(a)
        out[:, 0::2] = torch.sin(a[:, 0::2])
        out[:, 1::2] = torch.cos(a[:, 1::2])
        return out

    def forward(self, x):
        return self._feat(x)


class DoseEncFixedSinScale(nn.Module):
    """D1p: Fixed sinusoidal features with one trainable global scalar (1 parameter)."""

    def __init__(self, d=64):
        super().__init__()
        self.base = DoseEncFixedSin(d)
        self.out_dim = d
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.base._feat(x) * self.scale


class DoseEncLinear(nn.Module):
    """D2: Trainable linear projection 1 -> d (128 parameters when d=64)."""

    def __init__(self, d=64):
        super().__init__()
        self.fc = nn.Linear(1, d)
        self.out_dim = d

    def forward(self, x):
        return self.fc(x.unsqueeze(-1))


class DoseEncFourier(nn.Module):
    """D3: Fixed random Fourier features followed by a nonlinear MLP (16,576 parameters).

    Draws n_freq=32 fixed Gaussian frequencies B ~ N(0, sigma^2) forming 64-dim
    [sin(2*pi*x*B), cos(2*pi*x*B)], followed by Linear(64, 128) -> GELU -> Linear(128, d).
    Default sigma=0.25 limits phase advance per unit log10 step to <= pi/2 to avoid aliasing.
    """

    def __init__(self, d=64, n_freq=32, sigma=0.25, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(1, n_freq, generator=g) * sigma)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, 128), nn.GELU(), nn.Linear(128, d))
        self.out_dim = d

    def forward(self, x):
        a = 2 * np.pi * x.unsqueeze(-1) @ self.B
        return self.mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=-1))


class DoseEncFixedSinMLP(nn.Module):
    """D1u: Fixed sinusoidal features followed by the matched 64->128->64 MLP (16,576 parameters)."""

    def __init__(self, d=64):
        super().__init__()
        self.base = DoseEncFixedSin(d)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d))
        self.out_dim = d

    def forward(self, x):
        return self.mlp(self.base._feat(x))


class DoseEncLinearMLP(nn.Module):
    """D2u: Trainable linear projection followed by the 64->128->64 MLP (16,704 parameters)."""

    def __init__(self, d=64):
        super().__init__()
        self.lin = nn.Linear(1, d)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d))
        self.out_dim = d

    def forward(self, x):
        return self.mlp(self.lin(x.unsqueeze(-1)))


class DoseEncCompactMLP(nn.Module):
    """Compact scalar dose network: scalar -> 128 linear -> GELU -> d output (8,512 parameters)."""

    def __init__(self, d=64):
        super().__init__()
        self.lin = nn.Linear(1, 128)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(128, d))
        self.out_dim = d

    def forward(self, x):
        return self.mlp(self.lin(x.unsqueeze(-1)))


class DoseEncSinAffine(nn.Module):
    """Fixed sinusoidal features with dimension-wise affine scaling and shift (128 parameters)."""

    def __init__(self, d=64):
        super().__init__()
        self.fc = nn.Linear(1, d)
        self.base = DoseEncFixedSin(d)
        self.out_dim = d

    def forward(self, x):
        return self.base(x) * self.fc.weight.squeeze(1) + self.fc.bias


class DoseEncSinAffineMLP(nn.Module):
    """Fixed sinusoidal features with dimension-wise affine transform and 64->128->64 MLP (16,704 parameters)."""

    def __init__(self, d=64):
        super().__init__()
        self.lin = nn.Linear(1, d)
        self.base = DoseEncFixedSin(d)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d))
        self.out_dim = d

    def forward(self, x):
        features = self.base(x) * self.lin.weight.squeeze(1) + self.lin.bias
        return self.mlp(features)


def fit_dose_input_transform(x1, x2, train_indices, mode):
    """Fit shared dose input scaling statistics strictly on training indices."""
    if mode == "none":
        return dict(mode=mode, mean=0., std=1., n_fit_values=0)
    if mode != "train_standardized":
        raise ValueError(f"Unsupported dose input scaling mode: {mode}")
    values = np.concatenate((np.asarray(x1)[train_indices], np.asarray(x2)[train_indices])).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values found in training dose inputs")
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if std <= 1e-12:
        raise ValueError("Training dose inputs lack sufficient variance for standardization")
    return dict(mode=mode, mean=mean, std=std, n_fit_values=len(values))


def module_hash(module):
    """Compute a deterministic SHA-256 digest of a module's state dictionary."""
    if module is None:
        return None
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class DoseEncSIREN(nn.Module):
    """D5: Sinusoidal representation network (SIREN) with calibrated frequency w0."""

    def __init__(self, d=64, w0=7.0):
        super().__init__()
        self.w0 = w0
        self.l1 = nn.Linear(1, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, d)
        self.out_dim = d

    def forward(self, x):
        h = torch.sin(self.w0 * self.l1(x.unsqueeze(-1)))
        h = torch.sin(self.w0 * self.l2(h))
        return self.l3(h)


class FilmMod(nn.Module):
    """D4: Feature-wise linear modulation (FiLM) driven by random Fourier dose features."""

    def __init__(self, dru_dim, d=64, sigma=0.25):
        super().__init__()
        self.enc = DoseEncFourier(d, sigma=sigma)
        self.to_gb = nn.Linear(d, 2 * dru_dim)
        self.out_dim = d
        self.dru_dim = dru_dim

    def forward(self, x, drug):
        z = self.enc(x)
        g, b = self.to_gb(z).chunk(2, dim=-1)
        return drug * (1 + torch.tanh(g)) + b, z


DOSE_ENCS = {
    "raw": DoseEncRaw,
    "D0": DoseEncNoDose,
    "D1": DoseEncFixedSin,
    "D2": DoseEncLinear,
    "D3": DoseEncFourier,
    "D4": FilmMod,
    "D5": DoseEncSIREN,
    "D1p": DoseEncFixedSinScale,
    "D1u": DoseEncFixedSinMLP,
    "D2u": DoseEncLinearMLP,
    "linear64": DoseEncLinear,
    "sin_affine64": DoseEncSinAffine,
    "linear64_mlp": DoseEncLinearMLP,
    "sin_affine64_mlp": DoseEncSinAffineMLP,
    "compact_scalar_mlp": DoseEncCompactMLP,
}
DOSE_DIM = {
    "raw": 1, "D0": 0, "D1": 64, "D2": 64, "D3": 64, "D4": 64, "D5": 64,
    "D1p": 64, "D1u": 64, "D2u": 64, "linear64": 64, "sin_affine64": 64,
    "linear64_mlp": 64, "sin_affine64_mlp": 64, "compact_scalar_mlp": 64,
}
CORE_ENCS = ["D0", "D1", "D2", "D3", "D4", "D5"]


class DoseEncRaw64Fixed(nn.Module):
    """Direct log-dose input padded to 64 dimensions with fixed zeros (0 parameters)."""

    out_dim = 64

    def forward(self, x):
        return torch.cat((x.unsqueeze(-1), x.new_zeros((len(x), 63))), dim=-1)


DOSE_ENCS["raw64_fixed"] = DoseEncRaw64Fixed
DOSE_DIM["raw64_fixed"] = 64
