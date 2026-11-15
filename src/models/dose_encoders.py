"""Dose encoders from the frozen benchmark implementation."""
import hashlib

import numpy as np
import torch
import torch.nn as nn

class DoseEncNoDose(nn.Module):
    """D0: 不使用剂量"""
    out_dim = 0
    def forward(self, x):
        return None


class DoseEncRaw(nn.Module):
    """直接输入一个对数剂量，不增加可训练的剂量变换。"""
    out_dim = 1

    def forward(self, x):
        return x.unsqueeze(-1)


class DoseEncFixedSin(nn.Module):
    """D1: 固定 sin/cos —— 0 参数

    公式沿用 CDCR-Rank Eq.(1)：f_i = 10^(-4i/64)，i 偶用 sin、i 奇用 cos。
    ⚠️ 注意：CDCR-Rank 未定义剂量输入的数值尺度，而该尺度决定此编码是否退化。
       本研究统一使用 log10(µM)，此时 39/64 维有效（见 checks/dose_enc_capacity.py）。
    """
    def __init__(self, d=64):
        super().__init__()
        self.d = d; self.out_dim = d
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
    """D1p（块6）: D1 + 一个可学习缩放 —— 共 1 个参数

    回答「只给一个可学习参数，能否补上 D1 与 D3/D4 的差距？」
    """
    def __init__(self, d=64):
        super().__init__()
        self.base = DoseEncFixedSin(d)
        self.out_dim = d
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.base._feat(x) * self.scale


class DoseEncLinear(nn.Module):
    """D2: 线性投影 —— 128 参数（CDCR-Rank 消融里的 'linear' 变体）"""
    def __init__(self, d=64):
        super().__init__()
        self.fc = nn.Linear(1, d); self.out_dim = d
    def forward(self, x):
        return self.fc(x.unsqueeze(-1))


class DoseEncFourier(nn.Module):
    """D3: Fourier 特征 + MLP（Tancik et al. 2020 的随机 Fourier 特征）

    n_freq=32 → 输入 64 维 → MLP(64→128→64) = 16,576 参数

    ★ σ 的定标（2026-08-09 修正）
      Tancik 的规则：σ = 目标函数在**每单位坐标**上的周期数，且 B 是**固定** buffer。
      σ 过大会让网络去追目标里不存在的频率。原实现的 σ=2.0 是随手写的，实测：
        · |B| 中位 1.50 周期/log-单位，而剂量-响应的合理频率约 0.33 → 高 4.5 倍
        · 24/32 个维度的相邻剂量档相位推进 > π → **混叠**（相邻档完全错开）
      现按「一个剂量档步长（Δlog10=1）内相位推进不超过 π/2」定标：
        2π·σ·Δx ≤ π/2  ⇒  σ ≤ 1/(4Δx) = 0.25
      取 **σ = 0.25**。该值经 --sigma 暴露，便于做稳健性扫描。
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
    """D1u（块6）: D1 的固定 sin/cos → 与 D3 **完全相同**的 MLP

    参数量与 D3 逐位相同（16,576），唯一差别是输入特征：
      D3  = 固定随机频率的 Fourier 特征 + MLP
      D1u = 固定 sin/cos（频率不可学）+ 同一个 MLP
    → 在【参数量匹配】下检验「编码形式」是否重要
    """
    def __init__(self, d=64):
        super().__init__()
        self.base = DoseEncFixedSin(d)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d))
        self.out_dim = d
    def forward(self, x):
        return self.mlp(self.base._feat(x))


class DoseEncLinearMLP(nn.Module):
    """D2u（块6）: D2 的线性投影 → 与 D3 相同的 MLP（16,576 + 128 参数）

    与 D1u 的区别只在输入特征：线性投影 vs 固定 sin/cos。
    """
    def __init__(self, d=64):
        super().__init__()
        self.lin = nn.Linear(1, d)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, d))
        self.out_dim = d
    def forward(self, x):
        return self.mlp(self.lin(x.unsqueeze(-1)))


class DoseEncCompactMLP(nn.Module):
    """紧凑直接标量网络：标量→128维线性→GELU→64维输出，共8,512个参数。

    与 DoseEncLinearMLP 的区别是把 1→64 与 64→128 两个中间没有非线性的
    线性变换合并为一个 1→128 变换，用于检验额外线性投影是否必要。
    """
    def __init__(self, d=64):
        super().__init__()
        self.lin = nn.Linear(1, 128)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(128, d))
        self.out_dim = d

    def forward(self, x):
        return self.mlp(self.lin(x.unsqueeze(-1)))


class DoseEncSinAffine(nn.Module):
    """固定正弦特征的逐维缩放和平移；与1→64线性投影同为128个参数。"""
    def __init__(self, d=64):
        super().__init__()
        self.fc = nn.Linear(1, d)
        self.base = DoseEncFixedSin(d)
        self.out_dim = d

    def forward(self, x):
        return self.base(x) * self.fc.weight.squeeze(1) + self.fc.bias


class DoseEncSinAffineMLP(nn.Module):
    """固定正弦、逐维仿射变换及64→128→64网络，共16,704个参数。"""
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
    """两个药物共用一个变换，统计量只由训练剂量计算。"""
    if mode == "none":
        return dict(mode=mode, mean=0., std=1., n_fit_values=0)
    if mode != "train_standardized":
        raise ValueError("未知剂量输入尺度")
    values = np.concatenate((np.asarray(x1)[train_indices], np.asarray(x2)[train_indices])).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("训练剂量存在非有限值")
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if std <= 1e-12:
        raise ValueError("训练剂量没有足够变化，不能标准化")
    return dict(mode=mode, mean=mean, std=std, n_fit_values=len(values))


def module_hash(module):
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
    """D5: SIREN（Sitzmann et al. 2020 的周期激活 MLP）

    ★ ω₀ 的定标（2026-08-09 修正）
      SIREN 原文假设输入 x ~ U(−1,1)，第一层 sin(ω₀(Wx+b))、ω₀=30。
      原文明确写道：「若希望更平滑的响应，**或输入未正确归一化**，宜用更小的 ω₀
      或相应地对输入重缩放。」
      我们直接喂 log10(µM) ∈ [−5.22, 3.00]，跨度 8.22（原文假设的 4.1 倍）：
      原 ω₀=30 等效于归一化空间里的 30×4.1 ≈ 123，初始化预激活达 ±155 弧度
      （= 25 个周期，原文期望 ~2）。
      按原文建议缩小 ω₀：**ω₀ = 30 / (跨度/2) = 30/4.11 ≈ 7.3，取 7.0**。
      该值经 --w0 暴露，便于做稳健性扫描。
    """
    def __init__(self, d=64, w0=7.0):
        super().__init__()
        self.w0 = w0
        self.l1 = nn.Linear(1, 128); self.l2 = nn.Linear(128, 128); self.l3 = nn.Linear(128, d)
        self.out_dim = d
    def forward(self, x):
        h = torch.sin(self.w0 * self.l1(x.unsqueeze(-1)))
        h = torch.sin(self.w0 * self.l2(h))
        return self.l3(h)


class FilmMod(nn.Module):
    """D4: FiLM —— 剂量嵌入产生 (gamma, beta) 调制药嵌入

    内部剂量编码器与 D3 相同（Tancik 随机 Fourier 特征），因此共享同一 σ 定标。
    """
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
    # 块6 参数量对照
    "D1p": DoseEncFixedSinScale,
    "D1u": DoseEncFixedSinMLP,
    "D2u": DoseEncLinearMLP,
    "linear64": DoseEncLinear,
    "sin_affine64": DoseEncSinAffine,
    "linear64_mlp": DoseEncLinearMLP,
    "sin_affine64_mlp": DoseEncSinAffineMLP,
    "compact_scalar_mlp": DoseEncCompactMLP,
}
DOSE_DIM = {"raw": 1, "D0": 0, "D1": 64, "D2": 64, "D3": 64, "D4": 64, "D5": 64,
            "D1p": 64, "D1u": 64, "D2u": 64, "linear64": 64, "sin_affine64": 64,
            "linear64_mlp": 64, "sin_affine64_mlp": 64, "compact_scalar_mlp": 64}
CORE_ENCS = ["D0", "D1", "D2", "D3", "D4", "D5"]


# ---------------------------------------------------------------- 模型


class DoseEncRaw64Fixed(nn.Module):
    out_dim = 64

    def forward(self, x):
        return torch.cat((x.unsqueeze(-1), x.new_zeros((len(x), 63))), dim=-1)

DOSE_ENCS["raw64_fixed"] = DoseEncRaw64Fixed
DOSE_DIM["raw64_fixed"] = 64
