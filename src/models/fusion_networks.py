"""Prediction networks from the frozen benchmark implementation."""
from contextlib import nullcontext

import torch
import torch.nn as nn

from src.models.dose_encoders import DOSE_DIM, DOSE_ENCS, DoseEncFourier, DoseEncSIREN, FilmMod

class ComboModel(nn.Module):
    """
    arch:
      A1 = concat(e1,e2)                    —— MLP 可学到交互，但费力
      A2 = MLP(phi(e1)+phi(e2))             —— 加性池化后使用非线性网络
      A3 = concat + 双线性交互项 e1 ⊙ W e2  —— 显式交互
      A4 = 每种药物先与自身剂量融合，再与细胞表示联合预测
    """
    def __init__(self, drug_dim, cell_dim, dose_enc_id, arch,
                 hidden=512, dropout=0.2, out_dim=1, sigma=0.25, w0=7.0,
                 shared_backbone_init=False, dose_input_scale="none",
                 dose_input_mean=0., dose_input_std=1.):
        super().__init__()
        self.dose_enc_id = dose_enc_id
        self.arch = arch
        self.dose_input_scale = dose_input_scale
        if dose_input_scale != "none":
            if dose_input_std <= 0:
                raise ValueError("剂量输入标准差必须为正")
            self.register_buffer("dose_input_mean", torch.tensor(dose_input_mean, dtype=torch.float32))
            self.register_buffer("dose_input_std", torch.tensor(dose_input_std, dtype=torch.float32))
        dd = DOSE_DIM[dose_enc_id]

        self.drug_proj = nn.Sequential(nn.Linear(drug_dim, 256), nn.GELU())
        self.cell_proj = nn.Sequential(nn.Linear(cell_dim, 128), nn.GELU())
        self.dose_drug_dim = 256
        self.emb_dim = 256

        # 剂量层在独立CPU随机状态中构建，避免其参数量改变后续预测网络的初始化。
        context = torch.random.fork_rng(devices=[]) if shared_backbone_init else nullcontext()
        with context:
            if shared_backbone_init:
                generator = torch.Generator(device="cpu").manual_seed(torch.initial_seed() + 10000)
                torch.set_rng_state(generator.get_state())
            if dose_enc_id == "D4":
                self.dose_enc = FilmMod(256, 64, sigma=sigma)
            elif dose_enc_id == "D3":
                self.dose_enc = DoseEncFourier(sigma=sigma)
            elif dose_enc_id == "D5":
                self.dose_enc = DoseEncSIREN(w0=w0)
            else:
                self.dose_enc = DOSE_ENCS[dose_enc_id]()

        if arch == "A2":
            self.phi = nn.Sequential(nn.Linear(256, self.emb_dim), nn.GELU())
            joint_dim = self.emb_dim + dd * 2 + 128
        elif arch == "A3":
            self.W = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
            joint_dim = self.emb_dim * 2 + self.emb_dim + dd * 2 + 128
        elif arch == "A4":
            if dose_enc_id in ("D0", "D4"):
                raise ValueError("A4 不支持无剂量或 FiLM 剂量编码")
            self.dose_drug_fusion = nn.Sequential(
                nn.Linear(self.dose_drug_dim + dd, 128), nn.GELU())
            joint_dim = 128 * 2 + 128
        else:  # A1
            joint_dim = self.emb_dim * 2 + dd * 2 + 128

        self.head = nn.Sequential(
            nn.Linear(joint_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, d1, d2, c, x1, x2):
        if self.dose_input_scale != "none":
            x1 = (x1 - self.dose_input_mean) / self.dose_input_std
            x2 = (x2 - self.dose_input_mean) / self.dose_input_std
        e1, e2 = self.drug_proj(d1), self.drug_proj(d2)
        cell = self.cell_proj(c)

        if self.arch == "A4":
            z1, z2 = self.dose_enc(x1), self.dose_enc(x2)
            h1 = self.dose_drug_fusion(torch.cat([e1, z1], dim=-1))
            h2 = self.dose_drug_fusion(torch.cat([e2, z2], dim=-1))
            return self.head(torch.cat([h1, h2, cell], dim=-1)).squeeze(-1)

        if self.dose_enc_id == "D4":
            e1, z1 = self.dose_enc(x1, e1)
            e2, z2 = self.dose_enc(x2, e2)
            dose_feat = [z1, z2]
        elif self.dose_enc_id == "D0":
            dose_feat = []
        else:
            dose_feat = [self.dose_enc(x1), self.dose_enc(x2)]

        if self.arch == "A2":
            joint = [self.phi(e1) + self.phi(e2)]
        elif self.arch == "A3":
            joint = [e1, e2, e1 * self.W(e2)]
        else:
            joint = [e1, e2]

        parts = joint + dose_feat + [cell]
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


# ---------------------------------------------------------------- 剂量网格
