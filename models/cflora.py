"""CF-LoRA: frozen ViT backbone + LoRA on q/v, features exposed for a closed-form head (SLDA / random-projection ridge / NCM).

The classifier is NOT a module parameter: it lives in the trainer as streaming statistics. This module only provides
`features(x)` (pre-logit CLS token, with fc_norm) and the LoRA parameters. `lora_enabled(False)` yields the frozen backbone.
"""
import logging
import math

import timm
import torch
import torch.nn as nn

import models.vit as vit
from models.ranpac import Adapter

logger = logging.getLogger()


class LoRAQKV(nn.Module):
    """Wraps a fused qkv Linear(dim, 3*dim); rank-r updates on the q and v slices (B initialised to zero)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        dim = base.in_features; self.dim = dim; self.scale = alpha / rank
        self.A_q = nn.Parameter(torch.empty(rank, dim)); self.B_q = nn.Parameter(torch.zeros(dim, rank))
        self.A_v = nn.Parameter(torch.empty(rank, dim)); self.B_v = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.A_q, a=math.sqrt(5)); nn.init.kaiming_uniform_(self.A_v, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        dq = (x @ self.A_q.t()) @ self.B_q.t() * self.scale
        dv = (x @ self.A_v.t()) @ self.B_v.t() * self.scale
        return torch.cat([out[..., :self.dim] + dq, out[..., self.dim:2 * self.dim], out[..., 2 * self.dim:] + dv], dim=-1)


class CFLoRA(nn.Module):
    def __init__(self, task_num: int = 5, num_classes: int = 100, backbone_name: str = "vit_base_patch16_224",
                 pretrained: bool = True, cf_rank: int = 8, cf_alpha: float = 16.0, cf_head: str = "slda", **kwargs):
        super().__init__()
        self.num_classes = num_classes; self.task_count = 0; self.cf_head = cf_head
        if hasattr(vit, backbone_name):
            logger.info(f"[CFLoRA] custom ViT backbone: {backbone_name}")
            self.backbone = getattr(vit, backbone_name)(pretrained=pretrained, num_classes=num_classes)
        else:
            logger.info(f"[CFLoRA] timm backbone: {backbone_name}")
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.backbone.embed_dim
        self.cf_adapt = kwargs.get("cf_adapt", "lora")
        if self.cf_adapt == "lora" and cf_rank > 0:
            for blk in self.backbone.blocks:
                blk.attn.qkv = LoRAQKV(blk.attn.qkv, cf_rank, cf_alpha)
        elif self.cf_adapt == "adapter":                     # RanPAC / AdaptFormer-style bottleneck adapters, first k blocks
            n_blocks = int(kwargs.get("cf_adapter_blocks", 5)); dim = int(kwargs.get("cf_adapter_dim", 64))
            for blk in list(self.backbone.blocks)[:n_blocks]:
                blk.adapter = Adapter(down_size=dim, n_embd=blk.mlp.fc1.in_features, dropout=0.1)
                for p in blk.adapter.parameters():
                    p.requires_grad_(True)
                def make_forward(block):
                    def fwd(x):
                        x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
                        residual = x
                        return block.adapter(x) + block.drop_path2(block.ls2(block.mlp(block.norm2(x)))) + residual
                    return fwd
                blk.forward = make_forward(blk)
        # optional gradient-trained linear head (ablation "trained head")
        self.linear_head = nn.Linear(self.embed_dim, num_classes) if cf_head == "linear" else None
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode); self.backbone.eval(); return self   # no dropout / stochastic depth in the frozen ViT

    def lora_enabled(self, flag: bool):
        for m in self.backbone.modules():
            if isinstance(m, LoRAQKV):
                m.enabled = flag
            if isinstance(m, Adapter):                       # adapters: zero the learned scale to disable, restore to enable
                if not flag and not hasattr(m, "_saved_scale"):
                    m._saved_scale = m.scale.data.clone(); m.scale.data.zero_()
                elif flag and hasattr(m, "_saved_scale"):
                    m.scale.data.copy_(m._saved_scale); del m._saved_scale

    def lora_parameters(self):
        return [p for n, p in self.backbone.named_parameters() if p.requires_grad]

    def features(self, x):
        h = self.backbone.forward_features(x)[:, 0]
        return self.backbone.fc_norm(h)

    def forward(self, x):
        return self.features(x)

    def process_task_count(self):
        self.task_count += 1
