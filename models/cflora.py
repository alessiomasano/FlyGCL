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
from models.singleprompt import prefix_block

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


def gated_prefix_block(block, x, kp, vp, g=None, scale=None):
    """E45: zero-init attention (LLaMA-Adapter): the prefix gets its own softmax and enters scaled by tanh(g); g = 0 is the frozen block.
    E63: with g None the prefix term enters with the constant `scale` (ablation: two softmaxes without a learned gate)."""
    attn = block.attn; B, N, C = x.shape; H = attn.num_heads
    q, k, v = attn.qkv(block.norm1(x)).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
    kp, vp = (t.reshape(B, -1, H, C // H).transpose(1, 2) for t in (kp, vp))
    s = torch.tanh(g) if g is not None else scale
    o = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(-1) @ v + s * (((q @ kp.transpose(-2, -1)) * attn.scale).softmax(-1) @ vp)
    x = x + block.drop_path1(block.ls1(attn.proj_drop(attn.proj(o.transpose(1, 2).reshape(B, N, C)))))
    return x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))


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
        elif self.cf_adapt in ("prefix", "softprompt"):     # E45: prompt modules on the first K blocks (shallow soft prompt: input only)
            K = int(kwargs.get("cf_prompt_blocks", 5)); L = int(kwargs.get("cf_prompt_len", 20)); D = self.embed_dim
            self.prompt_K = K; self.prompt_shallow = bool(kwargs.get("cf_prompt_shallow", False))
            if self.cf_adapt == "prefix":                    # as SinglePrompt: uniform(-1, 1) K/V prefixes
                self.prompt = nn.Parameter(torch.empty(K, 2, L, D).uniform_(-1, 1))
                self.gate = nn.Parameter(torch.zeros(K)) if kwargs.get("cf_prompt_gate", False) else None
                self.gate_fixed = kwargs.get("cf_gate_fixed", None)   # E63: two softmaxes with a constant scale instead of the learned zero-init gate
                if self.gate_fixed is not None:
                    self.gate = None
                if kwargs.get("cf_prompt_v_zero", False):              # E63: P_V at zero (DIKI-style): the prefix term starts at zero without a gate
                    self.prompt.data[:, 1].zero_()
            else:                                            # as VPT: uniform with the patch-embedding fan
                v = math.sqrt(6.0 / (3 * 16 * 16 + D))
                self.prompt = nn.Parameter(torch.empty(1 if self.prompt_shallow else K, L, D).uniform_(-v, v))
        self.prompt_on = True
        # optional gradient-trained linear head (ablation "trained head")
        self.linear_head = nn.Linear(self.embed_dim, num_classes) if cf_head == "linear" else None
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode); self.backbone.eval(); return self   # no dropout / stochastic depth in the frozen ViT

    def lora_enabled(self, flag: bool):
        self.prompt_on = flag
        for m in self.backbone.modules():
            if isinstance(m, LoRAQKV):
                m.enabled = flag
            if isinstance(m, Adapter):                       # adapters: zero the learned scale to disable, restore to enable
                if not flag and not hasattr(m, "_saved_scale"):
                    m._saved_scale = m.scale.data.clone(); m.scale.data.zero_()
                elif flag and hasattr(m, "_saved_scale"):
                    m.scale.data.copy_(m._saved_scale); del m._saved_scale

    def lora_parameters(self):
        return [p for n, p in self.backbone.named_parameters() if p.requires_grad] + \
            [p for p in (getattr(self, "prompt", None), getattr(self, "gate", None)) if p is not None]

    def features(self, x):
        if not (self.cf_adapt in ("prefix", "softprompt") and self.prompt_on):
            return self.backbone.fc_norm(self.backbone.forward_features(x)[:, 0])
        bb = self.backbone; x = bb.patch_embed(x); B = x.shape[0]; K = self.prompt_K
        x = bb.pos_drop(torch.cat((bb.cls_token.expand(B, -1, -1), x), dim=1) + bb.pos_embed)
        if self.cf_adapt == "prefix":
            P = self.prompt + bb.pos_embed[:, :1]           # CLS pos-embed added, as SinglePrompt
            for n, blk in enumerate(bb.blocks):
                if n >= K:
                    x = blk(x)
                elif self.gate is None and getattr(self, "gate_fixed", None) is None:   # single softmax over [prefix; tokens] (SinglePrompt port)
                    x = prefix_block(blk, x, P[n, 0].expand(B, -1, -1), P[n, 1].expand(B, -1, -1))
                elif self.gate is None:                                                  # E63: two softmaxes, constant scale
                    x = gated_prefix_block(blk, x, P[n, 0].expand(B, -1, -1), P[n, 1].expand(B, -1, -1), scale=self.gate_fixed)
                else:
                    x = gated_prefix_block(blk, x, P[n, 0].expand(B, -1, -1), P[n, 1].expand(B, -1, -1), g=self.gate[n])
        else:                                                # soft prompt tokens right after CLS (VPT)
            L = self.prompt.shape[1]
            for n, blk in enumerate(bb.blocks):
                if n == 0 or (not self.prompt_shallow and n < K):    # insert at 0; deep: replace the prompt tokens in blocks < K
                    rest = x[:, 1:] if n == 0 else x[:, 1 + L:]
                    x = torch.cat((x[:, :1], self.prompt[n].expand(B, -1, -1).to(x.dtype), rest), dim=1)
                elif not self.prompt_shallow and n == K:            # deep: blocks >= K see the standard sequence
                    x = torch.cat((x[:, :1], x[:, 1 + L:]), dim=1)
                x = blk(x)
        return bb.fc_norm(bb.norm(x)[:, 0])

    def forward(self, x):
        return self.features(x)

    def process_task_count(self):
        self.task_count += 1
