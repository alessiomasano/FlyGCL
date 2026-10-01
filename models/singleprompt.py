"""SinglePrompt (Park, Lee, Lee, CVPR Findings 2026): frozen ViT + ONE prefix prompt per block on the first `sp_blocks` blocks,
cosine classifier with temperature `sp_tau`. No prompt selection, no query pass: a single forward per iteration.
Port of https://github.com/efficient-learning-lab/SinglePrompt (models/singlePrompt.py)."""
import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()


def prefix_block(block, x, kp, vp):
    """timm Block forward with key/value prefixes (B, L, C) concatenated after the k/v projection (queries untouched)."""
    attn = block.attn; B, N, C = x.shape; H = attn.num_heads
    q, k, v = attn.qkv(block.norm1(x)).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
    k = torch.cat([kp.reshape(B, -1, H, C // H).transpose(1, 2), k], dim=2)
    v = torch.cat([vp.reshape(B, -1, H, C // H).transpose(1, 2), v], dim=2)
    a = attn.attn_drop(((q @ k.transpose(-2, -1)) * attn.scale).softmax(dim=-1))
    x = x + block.drop_path1(block.ls1(attn.proj_drop(attn.proj((a @ v).transpose(1, 2).reshape(B, N, C)))))
    return x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))


class SinglePromptViT(nn.Module):
    def __init__(self, num_classes: int = 100, backbone_name: str = "vit_base_patch16_224", pretrained: bool = True,
                 sp_blocks: int = 5, sp_len: int = 20, sp_tau: float = 0.1, **kwargs):
        super().__init__()
        if hasattr(vit, backbone_name):
            logger.info(f"[SinglePrompt] custom ViT backbone: {backbone_name}")
            self.backbone = getattr(vit, backbone_name)(pretrained=pretrained, num_classes=num_classes)
        else:
            logger.info(f"[SinglePrompt] timm backbone: {backbone_name}")
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.embed_dim = D = self.backbone.embed_dim; self.sp_blocks = sp_blocks; self.sp_len = sp_len; self.tau = sp_tau
        self.prompt = nn.Parameter(torch.empty(sp_blocks, 2, sp_len, D).uniform_(-1, 1))   # original: randn then uniform_(-1, 1)
        self.fc = nn.Linear(D, num_classes, bias=False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode); self.backbone.eval(); return self   # no dropout / stochastic depth in the frozen ViT

    def trainable_parameters(self):
        return [self.prompt, self.fc.weight]

    def features(self, x):
        bb = self.backbone
        x = bb.patch_embed(x); B = x.shape[0]
        x = bb.pos_drop(torch.cat((bb.cls_token.expand(B, -1, -1), x), dim=1) + bb.pos_embed)
        prompt = self.prompt + bb.pos_embed[:, :1]                                      # (K, 2, L, D), CLS pos-embed added as in the original
        for n, block in enumerate(bb.blocks):
            if n < self.sp_blocks:
                kp = prompt[n, 0].expand(B, -1, -1); vp = prompt[n, 1].expand(B, -1, -1)
                x = prefix_block(block, x, kp, vp)
            else:
                x = block(x)
        return bb.norm(x)[:, 0]

    def forward(self, x):
        return F.normalize(self.features(x), dim=1) @ F.normalize(self.fc.weight, dim=1).T / self.tau
