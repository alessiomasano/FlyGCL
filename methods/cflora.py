"""CF-LoRA trainer: closed-form streaming head (SLDA / RP-ridge / NCM) + slowly adapted LoRA, task-free.

Per incoming batch (predict-then-learn):
  1. features of the (augmented) batch with the current adapter → head logits → prequential accuracy;
  2. statistics of the head updated with these pre-update features (closed form, no gradient);
  3. `online_iter` AdamW steps on the LoRA only, loss = CE(head logits with the head detached, y).
No task identity is used anywhere (online_before/after_task are no-ops). `--lr 0` gives the frozen closed-form baseline.
"""
import gc
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods._trainer import _Trainer
from torch.utils.data import DataLoader
from utils.onlinesampler import OnlineTestSampler

logger = logging.getLogger()


class StreamSLDA:
    """Streaming LDA: class sums/counts + shared within-class scatter (Welford), shrinkage; unseen classes get -1e9."""

    def __init__(self, d, n_classes, device, shrink=1e-2, bias_corr=0.0, mean_shrink=0.0, **kw):
        self.s = torch.zeros(n_classes, d, device=device); self.n = torch.zeros(n_classes, device=device)
        self.S = torch.zeros(d, d, device=device); self.N = 0.0; self.shrink = shrink; self.d = d; self.device = device; self._Wb = None
        self.bias_corr = bias_corr; self.mean_shrink = mean_shrink   # E34: small-n plug-in bias correction / mean shrinkage to the global mean

    @torch.no_grad()
    def learn(self, z, y):
        z = z.float(); mu = self.s / self.n.clamp(min=1).unsqueeze(1)
        seen = self.n[y] > 0
        if seen.any():
            r = (z - mu[y])[seen]; w = (self.n[y] / (self.n[y] + 1))[seen]
            self.S += (r * w.unsqueeze(1)).T @ r
        self.s.index_add_(0, y, z); self.n.index_add_(0, y, torch.ones(len(y), device=self.device)); self.N += len(y); self._Wb = None

    @torch.no_grad()
    def refresh(self):
        with torch.autocast(device_type="cuda", enabled=False):
            mu = self.s / self.n.clamp(min=1).unsqueeze(1)
            if self.mean_shrink > 0:   # mu_c <- n/(n+k) mu_c + k/(n+k) mu_bar
                w = (self.n / (self.n + self.mean_shrink)).unsqueeze(1); mu = w * mu + (1 - w) * (self.s.sum(0) / max(self.N, 1))
            Sig_raw = self.S / max(self.N - 1, 1); Sig = (1 - self.shrink) * Sig_raw + self.shrink * torch.eye(self.d, device=self.device)
            Sig_inv = torch.linalg.inv(Sig); W = mu @ Sig_inv; b = -0.5 * (W * mu).sum(1)
            if self.bias_corr > 0:     # alpha * E[eps^T Sig^-1 eps] = tr(Sig^-1 Sig_within) / n_c: remove the plug-in noise of the quadratic term
                b = b + self.bias_corr * 0.5 * torch.trace(Sig_inv @ Sig_raw) / self.n.clamp(min=1)
            b[self.n == 0] = -1e9
            self._Wb = (W, b)

    def logits(self, z):
        if self._Wb is None:
            self.refresh()
        W, b = self._Wb
        with torch.autocast(device_type="cuda", enabled=False):
            return z.float() @ W.T + b


class StreamRanDumb(StreamSLDA):
    """RanDumb head (Prabhu et al. 2024) on pre-trained features: with a pre-trained embedder RanDumb applies no random projection,
    so it is streaming LDA whose covariance is regularised by the OAS estimator (Chen et al. 2010; sklearn.covariance.OAS formula)
    instead of a fixed shrinkage."""

    @torch.no_grad()
    def refresh(self):
        with torch.autocast(device_type="cuda", enabled=False):
            mu = self.s / self.n.clamp(min=1).unsqueeze(1)
            n = max(self.N, 2.0); p = self.d
            S = self.S / (n - 1)
            m = torch.trace(S) / p; a = (S * S).mean()
            if m <= 0:                                     # no within-class scatter yet (first samples): plain NCM
                rho = torch.tensor(1.0, device=self.device); Sig = torch.eye(p, device=self.device)
            else:
                rho = torch.nan_to_num((a + m * m) / ((n + 1 - 2 / p) * (a - m * m / p)), nan=1.0).clamp(0.0, 1.0)
                Sig = (1 - rho) * S + rho * m * torch.eye(p, device=self.device)
            W = mu @ torch.linalg.inv(Sig); b = -0.5 * (W * mu).sum(1); b[self.n == 0] = -1e9
            self._Wb = (W, b); self.rho = float(rho)


class StreamNCM:
    def __init__(self, d, n_classes, device, **kw):
        self.s = torch.zeros(n_classes, d, device=device); self.n = torch.zeros(n_classes, device=device); self.device = device; self.N = 0.0

    @torch.no_grad()
    def learn(self, z, y):
        z = F.normalize(z.float(), dim=1); self.s.index_add_(0, y, z); self.n.index_add_(0, y, torch.ones(len(y), device=self.device)); self.N += len(y)

    def refresh(self):
        pass

    def logits(self, z, beta=30.0):
        with torch.autocast(device_type="cuda", enabled=False):
            mu = F.normalize(self.s / self.n.clamp(min=1).unsqueeze(1), dim=1)
            out = beta * F.normalize(z.float(), dim=1) @ mu.T
            out[:, self.n == 0] = -1e9
            return out


class StreamRP:
    """RanPAC-style head: fixed random projection d→M, ReLU, streaming ridge regression on one-hot targets
    (Q = H^T Y, G = H^T H, W = (G + λI)^-1 Q). Order-invariant. Refreshed every `refresh_every` learn() calls."""

    def __init__(self, d, n_classes, device, rp_dim=10000, ridge=1e5, refresh_every=10, seed=0, **kw):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W_rand = torch.randn(d, rp_dim, generator=g).to(device)
        self.Q = torch.zeros(rp_dim, n_classes, device=device); self.G = torch.zeros(rp_dim, rp_dim, device=device)
        self.n = torch.zeros(n_classes, device=device); self.N = 0.0; self.ridge = ridge; self.device = device
        self.refresh_every = refresh_every; self._since = 0; self._W = None; self.n_classes = n_classes

    def _h(self, z):
        return F.relu(z.float() @ self.W_rand)

    @torch.no_grad()
    def learn(self, z, y):
        with torch.autocast(device_type="cuda", enabled=False):
            h = self._h(z); Y = F.one_hot(y, self.n_classes).float()
            self.Q += h.T @ Y; self.G += h.T @ h
        self.n.index_add_(0, y, torch.ones(len(y), device=self.device)); self.N += len(y); self._since += 1
        if self._since >= self.refresh_every:
            self._W = None

    @torch.no_grad()
    def refresh(self):
        with torch.autocast(device_type="cuda", enabled=False):
            self._W = torch.linalg.solve(self.G + self.ridge * torch.eye(self.G.shape[0], device=self.device), self.Q)
        self._since = 0

    def logits(self, z):
        if self._W is None:
            self.refresh()
        with torch.autocast(device_type="cuda", enabled=False):
            out = self._h(z) @ self._W
            out[:, self.n == 0] = -1e9
            return out


HEADS = {"slda": StreamSLDA, "ncm": StreamNCM, "rp": StreamRP, "randumb": StreamRanDumb}


class CFLoRATrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = 0; self.head = None

    # ------------------------------------------------------------------ setup
    def setup_distributed_model(self):
        super().setup_distributed_model()
        m = self.model_without_ddp
        self.lora_params = m.lora_parameters()
        if self.cf_head != "linear":
            self.head = HEADS[self.cf_head](m.embed_dim, self.n_classes, self.device, shrink=self.cf_shrink, rp_dim=self.cf_rp_dim,
                                            bias_corr=self.cf_bias_corr, mean_shrink=self.cf_mean_shrink,
                                            ridge=self.cf_ridge, refresh_every=self.cf_refresh, seed=self.rnd_seed)
        # warm phase (first cf_warm samples): LoRA + linear head trained with CE at cf_warm_lr, prediction by the linear head,
        # closed-form statistics accumulated meanwhile (as RanPAC does during its first session). Then: closed-form prediction and
        # LoRA either frozen (cf_after=freeze, RanPAC-like) or slowly adapted at --lr (cf_after=slow, ours).
        self.warm = (self.cf_warm > 0 or self.cf_warm_auto) and self.cf_head != "linear"
        self.in_warm = self.warm; self.seen_samples = 0
        self.ema_closed = 0.0; self.ema_lin = 0.0; self.auto_streak = 0; self.switch_at = None; self.lin_hist = []; self.ok_hist = []; self.crit_hist = []
        # probe kept alive after the freeze (linear head trained on detached features, no extra forward) and critic re-arm bookkeeping
        self.probe_alive = self.cf_probe_alive or self.cf_rearm; self.probe_opt = None; self.rearm_streak = 0; self.switches = []
        # E31: mean compensation without stored features. During the plastic phase the SLDA sums are also accumulated in the
        # FROZEN reference space (adapter disabled, one extra forward) and an affine translator frozen -> adapted is fitted
        # online with exponential forgetting; at the freeze the class means are replaced by the translated reference means.
        if self.cf_comp == "ref" and self.head is not None:
            d = m.embed_dim
            self.ref_head = StreamSLDA(d, self.n_classes, self.device, shrink=self.cf_shrink)
            self.cG = torch.zeros(d + 1, d + 1, device=self.device); self.cH = torch.zeros(d + 1, d, device=self.device); self.comp_log = []
        self.last_comp = 0                                            # E32: samples seen at the last periodic compensation
        self.ema_ref = 0.0; self.ema_gain = 0.0; self.gain_hist = []; self.ref_eval = []; self.batch_i = 0; self._last_pair = None   # E35: paired gain (compensated head − reference head), translator residual, reference test eval
        self.t_floor = None; self.g_floor = None; self.t_event = None; self.g_freeze = None; self.t_freeze = None; self.resid_freeze = None
        self.resume_streak = 0; self.guard_streak = 0; self.guard_hit = False; self.resumed = False   # E36: growth stop / gain-drop resume / residual guard
        self.seen_idx = []; self.last_seen = {}                       # diagnostics (oracle) bookkeeping
        self.tele = {"sample": [], "d_step": [], "d_ref": []}; self.oracle_log = []
        if self.warm or self.cf_head == "linear":
            if m.linear_head is None:
                m.linear_head = nn.Linear(m.embed_dim, self.n_classes).to(self.device)
        self._set_phase_optimizer()
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        logger.info(f"[CFLoRA] head={self.cf_head} lr={self.lr} warm={self.cf_warm}@{self.cf_warm_lr} after={self.cf_after} "
                    f"plastic={self.plastic} lora_params={sum(p.numel() for p in self.lora_params)}")

    def _set_phase_optimizer(self):
        m = self.model_without_ddp
        if self.in_warm:
            params = self.lora_params + list(m.linear_head.parameters())
            self.optimizer = torch.optim.Adam(params, lr=self.cf_warm_lr); self.plastic = True
        elif self.cf_head == "linear":
            params = self.lora_params + list(m.linear_head.parameters())
            self.optimizer = torch.optim.AdamW(params, lr=max(self.lr, 1e-12), weight_decay=self.cf_wd); self.plastic = self.lr > 0
        else:
            lr = self.lr if self.cf_after == "slow" else 0.0
            self.optimizer = torch.optim.AdamW(self.lora_params, lr=max(lr, 1e-12), weight_decay=self.cf_wd) if self.lora_params else None
            self.plastic = lr > 0 and len(self.lora_params) > 0
            if self.probe_alive and m.linear_head is not None:
                self.probe_opt = torch.optim.Adam(m.linear_head.parameters(), lr=self.cf_warm_lr)

    def _feats(self, x, grad: bool):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            return self.model_without_ddp.features(x).float()

    def _logits(self, z, force_closed: bool = False):
        if self.head is not None and (force_closed or not self.in_warm or self.cf_comp_every > 0):   # E32: periodic compensation keeps the closed head usable while plastic
            return self.head.logits(z) + self.mask
        return self.model_without_ddp.linear_head(z) + self.mask

    # ------------------------------------------------------------------ online step
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        cap_hit = self.cf_warm > 0 and self.seen_samples >= self.cf_warm
        if self.cf_auto_patience_per_class > 0:
            n_ref = self.n_classes if self.cf_auto_scale == "total" else max(len(self.exposed_classes), 1)
            patience = max(int(self.cf_auto_patience), int(round(self.cf_auto_patience_per_class * n_ref)))   # floor at the fixed patience
        else:
            patience = self.cf_auto_patience
        if self.cf_auto_tol < 1.0:   # windowed: fraction of the last `patience` batches with closed >= plastic (after cf_auto_min samples)
            auto_hit = self.cf_warm_auto and len(self.ok_hist) >= patience and (sum(self.ok_hist[-patience:]) / patience) >= self.cf_auto_tol
        else:
            auto_hit = self.cf_warm_auto and self.auto_streak >= patience
        if self.in_warm and self.cf_comp == "ref" and self.cf_comp_every > 0 and self.head is not None and self.head.N > 0 \
                and self.seen_samples - self.last_comp >= self.cf_comp_every:            # E32: periodic compensation while plastic
            self._compensate_means(); self.last_comp = self.seen_samples
        reason = "auto" if auto_hit else "cap"
        if self.in_warm and self.cf_stop == "gain":                      # E36 growth rule: the critic only sets the floor
            if auto_hit and self.t_floor is None:
                self.t_floor = self.seen_samples; self.g_floor = self.ema_gain; self.t_event = None
                logger.info(f"[CFLoRA][gain] floor at {self.seen_samples} (critic), gain {self.ema_gain:+.3f}")
            auto_hit = False
            if self.t_floor is not None:
                if self.t_event is None and self.ema_gain - self.g_floor >= 2 * self.cf_gain_eps:
                    self.t_event = self.seen_samples                       # novelty seen on the signal (new classes or a new domain)
                if self.t_event is not None and self.seen_samples >= self.t_event + self.cf_gain_h and self._gain_slope() * self.cf_gain_h <= self.cf_gain_eps:
                    auto_hit = True; reason = "growth"
                elif self.t_event is None and self.seen_samples >= self.t_floor + 2 * self.cf_gain_h:
                    auto_hit = True; reason = "fallback"                   # nothing left to gain after the floor
            if self.guard_hit:
                auto_hit = True; reason = "guard"
        if self.in_warm and (cap_hit or auto_hit):
            if self.cf_comp == "ref" and self.head is not None:
                self._compensate_means()
            self.in_warm = False; self._set_phase_optimizer(); self.switches.append((self.seen_samples, f"freeze:{reason}" if self.cf_stop == "gain" else "freeze"))
            self.g_freeze = self.ema_gain; self.t_freeze = self.seen_samples; self.resume_streak = 0; self.guard_hit = False; self.guard_streak = 0
            self.resid_freeze = self.comp_log[-1].get("resid") if getattr(self, "comp_log", None) else None
            if self.switch_at is None:
                self.switch_at = self.seen_samples
            logger.info(f"[CFLoRA] warm phase over after {self.seen_samples} samples ({reason}) → closed-form head, after={self.cf_after}")
        elif not self.in_warm and self.cf_resume == "gain" and self.resume_streak >= self.cf_resume_patience:
            # E36 resume: the frozen representation lost >= cf_resume_drop of its paired gain over the reference for P sampled batches
            self.in_warm = True; self._set_phase_optimizer(); self.last_comp = self.seen_samples; self.resumed = True
            self.t_floor = self.seen_samples; self.g_floor = self.ema_gain; self.t_event = None; self.resume_streak = 0; self.guard_streak = 0; self.guard_hit = False
            self.auto_streak = 0; self.ok_hist = []
            self.switches.append((self.seen_samples, "resume:gain_drop"))
            logger.info(f"[CFLoRA][gain] resume after {self.seen_samples} samples: gain {self.ema_gain:+.3f} vs {self.g_freeze:+.3f} at the freeze → plastic phase")
        elif not self.in_warm and self.cf_rearm and self.rearm_streak >= (self.cf_rearm_patience or patience):
            # re-arm: the probe has been ahead of the closed-form head by >= cf_rearm_margin for P batches → representation no longer adequate
            self.in_warm = True; self._set_phase_optimizer(); self.auto_streak = 0; self.ok_hist = []; self.rearm_streak = 0
            self.switches.append((self.seen_samples, "rearm"))
            logger.info(f"[CFLoRA] re-arm after {self.seen_samples} samples: probe EMA {self.ema_lin:.3f} vs closed {self.ema_closed:.3f} → plastic phase")
        self.seen_samples += len(labels); self.batch_i += 1
        if self.cf_oracle:
            self.seen_idx.append(idx.clone())
            for c in labels.unique().tolist():
                self.last_seen[self.exposed_classes.index(c)] = self.seen_samples
        y = labels.clone()
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        x = images.to(self.device, non_blocking=True); y = y.to(self.device, non_blocking=True)
        self.model.train()
        n_iter = int(self.online_iter) if self.plastic else 1
        loss_sum = 0.0; acc = 0.0
        for it in range(n_iter):
            # frozen phase with --cf_clean_stats: no gradient is needed, so the single forward runs on the clean view (E31a)
            xa = self.test_transform_tensor(x) if (self.cf_clean_stats and not self.plastic) else self.train_transform(x.clone())
            z_stat = None; z_ref = None
            want_ref = self.cf_comp == "ref" and (self.in_warm or (self.cf_ref_stride > 0 and self.batch_i % self.cf_ref_stride == 0))   # E35: reference forward kept alive after the freeze at a stride
            if it == 0 and ((self.plastic and self.cf_clean_stats) or want_ref):
                with torch.no_grad():
                    xs = (self.test_transform_tensor(x) if self.cf_clean_stats else xa) if self.plastic else xa   # view used for the statistics (frozen phase: xa is already the clean view)
                    if want_ref:
                        self.model_without_ddp.lora_enabled(False); z_ref = self._feats(xs, grad=False); self.model_without_ddp.lora_enabled(True)
                    if self.cf_clean_stats and self.plastic:
                        z_stat = self._feats(xs, grad=False)
            z = self._feats(xa, grad=self.plastic)
            if z_stat is None:
                z_stat = z.detach()
            if it == 0:
                xa0 = xa; z0 = z.detach()
                with torch.no_grad():
                    logit0 = self._logits(z.detach()) if (self.head is None or self.head.N > 0) else None
                    acc = (logit0.argmax(1) == y).float().mean().item() if logit0 is not None else 0.0   # prediction BEFORE learning
                    if self.in_warm and self.cf_warm_auto and self.head is not None and self.head.N > 0:
                        zc = z_stat if self.cf_auto_clean else z.detach()   # E33b: critic on the clean view (same view as the statistics) when requested
                        a_c = (self._logits(zc, force_closed=True).argmax(1) == y).float().mean().item()
                        a_l = ((self.model_without_ddp.linear_head(zc) + self.mask).argmax(1) == y).float().mean().item()   # always the linear head (E32: _logits may be closed)
                        self.ema_closed = 0.95 * self.ema_closed + 0.05 * a_c; self.ema_lin = 0.95 * self.ema_lin + 0.05 * a_l
                        self.lin_hist.append(self.ema_lin)
                        plateau = True
                        if self.cf_auto_plateau > 0:
                            H = int(self.cf_auto_horizon); plateau = len(self.lin_hist) > H and (self.ema_lin - self.lin_hist[-H - 1]) < self.cf_auto_plateau
                        ok = self.seen_samples >= self.cf_auto_min and self.ema_closed >= self.ema_lin - self.cf_auto_margin and plateau
                        self.crit_hist.append((self.seen_samples, round(self.ema_closed, 4), round(self.ema_lin, 4), int(ok)))
                        self.auto_streak = self.auto_streak + 1 if ok else 0
                        if self.seen_samples >= self.cf_auto_min:
                            self.ok_hist.append(1 if ok else 0)
                    elif not self.in_warm and self.probe_alive and self.head is not None and self.head.N > 0:
                        # after the freeze: closed head predicts (acc), the probe is scored on the same features; classes with few
                        # samples are left out of the gap so that brand-new classes cannot trigger a re-arm by themselves
                        sel = self.head.n[y] >= self.cf_rearm_min_count
                        if sel.any():
                            a_l = (self.model_without_ddp.linear_head(z.detach())[sel].add(self.mask).argmax(1) == y[sel]).float().mean().item()
                            a_c = (logit0[sel].argmax(1) == y[sel]).float().mean().item()
                            self.ema_closed = 0.95 * self.ema_closed + 0.05 * a_c; self.ema_lin = 0.95 * self.ema_lin + 0.05 * a_l
                        bad = self.ema_lin >= self.ema_closed + self.cf_rearm_margin
                        self.rearm_streak = self.rearm_streak + 1 if bad else 0
                        self.crit_hist.append((self.seen_samples, round(self.ema_closed, 4), round(self.ema_lin, 4), int(bad), 0))
                    if z_ref is not None and self.head is not None and self.head.N > 0 and self.ref_head.N > 0:
                        # E35: paired prequential gain of the (compensated) closed head on the adapted features over the reference
                        # head on the frozen backbone, same batch, same counts, clean view, statistics BEFORE learning the batch
                        a_g = (self._logits(z_stat, force_closed=True).argmax(1) == y).float().mean().item()
                        a_r = ((self.ref_head.logits(z_ref) + self.mask).argmax(1) == y).float().mean().item()
                        self.ema_ref = 0.95 * self.ema_ref + 0.05 * a_r; self.ema_gain = 0.95 * self.ema_gain + 0.05 * (a_g - a_r)
                        self.gain_hist.append((self.seen_samples, round(a_g, 4), round(a_r, 4), round(self.ema_gain, 4), int(self.in_warm)))
                        if not self.in_warm and self.cf_resume == "gain" and self.g_freeze is not None and self.seen_samples >= (self.t_freeze or 0) + self.cf_gain_h:
                            self.resume_streak = self.resume_streak + 1 if self.ema_gain <= self.g_freeze - self.cf_resume_drop else 0   # E36
            if self.head is not None and not self.in_warm:
                if self.head.N > 0 and self.plastic:
                    loss = F.cross_entropy(self._logits(z), y)                    # head detached (statistics), gradient into LoRA
                else:
                    loss = None
            else:                                                                  # warm phase or trained-head ablation
                logit = self.model_without_ddp.linear_head(z)
                bm = torch.full_like(self.mask, -torch.inf); bm[torch.unique(y)] = 0    # batch logit mask (Si-Blurry practice)
                loss = F.cross_entropy(logit + (bm if not self.no_batchmask else self.mask), y)
            if it == 0 and self.head is not None:
                self.head.learn(z_stat, y)                                         # statistics from pre-update features
                if z_ref is not None:                                                 # reference-space statistics + translator pairs
                    self.ref_head.learn(z_ref, y); self._last_pair = (z_ref, z_stat)   # E35: last pair kept for the translator residual
                    A = torch.cat([z_ref, torch.ones(len(z_ref), 1, device=self.device)], 1)
                    self.cG.mul_(self.cf_comp_decay).add_(A.T @ A); self.cH.mul_(self.cf_comp_decay).add_(A.T @ z_stat)
            if it == 0 and not self.in_warm and self.probe_opt is not None:        # probe: linear head on detached features, no extra forward
                bm = torch.full_like(self.mask, -torch.inf); bm[torch.unique(y)] = 0
                loss_p = F.cross_entropy(self.model_without_ddp.linear_head(z.detach()) + bm, y)
                self.probe_opt.zero_grad(set_to_none=True); loss_p.backward(); self.probe_opt.step()
            if loss is not None and self.plastic:
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g['params']], 1.0)
                self.optimizer.step(); loss_sum += loss.item()
            if not self.plastic:
                break
        if self.cf_telemetry and self.plastic:
            with torch.no_grad():
                za = self._feats(xa0, grad=False)
                self.model_without_ddp.lora_enabled(False); zr = self._feats(xa0, grad=False); self.model_without_ddp.lora_enabled(True)
            self.tele["sample"].append(self.seen_samples); self.tele["d_step"].append((za - z0).norm(dim=1).mean().item())
            self.tele["d_ref"].append((za - zr).norm(dim=1).mean().item())
        del x, y; gc.collect()
        return loss_sum / max(n_iter, 1), acc

    @torch.no_grad()
    def _gain_slope(self):
        """E36: OLS slope (per sample) of the EMA paired gain over the last cf_gain_h samples; inf when the window is too short."""
        rows = [r for r in self.gain_hist if r[0] > self.seen_samples - self.cf_gain_h]
        if len(rows) < 3:
            return float("inf")
        x = torch.tensor([r[0] for r in rows], dtype=torch.float64); y = torch.tensor([r[3] for r in rows], dtype=torch.float64)
        x = x - x.mean(); return float((x * (y - y.mean())).sum() / (x * x).sum().clamp(min=1e-9))

    def _compensate_means(self):
        """E31: replace the class means accumulated during the plastic phase (a mixture of intermediate adapter states) with
        the reference-space means pushed through the affine translator fitted on the most recent batches. Covariance kept."""
        d = self.model_without_ddp.embed_dim
        W = torch.linalg.solve(self.cG + self.cf_comp_ridge * torch.eye(d + 1, device=self.device), self.cH)     # (d+1, d)
        seen = self.head.n > 0
        mu0 = self.ref_head.s / self.ref_head.n.clamp(min=1).unsqueeze(1)
        mu_new = torch.cat([mu0, torch.ones(len(mu0), 1, device=self.device)], 1) @ W
        mu_old = self.head.s / self.head.n.clamp(min=1).unsqueeze(1)
        shift = (mu_new - mu_old)[seen].norm(dim=1).mean().item(); scale = mu_old[seen].norm(dim=1).mean().item()
        self.head.s[seen] = mu_new[seen] * self.head.n[seen].unsqueeze(1); self.head._Wb = None
        resid = None
        if self._last_pair is not None:   # E35: fraction of the adapted-feature variance the affine translator leaves unexplained (non-affine drift)
            zr, zs = self._last_pair; pred = torch.cat([zr, torch.ones(len(zr), 1, device=self.device)], 1) @ W
            resid = (((zs - pred) ** 2).sum(1).mean() / ((zs - zs.mean(0)) ** 2).sum(1).mean().clamp(min=1e-8)).item()
            if self.cf_resid_guard > 0 and self.resumed and self.in_warm and self.resid_freeze:   # E36 stability guard in a resumed phase
                self.guard_streak = self.guard_streak + 1 if resid >= self.cf_resid_guard * self.resid_freeze else 0
                if self.guard_streak >= 2:
                    self.guard_hit = True
        self.comp_log.append({"samples": self.seen_samples, "n_seen": int(seen.sum()), "mean_shift": shift, "mean_norm": scale, "resid": resid})
        logger.info(f"[CFLoRA][comp] means compensated at {self.seen_samples}: {int(seen.sum())} classes, mean shift {shift:.3f} (mean norm {scale:.3f})")

    # ------------------------------------------------------------------ evaluation
    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes); num_data_l = torch.zeros(self.n_classes)
        if self.head is not None:
            self.head.refresh()
        ref_on = self.cf_diag_ref_eval and self.cf_comp == "ref" and self.head is not None and self.ref_head.N > 0; correct_r = 0.0   # E35 diagnostic
        if ref_on:
            self.ref_head.refresh()
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                x = x.to(self.device, non_blocking=True); y = y.to(self.device, non_blocking=True)
                logit = self._logits(self._feats(x, grad=False))
                loss = F.cross_entropy(logit, y)
                pred = logit.argmax(1)
                total_correct += (pred == y).sum().item(); total_num_data += y.size(0); total_loss += loss.item()
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.cpu(); num_data_l += xlabel_cnt.cpu()
                if ref_on:
                    self.model_without_ddp.lora_enabled(False); zr = self._feats(x, grad=False); self.model_without_ddp.lora_enabled(True)
                    correct_r += ((self.ref_head.logits(zr) + self.mask).argmax(1) == y).sum().item()
        if ref_on:
            self.ref_eval.append({"samples": self.seen_samples, "acc": total_correct / max(total_num_data, 1), "acc_ref": correct_r / max(total_num_data, 1), "in_warm": int(self.in_warm)})
        return {"avg_loss": total_loss / max(len(test_loader), 1), "avg_acc": total_correct / max(total_num_data, 1),
                "cls_acc": (correct_l / (num_data_l + 1e-5)).numpy().tolist()}

    def online_before_task(self, task_id):
        self.task_id = task_id            # bookkeeping only; the method never uses task identity

    def _rendition_breakdown(self):
        """DIAGNOSTIC (final task only): accuracy per style/domain of the test set, split into styles seen in the stream vs unseen.
        Needs an ImageFolder-like test set whose file names start with '<style>_'."""
        base = self.test_dataset
        if not hasattr(base, "samples"):
            return
        if "domainnet" in self.dataset:      # root/<domain>/<class>/<file>: quickdraw files carry no domain prefix
            rend = [os.path.normpath(p).split(os.sep)[-3] for p, _ in base.samples]
        else:
            rend = [os.path.basename(p).split("_")[0] for p, _ in base.samples]
        seen_styles = set(getattr(self.train_sampler, "sequence", []))
        loader = DataLoader(base, batch_size=self.batchsize * 2, shuffle=False, num_workers=self.n_worker)
        exposed = set(self.exposed_classes); correct = {}; total = {}; i = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                keep = torch.tensor([int(v) in exposed for v in y])
                yy = torch.tensor([self.exposed_classes.index(int(v)) if int(v) in exposed else -1 for v in y], device=self.device)
                pred = self._logits(self._feats(x.to(self.device, non_blocking=True), grad=False)).argmax(1)
                for j in range(len(y)):
                    r = rend[i + j]
                    if keep[j]:
                        total[r] = total.get(r, 0) + 1; correct[r] = correct.get(r, 0) + int(pred[j].item() == yy[j].item())
                i += len(y)
        acc = {r: correct[r] / total[r] for r in total}
        def agg(keys):
            n = sum(total[r] for r in keys); return (sum(correct[r] for r in keys) / n) if n else None
        out = {"per_style": acc, "n_per_style": total, "seen_styles": sorted(seen_styles),
               "acc_seen_styles": agg([r for r in total if r in seen_styles]), "acc_unseen_styles": agg([r for r in total if r not in seen_styles])}
        with open(os.path.join(self.log_dir, f"seed_{self.rnd_seed}_rend_acc.json"), "w") as f:
            json.dump(out, f)
        logger.info(f"[CFLoRA][styles] seen {out['acc_seen_styles']} unseen {out['acc_unseen_styles']} per-style {{{', '.join(f'{k}: {v:.3f}' for k, v in sorted(acc.items()))}}}")

    def online_after_task(self, cur_iter):
        if self.task_id == self.n_tasks - 1 and getattr(self, "stream", "siblurry") == "rendition":
            try:
                self._rendition_breakdown()
            except Exception as e:
                logger.exception(f"[CFLoRA] rendition breakdown failed: {e}")
        if self.cf_oracle and self.head is not None and isinstance(self.head, StreamSLDA):
            try:
                self._oracle_diagnostic()
            except Exception as e:  # diagnostics must never kill the run
                logger.exception(f"[CFLoRA] oracle diagnostic failed: {e}")
        if self.task_id == self.n_tasks - 1 and (self.crit_hist or self.gain_hist):   # critic trace (closed vs plastic EMA prequential acc), always saved
            with open(os.path.join(self.log_dir, f"seed_{self.rnd_seed}_critic.json"), "w") as f:
                json.dump({"switch_at": self.switch_at, "margin": self.cf_auto_margin, "switches": self.switches,
                           "rearm_margin": self.cf_rearm_margin, "hist": self.crit_hist,   # hist rows: (samples, ema_closed, ema_lin, flag[, 0 = after freeze])
                           "gain": self.gain_hist}, f)                                     # E35 gain rows: (samples, a_comp, a_ref, ema_gain, in_warm)
        if self.cf_telemetry or self.cf_oracle or self.cf_comp != "none":
            with open(os.path.join(self.log_dir, f"seed_{self.rnd_seed}_diag.json"), "w") as f:
                json.dump({"switch_at": self.switch_at, "telemetry": self.tele, "oracle": self.oracle_log, "comp": getattr(self, "comp_log", []),
                           "ref_eval": self.ref_eval}, f)
        self.model_without_ddp.process_task_count(); self.task_id += 1; gc.collect()

    # ------------------------------------------------------------------ DIAGNOSTIC (image replay, never used by the method)
    def _eval_with_head(self, head):
        saved = self.head; self.head = head; saved_warm = self.in_warm; self.in_warm = False
        sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
        loader = DataLoader(self.test_dataset, batch_size=self.batchsize * 2, sampler=sampler, num_workers=self.n_worker)
        acc = self.online_evaluate(loader)["avg_acc"]
        self.head = saved; self.in_warm = saved_warm
        return acc

    def _oracle_diagnostic(self):
        m = self.model_without_ddp; m.eval()
        idx = torch.cat(self.seen_idx).unique()
        sub = torch.utils.data.Subset(self.train_dataset, idx.tolist())
        loader = DataLoader(sub, batch_size=256, shuffle=False, num_workers=self.n_worker)
        h_aug = StreamSLDA(m.embed_dim, self.n_classes, self.device, shrink=self.cf_shrink)
        h_clean = StreamSLDA(m.embed_dim, self.n_classes, self.device, shrink=self.cf_shrink)
        with torch.no_grad():
            for x, yraw in loader:
                y = torch.tensor([self.exposed_classes.index(int(v)) for v in yraw], device=self.device)
                x = x.to(self.device, non_blocking=True)
                h_aug.learn(self._feats(self.train_transform(x.clone()), False), y)
                h_clean.learn(self._feats(self.test_transform_tensor(x), False), y)
        acc_stream = self._eval_with_head(self.head); acc_aug = self._eval_with_head(h_aug); acc_clean = self._eval_with_head(h_clean)
        # affine drift analysis on class means: stale (streaming) → true (oracle-aug); fit on recent classes, test on old ones
        n = self.head.n; seen = (n > 0)
        mu_stale = self.head.s / n.clamp(min=1).unsqueeze(1); mu_true = h_aug.s / h_aug.n.clamp(min=1).unsqueeze(1)
        recent = torch.zeros_like(seen)
        for c, t in self.last_seen.items():
            recent[c] = (self.seen_samples - t) <= 5000
        old = seen & ~recent; rec = seen & recent
        out = {"samples": self.seen_samples, "n_seen": int(seen.sum()), "n_old": int(old.sum()), "n_recent": int(rec.sum()),
               "acc_stream": acc_stream, "acc_oracle_aug": acc_aug, "acc_oracle_clean": acc_clean}
        if old.sum() > 0 and rec.sum() >= 3:
            X = mu_stale[rec]; Y = mu_true[rec]; d = X.shape[1]
            base = ((mu_true[old] - mu_stale[old]) ** 2).sum().item()
            def comp_acc(mu_comp):
                h = StreamSLDA(m.embed_dim, self.n_classes, self.device, shrink=self.cf_shrink)
                h.s = self.head.s.clone(); h.n = self.head.n.clone(); h.S = self.head.S.clone(); h.N = self.head.N
                h.s[old] = mu_comp[old] * h.n[old].unsqueeze(1); return self._eval_with_head(h)
            # translation only
            b = (Y - X).mean(0); mu_t = mu_stale + b
            out["frac_removed_translation"] = 1 - ((mu_true[old] - mu_t[old]) ** 2).sum().item() / max(base, 1e-9)
            out["acc_comp_translation"] = comp_acc(mu_t)
            # ridge affine  Y - X ≈ X M + b
            Xc = X - X.mean(0); Dc = (Y - X) - (Y - X).mean(0)
            for lam in (1e1, 1e3):
                M = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(d, device=self.device), Xc.T @ Dc)
                mu_a = mu_stale + (mu_stale - X.mean(0)) @ M + (Y - X).mean(0)
                out[f"frac_removed_affine_l{lam:g}"] = 1 - ((mu_true[old] - mu_a[old]) ** 2).sum().item() / max(base, 1e-9)
                out[f"acc_comp_affine_l{lam:g}"] = comp_acc(mu_a)
            out["stale_dist_old"] = (mu_true[old] - mu_stale[old]).norm(dim=1).mean().item()
            out["stale_dist_recent"] = (mu_true[rec] - mu_stale[rec]).norm(dim=1).mean().item()
            # attribution: which stale statistic costs accuracy? (oracle means + stale Sigma; stale means + oracle Sigma)
            def acc_with(s_, S_, N_):
                h = StreamSLDA(m.embed_dim, self.n_classes, self.device, shrink=self.cf_shrink)
                h.s = s_.clone(); h.n = self.head.n.clone(); h.S = S_.clone(); h.N = N_; return self._eval_with_head(h)
            out["acc_oracle_means_stale_cov"] = acc_with(mu_true * self.head.n.unsqueeze(1), self.head.S, self.head.N)
            out["acc_stale_means_oracle_cov"] = acc_with(self.head.s, h_aug.S * (self.head.N / max(h_aug.N, 1)), self.head.N)
            # affine compensation fitted on recent classes, applied to ALL seen classes (upper bound for A1/A2-style compensation)
            M = torch.linalg.solve(Xc.T @ Xc + 1e1 * torch.eye(d, device=self.device), Xc.T @ Dc)
            mu_all = mu_stale + (mu_stale - X.mean(0)) @ M + (Y - X).mean(0)
            out["acc_comp_affine_all_l10"] = acc_with(mu_all * self.head.n.unsqueeze(1), self.head.S, self.head.N)
            out["acc_comp_translation_all"] = acc_with((mu_stale + b) * self.head.n.unsqueeze(1), self.head.S, self.head.N)
        self.oracle_log.append(out); logger.info(f"[CFLoRA][oracle] {out}")
