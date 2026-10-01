"""SinglePrompt trainer (Park, Lee, Lee, CVPR Findings 2026): task-free, one forward per iteration.
Per batch: `online_iter` Adam steps, each on a fresh augmented view, CE with the mini-batch logit mask (Si-Blurry practice).
No task identity is used anywhere (online_before/after_task are no-ops). Port of their methods/default.py."""
import gc
import logging

import torch
import torch.nn.functional as F

from methods._trainer import _Trainer

logger = logging.getLogger()


class SinglePromptTrainer(_Trainer):
    def setup_distributed_model(self):
        super().setup_distributed_model()
        params = self.model_without_ddp.trainable_parameters()
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        logger.info(f"[SinglePrompt] blocks={self.sp_blocks} len={self.sp_len} tau={self.sp_tau} lr={self.lr} "
                    f"trainable_params={sum(p.numel() for p in params)}")

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        y = torch.tensor([self.exposed_classes.index(int(v)) for v in labels], device=self.device)
        x = images.to(self.device, non_blocking=True)
        bm = torch.full_like(self.mask, -torch.inf); bm[y.unique()] = 0     # batch logit mask
        self.model.train()
        n_iter = int(self.online_iter); loss_sum = 0.0; acc_sum = 0.0
        for _ in range(n_iter):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit = self.model(self.train_transform(x.clone())) + (self.mask if self.no_batchmask else bm)
                loss = F.cross_entropy(logit, y)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
            loss_sum += loss.item(); acc_sum += (logit.argmax(1) == y).float().mean().item()
        del x, y; gc.collect()
        return loss_sum / n_iter, acc_sum / n_iter

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes); num_data_l = torch.zeros(self.n_classes)
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                y = torch.tensor([self.exposed_classes.index(int(v)) for v in y], device=self.device)
                x = x.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logit = self.model(x) + self.mask
                loss = F.cross_entropy(logit.float(), y)
                pred = logit.argmax(1)
                total_correct += (pred == y).sum().item(); total_num_data += y.size(0); total_loss += loss.item()
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.cpu(); num_data_l += xlabel_cnt.cpu()
        return {"avg_loss": total_loss / max(len(test_loader), 1), "avg_acc": total_correct / max(total_num_data, 1),
                "cls_acc": (correct_l / (num_data_l + 1e-5)).numpy().tolist()}

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass
