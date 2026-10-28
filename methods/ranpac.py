import copy
import gc
import logging

import os
import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class RanPAC(_Trainer):
    def __init__(self, *args, **kwargs):
        super(RanPAC, self).__init__(*args, **kwargs)

        self.task_id = 0
        self.first_task_completed = False
        self.seen_idx = []; self.arr_s = None; self.arr_n = None                  # E55c: stream indices + per-class clean mean at arrival (before learning the batch), for the feature dump
        self.first_tasks = int(getattr(self, 'ranpac_first_tasks', 1) or 1)   # length of the first session in tasks (paper: 1)
        self.first_samples = int(getattr(self, 'ranpac_first_samples', 0) or 0)  # E52: end the first session after N stream samples (0 = task mode)
        if self.first_samples:
            self.first_tasks = 10 ** 9  # sample mode: the task-boundary closure in online_after_task never fires; set to the closing task at closure
        self.seen_samples = 0

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        self.seen_samples += int(labels.shape[0])
        if self.cf_dump_feats:   # E55c: where the class sits when the head learns it (clean view, current model, eval, no grad: RNG untouched)
            self.seen_idx.append(idx.clone()); m = self.model.module if self.distributed else self.model; m.eval()
            y = torch.tensor([self.exposed_classes.index(int(v)) for v in labels], device=self.device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
                z = m.forward_features(self.test_transform_tensor(images.to(self.device, non_blocking=True))).float()
            if self.arr_s is None:
                self.arr_s = torch.zeros(self.n_classes, z.shape[1], device=self.device); self.arr_n = torch.zeros(self.n_classes, device=self.device)
            self.arr_s.index_add_(0, y, z); self.arr_n.index_add_(0, y, torch.ones(len(y), device=self.device))
        if self.first_samples and not self.first_task_completed and self.seen_samples >= self.first_samples:
            logger.info(f"Completing first session at {self.seen_samples} samples (budget {self.first_samples}), setting up random projection")
            self._complete_first_session()
            self.first_tasks = self.task_id            # the rest of the stream is collection: before/after-task hooks and evaluation see task_id >= first_tasks
        if not self.first_task_completed:
            return self._train_first_task(images, labels)
        else:
            return self._collect_features_for_statistics(images, labels)

    def _complete_first_session(self):
        m = self.model.module if self.distributed else self.model
        m.setup_rp()
        m.freeze_all_except_classifier()
        m.update_statistics_and_classifier()
        self.first_task_completed = True
        logger.info("Random projection initialized, adapters frozen")

    def _train_first_task(self, images, labels):
        _loss, _acc, _iter = 0.0, 0.0, 0

        # Train with multiple iterations as specified
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        self._collect_features_for_statistics(images, labels)

        return _loss / _iter, _acc / _iter

    def _collect_features_for_statistics(self, images, labels):
        images_copy = images.clone()
        labels_copy = labels.clone()
        
        # Map labels to exposed class indices
        for j in range(len(labels_copy)):
            labels_copy[j] = self.exposed_classes.index(labels_copy[j].item())

        images_copy = images_copy.to(self.device)
        labels_copy = labels_copy.to(self.device)

        # images_copy = self.test_transform_tensor(images_copy)
        images_copy = self.train_transform(images_copy)

        with torch.no_grad():
            self.model.eval()
            if self.distributed:
                self.model.module.collect_features_labels(images_copy, labels_copy)
            else:
                self.model.collect_features_labels(images_copy, labels_copy)

        del images_copy, labels_copy
        gc.collect()
        return 0.0, 0.0  # No training loss/acc for subsequent tasks

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)

        self.optimizer.zero_grad()
        if not self.no_batchmask:
            logit, loss = self.model_forward(x,y,mask=logit_mask)
        else:
            logit, loss = self.model_forward(x,y)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        if not end and self.task_id >= self.first_tasks:
            if self.distributed:
                saved_state = self.model.module.save_classifier_state()
            else:
                saved_state = self.model.save_classifier_state()

        if self.task_id >= self.first_tasks and self.first_task_completed:
            if self.distributed:
                self.model.module.update_statistics_and_classifier()
            else:
                self.model.update_statistics_and_classifier()

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                
                # Map labels to exposed class indices
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x)
                logit = logit + self.mask
                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        if not end and self.task_id >= self.first_tasks:
            if self.distributed:
                self.model.module.restore_classifier_state(saved_state)
            else:
                self.model.restore_classifier_state(saved_state)

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def online_before_task(self, task_id):
        if task_id < self.first_tasks:
            if not self.distributed:
                if self.model.use_g_prompt:
                    self.model.freeze_backbone_except_prompts()
                    logger.info("First task: g-prompts and classifier enabled for training")
                else:
                    self.model.freeze_backbone_except_adapters()
                    logger.info("First task: adapters and classifier enabled for training")
            else:
                if self.model.module.use_g_prompt:
                    self.model.module.freeze_backbone_except_prompts()
                    logger.info("First task: g-prompts and classifier enabled for training")
                else:
                    self.model.module.freeze_backbone_except_adapters()
                    logger.info("First task: adapters and classifier enabled for training")
        else:
            if not self.distributed:
                self.model.freeze_all_except_classifier()
            else:
                self.model.module.freeze_all_except_classifier()
            
            if not self.distributed:
                mode = "g-prompt" if self.model.use_g_prompt else "adapter"
            else:
                mode = "g-prompt" if self.model.module.use_g_prompt else "adapter"
            logger.info(f"Task {task_id} ({mode} mode): collecting features for RanPAC statistics")

    def online_after_task(self, cur_iter):
        if self.cf_dump_feats and self.task_id == self.n_tasks - 1:
            try:
                self._dump_feats()
            except Exception as e:
                logger.exception(f"[RanPAC] feature dump failed: {e}")
        if self.task_id == self.first_tasks - 1 and not self.first_task_completed:
            logger.info("Completing first task training, setting up random projection")
            self._complete_first_session()

        if not self.distributed:
            self.model.process_task_count()
        else:
            self.model.module.process_task_count()
        
        logger.info(f"Task {self.task_id} completed, statistics updated")
        self.task_id += 1

    def _dump_feats(self):
        """E55c: clean features of the seen train stream and of the test set under the final (frozen) model, original labels; head = arrival-time
        clean means (s/n, rows in exposure order), the RanPAC prototypes Q (rows in exposure order, same counts n) and W_rand. Same layout as E55/E55b."""
        m = self.model.module if self.distributed else self.model; m.update_statistics_and_classifier(); m.eval(); first = {}
        for t, i in enumerate(torch.cat(self.seen_idx).tolist()):
            first.setdefault(i, t)
        uidx = sorted(first, key=first.get); C = len(self.exposed_classes); cl = m.classifier
        out = {"exposed": list(self.exposed_classes), "head": {"s": self.arr_s.cpu(), "n": self.arr_n.cpu(), "Q": cl.Q[:, :C].T.detach().float().cpu(),
                                                              "W_rand": cl.W_rand.detach().float().cpu(), "first_samples": self.first_samples}}
        sets = {"train": (torch.utils.data.Subset(self.train_dataset, uidx), self.test_transform_tensor), "test": (self.test_dataset, None)}
        with torch.no_grad():
            for name, (ds, tf) in sets.items():
                zs, ys = [], []
                for x, y in torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=self.n_worker):
                    x = x.to(self.device, non_blocking=True); x = tf(x) if tf is not None else x
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        zs.append(m.forward_features(x).float().half().cpu())
                    ys.append(y)
                out[name] = {"z": torch.cat(zs), "y": torch.cat(ys)}
        out["train"]["idx"] = torch.tensor(uidx); out["train"]["first"] = torch.tensor([first[i] for i in uidx])
        torch.save(out, os.path.join(self.log_dir, f"seed_{self.rnd_seed}_feats.pt"))
        logger.info(f"[RanPAC] features dumped: train {len(uidx)}, test {len(out['test']['y'])}")
