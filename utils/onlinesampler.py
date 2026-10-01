import logging
from typing import Optional, Sized

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

logger = logging.getLogger()

class OnlineSampler(DistributedSampler):
    def __init__(self, data_source: Optional[Sized], num_tasks, m, n, rnd_seed, cur_iter = 0, varing_NM = False, num_replicas=None, rank=None) -> None:

        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.generator  = torch.Generator().manual_seed(rnd_seed)
        
        self.n  = n
        self.m  = m
        self.varing_NM = varing_NM
        self.task = cur_iter

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
            logger.info(f'num_replicas: {num_replicas}')
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()
            

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        # self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            logger.info(f"disjoint classes {self.disjoint_num}: {self.disjoint_classes}")
            logger.info(f"blurry classes {self.blurry_num}: {self.blurry_classes}")
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            logger.info(f"blurry indices: {len(blurred)}")
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                logger.info(f"task {i}: disjoint {len(self.disjoint_indices[i])}, blurry {len(self.blurry_indices[i])}")
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                if n > 100:
                    num_cls_per_task = int(len(self.classes) / num_tasks)
                    self.disjoint_classes = [[i for i in range(j*num_cls_per_task, (j+1)*num_cls_per_task)] for j in range(num_tasks)]
                else:
                    self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                    self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            if self.blurry_num > 0:
                if m > 100:
                    # assert len(self.classes) == num_tasks
                    self.blurry_classes = [[i] for i in range(len(self.classes))]
                else:
                    self.blurry_slice = [0] + torch.randint(0, self.blurry_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.blurry_num]
                    self.blurry_classes = [class_order[self.disjoint_num + self.blurry_slice[i]:self.disjoint_num + self.blurry_slice[i + 1]].tolist() for i in range(num_tasks)]
            else:
                self.blurry_classes = [[] for _ in range(num_tasks)]
            # self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            # self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            logger.info(f"disjoint classes: {self.disjoint_classes}")
            logger.info(f"blurry classes: {self.blurry_classes}")
            
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100

            if m > 100:
                blurred_101 = []
                # Split each class into subset: class 1, 100 subset
                for i in range(num_tasks):
                    num_subset = num_tasks - i
                    num_samples = len(self.blurry_indices[i])
                    ratio = 0.75
                    start = int(num_samples*ratio)
                    temp = torch.randint(start, num_samples, (num_subset-1,), generator=self.generator).sort().values.tolist()
                    split_idx = [0] + temp + [num_samples]
                    split = [self.blurry_indices[i][split_idx[j]:split_idx[j + 1]] for j in range(num_subset)]
                    blurred_101.append(split)
                    
                for i in range(num_tasks):
                    task_indices = []
                    for j in range(0, i+1):
                        task_indices += blurred_101[j][i-j]
                    self.blurry_indices[i] = task_indices
                

            elif num_blurred > 0:
                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                logger.info(f"blurry indices: {len(blurred)}")
                # num_blurred = len(blurred) // num_tasks
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                logger.info(f"task {i}: disjoint {len(self.disjoint_indices[i])}, blurry {len(self.blurry_indices[i])}")
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self):
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self):
        return self.num_selected_samples

    def set_task(self, cur_iter):

        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))
    
    def get_task(self, cur_iter):
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

class OnlineBatchSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], num_tasks, m, n, rnd_seed, 
                 batchsize=16, online_iter=1, cur_iter=0, varing_NM=False, num_replicas=None, rank=None) -> None:
        super().__init__(data_source)
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.num_tasks  = num_tasks
        self.m      = m
        self.n      = n
        self.rnd_seed   = rnd_seed
        self.batchsize  = batchsize
        self.online_iter    = online_iter
        self.cur_iter   = cur_iter
        self.varing_NM  = varing_NM

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            logger.info(f"disjoint classes: {self.disjoint_classes}")
            logger.info(f"blurry classes: {self.blurry_classes}")
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            logger.info(f"blurry indices: {len(blurred)}")
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                logger.info(f"task {i}: disjoint {len(self.disjoint_indices[i])}, blurry {len(self.blurry_indices[i])}")
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)]
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            logger.info(f"disjoint classes: {self.disjoint_classes}")
            logger.info(f"blurry classes: {self.blurry_classes}")
            
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100
            num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            logger.info(f"blurry indices: {len(blurred)}")
            # num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                logger.info(f"task {i}: disjoint {len(self.disjoint_indices[i])}, blurry {len(self.blurry_indices[i])}")
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()

    def __iter__(self):
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self):
        return self.num_selected_samples

    def set_task(self, cur_iter):

        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))
    
    def get_task(self, cur_iter):
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

    def get_task_classes(self, cur_iter):
        return list(set(self.classes[self.indices[cur_iter]]))


class OnlineTestSampler(DistributedSampler):
    def __init__(self, data_source: Optional[Sized], exposed_class, num_replicas=None, rank=None) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.exposed_class  = exposed_class
        self.indices    = [i for i in range(self.data_source.__len__()) if self.targets[i] in self.exposed_class]

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        if self.distributed:
            self.num_samples = int(len(self.indices) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices))

    def __iter__(self):
        if self.distributed:
            # subsample
            indices = self.indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices)

    def __len__(self):
        return self.num_selected_samples

class CIRSampler(Sampler):
    """Class-Incremental with Repetition (Hemati et al., CoLLAs 2023, G_samp-style) stream for the FlyGCL trainer.

    Occurrence matrix O (C x E): class c first appears in experience e_c ~ Uniform{0 .. E-3} ('uniform') or, with 'front',
    half of the classes at e=0 and the rest uniform; afterwards it re-appears in each later experience with probability p_rep
    (at least one repetition is forced when p_rep > 0 so every class can be measured after a gap). Each class's training
    samples are shuffled and split equally over its occurrences: single pass, stream length = dataset size.
    Interface used by the trainer: set_task(t) (in place), __iter__, __len__, disjoint_classes (unused, kept for compatibility)."""

    def __init__(self, data_source, num_experiences: int, p_rep: float, rnd_seed: int, first: str = "uniform", **unused):
        self.data_source = data_source; self.E = int(num_experiences); self.p_rep = float(p_rep)
        g = torch.Generator().manual_seed(int(rnd_seed))
        targets = torch.as_tensor(data_source.targets); self.classes = list(range(int(targets.max()) + 1)); C = len(self.classes)
        first_e = torch.randint(0, max(self.E - 2, 1), (C,), generator=g)
        if first == "front":
            front = torch.randperm(C, generator=g)[: C // 2]; first_e[front] = 0
        O = torch.zeros(C, self.E, dtype=torch.bool)
        for c in range(C):
            O[c, first_e[c]] = True
            later = torch.arange(first_e[c] + 1, self.E)
            if len(later) > 0 and self.p_rep > 0:
                rep = torch.rand(len(later), generator=g) < self.p_rep
                if not rep.any():
                    rep[torch.randint(0, len(later), (1,), generator=g)] = True   # force >= 1 repetition
                O[c, later[rep]] = True
        self.occurrence = O
        self.indices = [[] for _ in range(self.E)]
        for c in range(C):
            idx = torch.nonzero(targets == c).flatten(); idx = idx[torch.randperm(len(idx), generator=g)]
            occ = torch.nonzero(O[c]).flatten().tolist()
            for k, chunk in enumerate(torch.tensor_split(idx, len(occ))):
                self.indices[occ[k]].extend(chunk.tolist())
        for e in range(self.E):
            perm = torch.randperm(len(self.indices[e]), generator=g); self.indices[e] = [self.indices[e][i] for i in perm.tolist()]
        self.disjoint_classes = [[] for _ in range(self.E)]
        self.task = 0
        for e in range(self.E):
            logger.info(f"[CIR] experience {e}: {int(O[:, e].sum())} classes, {len(self.indices[e])} samples, "
                        f"{int((O[:, e] & (O[:, :e].sum(1) > 0)).sum())} returning")

    def set_task(self, cur_iter):
        self.task = int(cur_iter)

    def __iter__(self):
        return iter(self.indices[self.task])

    def __len__(self):
        return len(self.indices[self.task])


class FewShotSampler(Sampler):
    """Base-then-few-shot task-free stream. Experience 0: a fraction `base_frac` of the classes with (1 - trickle) of their
    training samples. Experiences 1..E-1: the remaining classes split evenly, each contributing only `shots` samples (the rest
    are dropped), mixed with the reserved `trickle` fraction of base-class samples spread uniformly so the common classes never
    stop flowing (no clean boundary). Single pass. Interface: set_task(t), __iter__, __len__, disjoint_classes (classes that
    first appear in each experience)."""

    def __init__(self, data_source, num_experiences: int, shots: int, rnd_seed: int, base_frac: float = 0.5, trickle: float = 0.2, **unused):
        self.E = int(num_experiences); g = torch.Generator().manual_seed(int(rnd_seed))
        targets = torch.as_tensor(data_source.targets); C = int(targets.max()) + 1
        perm = torch.randperm(C, generator=g); n_base = int(round(base_frac * C))
        base, novel = perm[:n_base].tolist(), perm[n_base:]
        self.indices = [[] for _ in range(self.E)]
        for c in base:
            idx = torch.nonzero(targets == c).flatten(); idx = idx[torch.randperm(len(idx), generator=g)]
            k = int(round(trickle * len(idx))) if self.E > 1 else 0
            self.indices[0].extend(idx[k:].tolist())
            for e, chunk in enumerate(torch.tensor_split(idx[:k], max(self.E - 1, 1)), start=1):
                self.indices[e].extend(chunk.tolist())
        groups = torch.tensor_split(novel, self.E - 1) if self.E > 1 else []
        self.disjoint_classes = [sorted(base)] + [sorted(grp.tolist()) for grp in groups]
        for e, grp in enumerate(groups, start=1):
            for c in grp.tolist():
                idx = torch.nonzero(targets == c).flatten(); idx = idx[torch.randperm(len(idx), generator=g)][: int(shots)]
                self.indices[e].extend(idx.tolist())
        for e in range(self.E):
            perm_e = torch.randperm(len(self.indices[e]), generator=g); self.indices[e] = [self.indices[e][i] for i in perm_e.tolist()]
            logger.info(f"[FEWSHOT] experience {e}: {len(self.indices[e])} samples, {len(self.disjoint_classes[e])} new classes"
                        f"{' (' + str(int(shots)) + ' shots each)' if e > 0 else ''}")
        self.task = 0

    def set_task(self, cur_iter):
        self.task = int(cur_iter)

    def __iter__(self):
        return iter(self.indices[self.task])

    def __len__(self):
        return len(self.indices[self.task])


class RenditionSampler(Sampler):
    """ImageNet-R stream whose experiences are RENDITION blocks that recur (e.g. sketch cartoon painting sketch tattoo cartoon
    misc sketch): every class can recur, but with a domain change between experiences. A rendition appearing k times has its
    samples split into k disjoint chunks (single pass). Requires an ImageFolder-like dataset whose file names start with
    '<rendition>_' (the standard ImageNet-R naming)."""

    def __init__(self, data_source, sequence, rnd_seed: int, rend_len: int = 8, rend_p_return: float = 0.5, rend_min: int = 1000, **unused):
        base = data_source.dataset if hasattr(data_source, "dataset") else data_source
        paths = [p for p, _ in base.samples]
        import os
        rend = [os.path.basename(p).split("_")[0] for p in paths]
        g = torch.Generator().manual_seed(int(rnd_seed))
        if list(sequence) == ["random"]:
            # randomized domain sequence (seeded): domains with >= rend_min train images; each step returns to an already
            # seen domain with probability rend_p_return (uniform over seen ones, never the current), otherwise moves to an
            # unseen domain (uniform); when all are seen, always returns.
            counts = {}
            for r in rend:
                counts[r] = counts.get(r, 0) + 1
            pool = sorted(r for r, c in counts.items() if c >= rend_min)
            order = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
            seq = [order[0]]; unseen = order[1:]
            while len(seq) < int(rend_len):
                seen = [r for r in dict.fromkeys(seq) if r != seq[-1]]
                go_back = (torch.rand(1, generator=g).item() < rend_p_return and seen) or not unseen
                if go_back and seen:
                    seq.append(seen[torch.randint(0, len(seen), (1,), generator=g).item()])
                else:
                    seq.append(unseen.pop(0))
            sequence = seq
            logger.info(f"[REND] random sequence (seed {rnd_seed}, p_return {rend_p_return}): {sequence}")
        self.sequence = list(sequence); self.E = len(self.sequence)
        self.indices = [[] for _ in range(self.E)]
        for r in sorted(set(self.sequence)):
            idx = torch.tensor([i for i, x in enumerate(rend) if x == r]); idx = idx[torch.randperm(len(idx), generator=g)]
            occ = [e for e, x in enumerate(self.sequence) if x == r]
            for k, chunk in enumerate(torch.tensor_split(idx, len(occ))):
                self.indices[occ[k]] = chunk.tolist()
        self.disjoint_classes = [[] for _ in range(self.E)]; self.task = 0
        targets = torch.as_tensor(base.targets)
        for e in range(self.E):
            cls = targets[torch.tensor(self.indices[e], dtype=torch.long)].unique() if self.indices[e] else torch.tensor([])
            logger.info(f"[REND] experience {e} ({self.sequence[e]}): {len(self.indices[e])} samples, {len(cls)} classes")

    def set_task(self, cur_iter):
        self.task = int(cur_iter)

    def __iter__(self):
        return iter(self.indices[self.task])

    def __len__(self):
        return len(self.indices[self.task])
