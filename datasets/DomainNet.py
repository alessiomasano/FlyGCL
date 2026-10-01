"""DomainNet (Peng et al., 2019) for recurring-domain streams: 6 domains (clipart, infograph, painting, quickdraw, real,
sketch), 345 classes. Uses the official {domain}_train.txt / {domain}_test.txt lists; to keep the stream affordable each
domain is subsampled (seeded) to `per_domain_train` / `per_domain_test` images. File names start with '<domain>_' so the
RenditionSampler can build domain-block streams from them. ImageFolder-like attributes: samples, targets, classes."""
import os
import random
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


class DomainNet(Dataset):
    per_domain_train = 4000
    per_domain_test = 1000
    subsample_seed = 0

    def __init__(self, root: str, train: bool, transform: Optional[Callable] = None, target_transform=None, download: bool = False):
        self.root = root; self.transform = transform if transform is not None else transforms.ToTensor(); self.target_transform = target_transform
        self.samples = []
        rng = random.Random(self.subsample_seed)
        cap = self.per_domain_train if train else self.per_domain_test
        for d in DOMAINS:
            lst = os.path.join(root, f"{d}_{'train' if train else 'test'}.txt")
            if not os.path.exists(lst):
                raise RuntimeError(f"DomainNet list not found: {lst}")
            rows = [ln.split() for ln in open(lst) if ln.strip()]
            rng.shuffle(rows)
            for path, y in rows[:cap]:
                self.samples.append((os.path.join(root, path), int(y)))
        self.targets = [y for _, y in self.samples]
        self.classes = list(range(345))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return img, y
