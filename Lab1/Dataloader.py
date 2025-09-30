from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from imbalanced import ImbalancedDatasetSampler

class KeepAspectSquareResize:
    """Resize to square (img_size x img_size) while keeping aspect ratio by padding."""
    def __init__(self, img_size: int, fill: int = 0):
        self.img_size = img_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return img
        scale = self.img_size / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        pad_left = (self.img_size - new_w) // 2
        pad_top = (self.img_size - new_h) // 2
        pad_right = self.img_size - new_w - pad_left
        pad_bottom = self.img_size - new_h - pad_top
        if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        return img


class To3Channels:
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "L":
            img = img.convert("L")
        return Image.merge("RGB", (img, img, img))


class RandomGaussianNoise(torch.nn.Module):
    """Additive Gaussian noise with small std, applied with given p."""
    def __init__(self, p: float = 0.25, std: float = 0.01):
        super().__init__()
        self.p = p
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std
            return tensor + noise
        return tensor

@dataclass
class CSVSpec:
    data_root: Path          
    csv_name: str            

class ChestXRayCSVDataset(Dataset):
    def __init__(
        self,
        spec: CSVSpec,
        transform: Optional[torch.nn.Module] = None,
        equalize: bool = True,
        to_3ch: bool = True,
    ):
        self.data_root = Path(spec.data_root)
        csv_path = self.data_root / spec.csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        path_col = None
        for c in ["path", "filepath", "img"]:
            if c in df.columns:
                path_col = c
                break
        if path_col is None or "label" not in df.columns:
            raise ValueError(f"CSV must contain columns: (path|filepath|img), label — got {df.columns.tolist()}")

        self.paths = df[path_col].astype(str).str.strip().tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform
        self.equalize = equalize
        self.to_3ch = to_3ch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel = Path(self.paths[idx])
        img_path = rel if rel.is_absolute() else (self.data_root / rel)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}\n")

        img = Image.open(img_path).convert("L")

        box = img.getbbox()
        if box is not None:
            img = img.crop(box)

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.to_3ch:
            img = To3Channels()(img)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

    def get_labels(self) -> List[int]:
        return self.labels


def build_transforms(
    img_size: int = 224,
    train_aug: bool = True,
    imagenet_norm: bool = True,
    mild_rot_deg: int = 7,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    resize = KeepAspectSquareResize(img_size=img_size, fill=0)

    if imagenet_norm:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    else:
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if train_aug:
        train_tf = T.Compose([
            resize,
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.05),
            T.RandomAffine(
                degrees=mild_rot_deg, translate=(0.02, 0.02), scale=(0.98, 1.02),
                fill=0, interpolation=T.InterpolationMode.BILINEAR
            ),
            T.ColorJitter(brightness=0.04, contrast=0.04),
            T.ToTensor(),
            RandomGaussianNoise(p=0.10, std=0.005),
            normalize,
        ])
    else:
        train_tf = T.Compose([
            resize,
            T.ToTensor(),
            normalize,
        ])

    val_tf = T.Compose([
        resize,
        T.ToTensor(),
        normalize,
    ])

    return train_tf, val_tf


def get_loaders(
    data_root: str | Path = "dataset/chest_xray",
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    use_imagenet_norm: bool = True,
    equalize: bool = True,
    to_3ch: bool = True,
    train_aug: bool = True,
    sampler: Optional[str] = None,  # None | "imbalanced" | "weighted"
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_tf, eval_tf = build_transforms(
        img_size=img_size,
        train_aug=train_aug,
        imagenet_norm=use_imagenet_norm,
    )

    data_root = Path(data_root)

    train_ds = ChestXRayCSVDataset(
        CSVSpec(data_root=data_root, csv_name="train.csv"),
        transform=train_tf, equalize=equalize, to_3ch=to_3ch
    )
    val_ds = ChestXRayCSVDataset(
        CSVSpec(data_root=data_root, csv_name="val.csv"),
        transform=eval_tf, equalize=equalize, to_3ch=to_3ch
    )
    test_ds = ChestXRayCSVDataset(
        CSVSpec(data_root=data_root, csv_name="test.csv"),
        transform=eval_tf, equalize=equalize, to_3ch=to_3ch
    )

    train_sampler = None
    if sampler == "imbalanced":
        train_sampler = ImbalancedDatasetSampler(train_ds)
    elif sampler == "weighted":
        labels = torch.tensor(train_ds.get_labels())
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float().clamp_min(1)
        sample_weights = class_weights[labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
