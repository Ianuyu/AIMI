import os, cv2, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

CLASSES = ["normal", "bacteria", "virus", "COVID-19"]

def build_tfms(train=True, img_size=256, center_crop_ratio=0.95): 
    crop_sz = int(img_size * center_crop_ratio)

    base_preproc = [
        A.CLAHE(clip_limit=1.5, tile_grid_size=(8,8), p=1.0),
        A.SmallestMaxSize(max_size=max(img_size, crop_sz)),
        A.CenterCrop(height=crop_sz, width=crop_sz),
        A.Resize(img_size, img_size),
    ]

    if train:
        aug = [
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.08, rotate_limit=5,
                               border_mode=cv2.BORDER_REFLECT, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.12, p=0.6),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.15),
            A.HorizontalFlip(p=0.5),
        ]
        tfms = A.Compose(base_preproc + aug + [
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        tfms = A.Compose(base_preproc + [
            A.Normalize(),
            ToTensorV2(),
        ])
    return tfms

# Dataset
class CXRCSV(Dataset):
    def __init__(self, csv_path, img_dir, train=True, img_size=256):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.train = train
        self.tfms = build_tfms(train, img_size)

        if train:
            self.targets = self.df[CLASSES].values.argmax(1)
        self.files = self.df["new_filename"].tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        img_path = os.path.join(self.img_dir, fn)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfms(image=img)["image"]
        if self.train:
            y = int(self.targets[i])
            return img, y
        else:
            return img, fn


# Sampler
def make_sampler(targets):
    cnt = Counter(targets)
    n = len(targets)
    weights = [n / cnt[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)

def get_loaders(csv_train, dir_train, csv_val, dir_val, bs=32, img_size=256):
    ds_tr = CXRCSV(csv_train, dir_train, train=True, img_size=img_size)
    ds_va = CXRCSV(csv_val,   dir_val,   train=True, img_size=img_size) 
    ds_va.tfms = build_tfms(train=False, img_size=img_size)

    sampler = make_sampler(ds_tr.targets)
    num_w = max(1, os.cpu_count() // 2)

    tr = DataLoader(ds_tr, batch_size=bs, sampler=sampler,
                    num_workers=num_w, pin_memory=True, persistent_workers=(num_w > 0))
    va = DataLoader(ds_va, batch_size=bs, shuffle=False,
                    num_workers=num_w, pin_memory=True, persistent_workers=(num_w > 0))
    return tr, va, ds_tr.targets

