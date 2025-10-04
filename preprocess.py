import os
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import math, random


# ---------- raster utils ----------

def flatten_annotations(record: dict) -> List[dict]:
    flat = []
    for group in record.get("annotations", []):
        flat.extend(group)
    return flat


def rasterize_polygons(polys: List[List[List[float]]], out_h: int, out_w: int) -> np.ndarray:
    """
    polys: list of polygons; each polygon is [[x,y],[x,y],[x,y],[x,y]]
    returns uint8 mask {0,1} of shape (H, W)
    """
    m = Image.new("L", (out_w, out_h), 0)
    d = ImageDraw.Draw(m)
    for poly in polys:
        pts = [(max(0, min(out_w - 1, p[0])), max(0, min(out_h - 1, p[1]))) for p in poly]
        d.polygon(pts, outline=1, fill=1)
    return np.array(m, dtype=np.uint8)


# ---------- dataset ----------

class CTWMaskDataset(Dataset):
    """
    Loads an RGB image and builds 1-channel masks:
      - y: positive pixels (Chinese characters) as {0,1}
      - ignore: pixels to ignore in loss/metrics as {0,1}
    """
    def __init__(self, file_list, ann_dict, img_dir, size: int = 512, augment: bool = False):
        self.files = file_list
        self.ann = ann_dict
        self.img_dir = img_dir
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        rec = self.ann[fn]
        H0, W0 = rec["height"], rec["width"]
        H = W = self.size
        sx, sy = W / W0, H / H0

        # image
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB").resize((W, H))
        flip = False
        if self.augment:
            import random
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                flip = True

        # positives
        flat = flatten_annotations(rec)
        pos_polys = []
        for a in flat:
            if a.get("is_chinese", False) and "polygon" in a:
                poly = [[v[0] * sx, v[1] * sy] for v in a["polygon"]]
                if flip:
                    poly = [[W - 1 - p[0], p[1]] for p in poly]
                pos_polys.append(poly)
        y = rasterize_polygons(pos_polys, H, W)

        # ignore
        ign_polys = []
        for ig in rec.get("ignore", []):
            if "polygon" in ig:
                poly = [[v[0] * sx, v[1] * sy] for v in ig["polygon"]]
                if flip:
                    poly = [[W - 1 - p[0], p[1]] for p in poly]
                ign_polys.append(poly)
        ignore = rasterize_polygons(ign_polys, H, W) if ign_polys else np.zeros((H, W), np.uint8)

        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)  # 3xHxW
        y = torch.from_numpy(y.astype(np.float32))[None, ...]                              # 1xHxW
        ign = torch.from_numpy(ignore.astype(np.float32))[None, ...]                       # 1xHxW
        return x, y, ign, fn


# ---------- dataloader helpers ---------- 

def build_dataloaders(train_files, val_files, test_files, ann_by_file, img_dir,
                      size: int = 512, batch: int = 8, workers: int = 0):
    """
    Returns (train_dl, val_dl, test_dl, train_ds, val_ds, test_ds)
    """
    train_ds = CTWMaskDataset(train_files, ann_by_file, img_dir, size=size, augment=True)
    val_ds   = CTWMaskDataset(val_files,   ann_by_file, img_dir, size=size, augment=False)
    test_ds  = CTWMaskDataset(test_files,  ann_by_file, img_dir, size=size, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)
    return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds

# ---------- load data in patches from full-res ----------

class PatchDataset(IterableDataset):
    def __init__(self, base_ds, patch=512, crops_per_image=4,
                 pos_ratio=0.7, min_pos_px=50, max_tries=20, shuffle=True):
        super().__init__()
        self.base = base_ds
        self.patch = patch
        self.crops_per_image = crops_per_image
        self.pos_ratio = pos_ratio
        self.min_pos_px = min_pos_px
        self.max_tries = max_tries
        self.shuffle = shuffle

    def _sample_crop(self, x, y, ign, want_positive):
        _, H, W = x.shape
        ps = self.patch
        for _ in range(self.max_tries):
            top  = random.randint(0, max(0, H-ps))
            left = random.randint(0, max(0, W-ps))
            yc = y[:, top:top+ps, left:left+ps]
            is_pos = (yc.sum().item() >= self.min_pos_px)
            if (want_positive and is_pos) or (not want_positive and not is_pos):
                break
        xc = x[:, top:top+ps, left:left+ps]
        ic = ign[:, top:top+ps, left:left+ps]
        return xc, yc, ic

    def __iter__(self):
        n = len(self.base)
        wi = get_worker_info()
        if wi is None:
            start, end = 0, n
        else:
            per = math.ceil(n / wi.num_workers)
            start = wi.id * per
            end = min(start + per, n)

        indices = list(range(start, end))
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            x, y, ign, fn = self.base[idx]   # index the base dataset
            for _ in range(self.crops_per_image):
                want_pos = (random.random() < self.pos_ratio)
                xc, yc, ic = self._sample_crop(x, y, ign, want_pos)
                yield xc, yc, ic, fn


def build_patch_dataloaders(train_files, val_files, test_files, ann_by_file, img_dir,
                            full_size=2048, patch=512, batch=8, workers=0,
                            crops_per_image=4, pos_ratio=0.7, min_pos_px=10):
    """
    Builds full-res base datasets (size=full_size) and wraps them in PatchDataset for loaders.
    Validation/test loaders also use patch sampling so the training/eval distribution matches.
    """
    # base full-res datasets
    train_base = CTWMaskDataset(train_files, ann_by_file, img_dir, size=full_size, augment=False)
    val_base   = CTWMaskDataset(val_files,   ann_by_file, img_dir, size=full_size, augment=False)
    test_base  = CTWMaskDataset(test_files,  ann_by_file, img_dir, size=full_size, augment=False)

    train_ds = PatchDataset(train_base, patch=patch, crops_per_image=crops_per_image,
                            pos_ratio=pos_ratio, min_pos_px=min_pos_px)
    val_ds   = PatchDataset(val_base,   patch=patch, crops_per_image=crops_per_image,
                            pos_ratio=pos_ratio, min_pos_px=min_pos_px)
    test_ds  = PatchDataset(test_base,  patch=patch, crops_per_image=crops_per_image,
                            pos_ratio=pos_ratio, min_pos_px=min_pos_px)

    # IterableDataset -> shuffle=False
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)
    return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds