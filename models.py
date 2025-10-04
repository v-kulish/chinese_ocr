from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss helpers

def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    logits, target, ignore: (B,1,H,W). ignore==1 means 'exclude pixel from loss'.
    """
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    w = (1.0 - ignore)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss for binary maps.
    """
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1,2,3))
    denom = (p + target).sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def masked_bce_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    dice_w: float = 1.0,
) -> torch.Tensor:
    """
    BCE (masked) + Dice (unmasked). 
    """
    bce = masked_bce_with_logits(logits, target, ignore, pos_weight=pos_weight)
    dl = dice_loss_from_logits(logits * (1.0 - ignore), target * (1.0 - ignore))
    return bce + dice_w * dl

# Models

class TinyFCN(nn.Module):
    """
    Very small fully-convolutional baseline. Output is 1xHxW logits.
    """
    def __init__(self, in_ch: int = 3, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1)
        )
    def forward(self, x): return self.net(x)


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNetTiny(nn.Module):
    """
    Lightweight U-Net: (enc1→enc3) → bottleneck → (dec3→dec1). Output logits (1xHxW).
    """
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = _conv_block(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = _conv_block(base, base*2)
        self.enc3 = _conv_block(base*2, base*4)
        self.bott = _conv_block(base*4, base*8)

        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = _conv_block(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = _conv_block(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = _conv_block(base*2, base)
        self.out  = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)              # H
        e2 = self.enc2(self.pool(e1))  # H/2
        e3 = self.enc3(self.pool(e2))  # H/4
        b  = self.bott(self.pool(e3))  # H/8

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)            # logits


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("fcn", "tinyfcn", "baseline"):
        return TinyFCN(**kwargs)
    if name in ("unet", "unet_tiny", "unettiny"):
        return UNetTiny(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Try 'fcn' or 'unet'.")
