import os
import argparse
import torch
import numpy as np
from PIL import Image

from load import load_manifest, read_annotations
from preprocess import CTWMaskDataset
from models import build_model
from evaluation import sliding_window_predict 

def save_heat_overlay(rgb01, prob01, out_png, alpha=0.45):
    # red overlay on RGB
    red = np.zeros_like(rgb01); red[..., 0] = 1.0
    overlay = (1 - alpha) * rgb01 + alpha * (prob01[..., None] * red)
    Image.fromarray((overlay.clip(0,1) * 255).astype(np.uint8)).save(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/manifest.json")
    ap.add_argument("--ckpt", required=True, help="Path to saved checkpoint (e.g., outputs/best_unet.pt)")
    ap.add_argument("--split", default="test", choices=["val","test","train"])
    ap.add_argument("--full_size", type=int, default=2048)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--limit", type=int, default=8, help="Max images to dump")
    ap.add_argument("--out_dir", default="outputs/overlays")
    ap.add_argument("--thresh", type=float, default=-1.0, help="Override threshold; <0 uses ckpt best_thresh")
    ap.add_argument("--model", default="", help="Override model name if ckpt lacks it (fcn|unet)")
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    ann = read_annotations(man["train_jsonl"])
    split_dict = man["splits"]

    files = split_dict[args.split]
    os.makedirs(args.out_dir, exist_ok=True)

    # Build full-res dataset 
    base_ds = CTWMaskDataset(files, ann, man["img_dir"], size=args.full_size, augment=False)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)

    model_name = (ckpt.get("model") or args.model or "unet").lower()
    model = build_model("unet" if "unet" in model_name else "fcn").to(device)
    model.load_state_dict(ckpt["state_dict"])

    thresh = args.thresh if args.thresh >= 0 else float(ckpt.get("best_thresh", 0.5))
    print(f"Using model={model_name} | threshold={thresh:.2f}")

    # Save a few samples
    for i in range(min(args.limit, len(base_ds))):
        x, y, ign, fn = base_ds[i]               # x: 3xH xW in [0,1]
        x1 = x.unsqueeze(0).to(device)          # 1x3xH xW
        prob = sliding_window_predict(model, x1, device, patch=args.patch, stride=args.stride)[0,0].cpu().numpy()
        rgb = x.permute(1,2,0).cpu().numpy()

        stem = os.path.splitext(fn)[0]
        out_heat = os.path.join(args.out_dir, f"{stem}_heat.png")
        out_mask = os.path.join(args.out_dir, f"{stem}_mask.png")

        save_heat_overlay(rgb, prob, out_heat, alpha=0.45)

        mask = (prob >= thresh).astype(np.uint8) * 255
        Image.fromarray(mask).save(out_mask)

        print(f"[saved] {out_heat} | {out_mask}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
