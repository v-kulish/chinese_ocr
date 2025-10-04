"""
Before train run: python main.py make-splits --size 512 (if manifest.json does not exist already)
After train: python viz.py --ckpt outputs/best_unet.pt --split val --limit 8 --patch 512 --stride 256

Train either FCN or UNet on the splits from a manifest.

Examples:
  U-Net with downsample, no patches
  python train.py --manifest outputs/manifest.json --model unet --epochs 5 --batch 6 --size 512
  U-Net
  python train.py --manifest outputs/manifest.json   --model unet --use_patches 
  --full_size 2048 --patch 512   --epochs 5 --batch 6 --workers 4   --pos_weight 20 
  --dice_w 0.5 --lr 1e-4
  FCN
  python train.py --manifest outputs/manifest.json   --model fcn --use_patches 
  --full_size 2048 --patch 512   --epochs 5 --batch 6 --workers 4   --pos_weight 60 
  --dice_w 0.5 --lr 1e-4
"""
import os
import json
import argparse
import torch
from load import load_manifest, read_annotations
from preprocess import build_dataloaders, build_patch_dataloaders
from models import build_model, masked_bce_dice
from evaluation import eval_metrics, sweep_threshold

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifest.json")
    ap.add_argument("--model", default="fcn", choices=["fcn","unet"], help="Model type")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--size",   type=int, default=512)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)  
    ap.add_argument("--dice_w",  type=float, default=1.0, help="Weight of Dice term in loss")
    ap.add_argument("--pos_weight", type=float, default=20.0, help="BCE positive weight, 0 to disable")
    ap.add_argument("--steps_thresh", type=int, default=9, help="Threshold sweep steps (0.1 * number of steps)")
    ap.add_argument("--out_dir", default="outputs", help="Where to save checkpoints")
    ap.add_argument("--use_patches", action="store_true", help="Train on random crops from full-res images")
    ap.add_argument("--full_size", type=int, default=2048, help="Full image size used before patching")
    ap.add_argument("--patch", type=int, default=512, help="Patch size for PatchDataset")
    ap.add_argument("--crops_per_image", type=int, default=4, help="Crops per full image per pass")
    ap.add_argument("--pos_ratio", type=float, default=0.7, help="Target fraction of positive patches")
    ap.add_argument("--min_pos_px", type=int, default=50, help="Min positive pixels to accept a 'positive' patch")

    return ap.parse_args()

def main():
    args = parse_args()
    man = load_manifest(args.manifest)
    img_dir = man["img_dir"]
    size = args.size or man.get("size", 512)
    train_files = man["splits"]["train"]
    val_files   = man["splits"]["val"]
    test_files  = man["splits"]["test"]

    ann_by_file = read_annotations(man["train_jsonl"])
    if args.use_patches:
        train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = build_patch_dataloaders(
            train_files, val_files, test_files, ann_by_file, img_dir,
            full_size=args.full_size, patch=args.patch, batch=args.batch, workers=args.workers,
            crops_per_image=args.crops_per_image, pos_ratio=args.pos_ratio, min_pos_px=args.min_pos_px
        )
    else:
        train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = build_dataloaders(
            train_files, val_files, test_files, ann_by_file, img_dir,
            size=args.size, batch=args.batch, workers=args.workers
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_w = None
    if args.pos_weight > 0:
        pos_w = torch.tensor([args.pos_weight], device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

    best_iou, best_t = -1.0, 0.5
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_pix = 0

        for step, (x, y, ign, _) in enumerate(train_dl):
            x, y, ign = x.to(device), y.to(device), ign.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = masked_bce_dice(logits, y, ign, pos_weight=pos_w, dice_w=args.dice_w)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n_pix += x.size(0)
            if step % 50 == 0:
                print(f"[train] e{epoch} s{step} loss {loss.item():.4f}")

        train_loss = running / max(n_pix, 1)
        print(f"[train] epoch {epoch} avg loss {train_loss:.4f}")

    # Changed to val after all epochs are done
    acc, iou = eval_metrics(model, val_dl, device, thresh=best_t)
    print(f"[val]   epoch {epoch} Acc {acc:.4f} | IoU {iou:.4f} (t={best_t:.2f})")

    # Optional: iterate thresholds each epoch
    cand_iou, cand_t = sweep_threshold(model, val_dl, device, steps=args.steps_thresh)
    if cand_iou > best_iou:
        best_iou, best_t = cand_iou, cand_t
        torch.save({"state_dict": model.state_dict(), "best_thresh": best_t, "model": args.model}, ckpt_path)
        print(f"  ↳ new best IoU {best_iou:.4f} @ t={best_t:.2f} (saved {ckpt_path})")

    print(f"Best val IoU: {best_iou:.4f} @ t={best_t:.2f}")
    ### Optionally evaluate on test with the chosen threshold
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        t = float(ckpt.get("best_thresh", best_t))
        acc, iou = eval_metrics(model, test_dl, device, thresh=t)
        print(f"[test] Acc {acc:.4f} | IoU {iou:.4f} (t={t:.2f})")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
