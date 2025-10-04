import os
import argparse
import torch
from load import read_info, read_annotations, get_official_train_list, intersect_existing, split_80_10_10, save_manifest
from preprocess import build_dataloaders

ROOT = "/srv/.../a1" #Replace with the correct path
IMG_DIR = f"{ROOT}/images"
INFO_JSON = f"{ROOT}/info.json"
TRAIN_JSONL = f"{ROOT}/train.jsonl"

def cmd_make_splits(args):
    info = read_info(INFO_JSON)
    official = get_official_train_list(info)
    ann_by_file = read_annotations(TRAIN_JSONL)
    candidates = intersect_existing(official, ann_by_file, IMG_DIR)
    print(f"Official train listed: {len(official)} | With ann & on disk: {len(candidates)}")
    print("Example:", candidates[:3])

    train_files, val_files, test_files = split_80_10_10(candidates, seed=13)
    print(f"split → train {len(train_files)} | val {len(val_files)} | test {len(test_files)}")

    os.makedirs("outputs", exist_ok=True)
    save_manifest("outputs/manifest.json", IMG_DIR, INFO_JSON, TRAIN_JSONL, args.size,
                  train_files, val_files, test_files)
    print("Saved manifest to outputs/manifest.json")

def cmd_smoke(args):
    from load import load_manifest
    man = load_manifest(args.manifest)
    ann_by_file = read_annotations(man["train_jsonl"])
    train = man["splits"]["train"]; val = man["splits"]["val"]; test = man["splits"]["test"]
    train_dl, val_dl, test_dl, *_ = build_dataloaders(
        train, val, test, ann_by_file, man["img_dir"], size=man.get("size", args.size), batch=args.batch, workers=0
    )
    xb, yb, igb, fns = next(iter(train_dl))
    print("x:", xb.shape, xb.min().item(), xb.max().item())
    print("y:", yb.shape, yb.unique())
    print("ignore:", igb.shape, igb.unique())
    print("Smoke OK.")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("make-splits", help="Create 80/10/10 manifest from official train files present on disk")
    p1.add_argument("--size", type=int, default=512)
    p1.set_defaults(func=cmd_make_splits)

    p2 = sub.add_parser("smoke", help="Load one batch and print shapes")
    p2.add_argument("--manifest", default="outputs/manifest.json")
    p2.add_argument("--batch", type=int, default=8)
    p2.add_argument("--size",  type=int, default=512)
    p2.set_defaults(func=cmd_smoke)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
