import os
import json
import random
from typing import Dict, List, Tuple

ROOT_DEFAULT = "/srv/.../a1" #Replace with the correct path
IMG_DIR_DEFAULT = f"{ROOT_DEFAULT}/images"
INFO_JSON_DEFAULT = f"{ROOT_DEFAULT}/info.json"
TRAIN_JSONL_DEFAULT = f"{ROOT_DEFAULT}/train.jsonl"


def read_info(info_path: str) -> dict:
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_annotations(jsonl_path: str) -> Dict[str, dict]:
    """
    Returns: {file_name: full_record_per_image}
    One line per image in the JSONL.
    """
    out = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["file_name"]] = rec
    return out


def get_official_train_list(info: dict) -> List[str]:
    if "train" in info:
        return [d["file_name"] if isinstance(d, dict) else d for d in info["train"]]
    return []


def intersect_existing(official_files: List[str], ann_by_file: Dict[str, dict], img_dir: str) -> List[str]:
    return [fn for fn in official_files if fn in ann_by_file and os.path.exists(os.path.join(img_dir, fn))]


def split_80_10_10(items: List[str], seed: int = 13) -> Tuple[List[str], List[str], List[str]]:
    r = random.Random(seed)
    arr = items[:]
    r.shuffle(arr)
    n = len(arr)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = arr[:n_train]
    val = arr[n_train:n_train + n_val]
    test = arr[n_train + n_val:]
    return train, val, test


def save_manifest(path: str,
                  img_dir: str,
                  info_path: str,
                  train_jsonl: str,
                  size: int,
                  train_files: List[str],
                  val_files: List[str],
                  test_files: List[str]) -> None:
    man = {
        "img_dir": img_dir,
        "info": info_path,
        "train_jsonl": train_jsonl,
        "size": size,
        "splits": {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man, f)


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
