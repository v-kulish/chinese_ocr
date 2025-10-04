import torch

@torch.no_grad()
def eval_metrics(model, loader, device, thresh: float = 0.2):
    """
    Pixel Accuracy and IoU at threshold, respecting 'ignore' mask.
    """
    model.eval()
    tot_acc, tot_iou, n = 0.0, 0.0, 0
    for x, y, ign, _ in loader:
        x, y, ign = x.to(device), y.to(device), ign.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob >= thresh).float()

        keep = (ign < 0.5).float()  # 1 where valid
        total = keep.sum().clamp_min(1.0)

        correct = ((pred == y).float() * keep).sum()
        acc = (correct / total).item()

        tp = ((pred * y) * keep).sum(dim=(1,2,3))
        fp = ((pred * (1 - y)) * keep).sum(dim=(1,2,3))
        fn = (((1 - pred) * y) * keep).sum(dim=(1,2,3))
        iou = (tp / (tp + fp + fn + 1e-6)).mean().item()

        tot_acc += acc; tot_iou += iou; n += 1

    if n == 0:
        return 0.0, 0.0
    return tot_acc / n, tot_iou / n


@torch.no_grad()
def sweep_threshold(model, loader, device, steps: int = 9):
    """
    Try thresholds in (0,1): e.g., 0.1..0.5 for steps=5. Return (best_iou, best_thresh).
    """
    best_iou, best_t = -1.0, 0.5
    for i in range(1, steps + 1):
        t = 0.1 * i
        acc, iou = eval_metrics(model, loader, device, thresh=t)
        print(f"[val] thresh {t:.2f} | Acc {acc:.4f} | IoU {iou:.4f}")
        if iou > best_iou:
            best_iou, best_t = iou, t
    return best_iou, best_t

@torch.no_grad()
def sliding_window_predict(model, x_full, device, patch=512, stride=256):
    """
    x_full: 1x3xH xW tensor (full-res image).
    Returns: 1x1xH xW probability map in [0,1] by tiling/averaging overlapping patches.
    """
    model.eval()
    _, _, H, W = x_full.shape
    out = torch.zeros(1, 1, H, W, device=device)
    acc = torch.zeros(1, 1, H, W, device=device)

    def put(tile_logits, t, l):
        prob = torch.sigmoid(tile_logits)
        out[:, :, t:t+patch, l:l+patch] += prob
        acc[:, :, t:t+patch, l:l+patch] += 1.0

    # main grid
    for top in range(0, max(1, H - patch + 1), stride):
        for left in range(0, max(1, W - patch + 1), stride):
            tile = x_full[:, :, top:top+patch, left:left+patch]
            put(model(tile.to(device)), top, left)

    # bottom strip if needed
    if (H - patch) % stride != 0:
        top = H - patch
        for left in range(0, max(1, W - patch + 1), stride):
            tile = x_full[:, :, top:top+patch, left:left+patch]
            put(model(tile.to(device)), top, left)
    # right strip if needed
    if (W - patch) % stride != 0:
        left = W - patch
        for top in range(0, max(1, H - patch + 1), stride):
            tile = x_full[:, :, top:top+patch, left:left+patch]
            put(model(tile.to(device)), top, left)
    # bottom-right corner if both remainders
    if (H - patch) % stride != 0 and (W - patch) % stride != 0:
        top = H - patch; left = W - patch
        tile = x_full[:, :, top:top+patch, left:left+patch]
        put(model(tile.to(device)), top, left)

    return (out / acc.clamp_min(1.0)).clamp(0, 1)