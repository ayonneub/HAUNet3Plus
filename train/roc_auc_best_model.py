import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# project imports

from utils.dataset import ImageDataset, base_transform, train_transform
from utils.dataprocess import DatasetSplitter
from models.HAUNET_3Plus import UNet_3Plus_DeepSup_AC



# Reproducibility
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Collection helpers

@torch.no_grad()
def collect_scores_and_labels(
    model,
    loader,
    device,
    max_pixels: int | None = None,
    seed: int = 42,
):
   
    rng = np.random.default_rng(seed)
    model.eval()

    all_probs = []
    all_labels = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs  # (B,1,H,W)
        probs = torch.sigmoid(logits)

        probs_flat = probs.detach().view(-1).float().cpu().numpy()
        labels_flat = (masks > 0.5).detach().view(-1).float().cpu().numpy()

        if max_pixels is not None and probs_flat.shape[0] > max_pixels:
            idx = rng.choice(probs_flat.shape[0], size=max_pixels, replace=False)
            probs_flat = probs_flat[idx]
            labels_flat = labels_flat[idx]

        all_probs.append(probs_flat)
        all_labels.append(labels_flat)

    y_score = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_score



# Metric helpers

def f1_iou_from_scores(y_true, y_score, threshold: float, eps: float = 1e-9):
    """
    Compute F1 and IoU from flattened vectors using a probability threshold.
    y_true: {0,1} float or int
    y_score: probabilities in [0,1]
    """
    y_pred = (y_score >= threshold).astype(np.uint8)
    y_true_b = (y_true >= 0.5).astype(np.uint8)

    tp = np.sum((y_pred == 1) & (y_true_b == 1))
    fp = np.sum((y_pred == 1) & (y_true_b == 0))
    fn = np.sum((y_pred == 0) & (y_true_b == 1))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    iou = tp / (tp + fp + fn + eps)
    return precision, recall, f1, iou


def find_best_f1_threshold(y_true, y_score, n_thresholds: int = 201):
    """
    Find threshold that maximizes F1 over a uniform grid in [0,1].
    Returns (best_thr, best_precision, best_recall, best_f1, best_iou).
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best = (-1.0, 0.5, 0.0, 0.0, 0.0)  # (f1, thr, p, r, iou)

    for thr in thresholds:
        p, r, f1, iou = f1_iou_from_scores(y_true, y_score, thr)
        if f1 > best[0]:
            best = (f1, thr, p, r, iou)

    best_f1, best_thr, best_p, best_r, best_iou = best
    return best_thr, best_p, best_r, best_f1, best_iou


# Plot helpers 
def plot_roc(y_true, y_score, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return roc_auc


def plot_pr(y_true, y_score, save_path, title):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return ap


# Main

def main():
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument("--image_dir", type=str, default="./Sinkhole_dataset_SAMRefiner/images")
    parser.add_argument("--mask_dir", type=str, default="./Sinkhole_dataset_SAMRefiner/masks")

    # Best model settings
    parser.add_argument("--alpha", type=float, default=0.67)
    parser.add_argument("--ds", type=str, default="exp", choices=["linear", "exp", "uniform"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # Device & memory control
    parser.add_argument("--device", type=str, default=None, help='e.g. "cuda:1" or "cpu". If None -> auto')
    parser.add_argument("--max_pixels", type=int, default=2_000_000, help="subsample pixels per batch (memory control)")
    parser.add_argument("--thresholds", type=int, default=201, help="grid size for best-F1 threshold search")

    # Model checkpoint / outputs
    parser.add_argument("--ckpt_dir", type=str, default="./pretrained_models/alpha_fine")
    parser.add_argument("--out_dir", type=str, default="./pretrained_models/alpha_fine/curves")

    args = parser.parse_args()
    set_seeds(args.seed)

    # Device
    if args.device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

   
    # Dataset split 

    image_list = sorted(os.listdir(args.image_dir))
    mask_list = sorted(os.listdir(args.mask_dir))
    data = list(zip(image_list, mask_list))

    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=args.seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed)

    train_dataset = ImageDataset(train_data, args.image_dir, args.mask_dir, transform=train_transform, seed=args.seed)
    val_dataset = ImageDataset(val_data, args.image_dir, args.mask_dir, transform=base_transform, seed=args.seed)
    test_dataset = ImageDataset(test_data, args.image_dir, args.mask_dir, transform=base_transform, seed=args.seed)

    splitter = DatasetSplitter(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size, seed=args.seed)
    test_loader = splitter.test_loader


    # Build model + load checkpoint
  
    model_name = "HAUNet3+"
    ckpt_name = f"{model_name}_a{args.alpha}_ds{args.ds}.pth"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {ckpt_path}\n"
            f"Please confirm the filename under: {args.ckpt_dir}"
        )

    model = UNet_3Plus_DeepSup_AC(in_channels=3, n_classes=1).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()


    # Collect scores + labels
    y_true, y_score = collect_scores_and_labels(
        model,
        test_loader,
        device=device,
        max_pixels=args.max_pixels,
        seed=args.seed,
    )

   
    # Curves + AUC/AP

    tag = f"{model_name}_a{args.alpha}_ds{args.ds}"
    roc_path = os.path.join(args.out_dir, f"ROC_{tag}.png")
    pr_path = os.path.join(args.out_dir, f"PR_{tag}.png")

    roc_title = f"ROC Curve ({model_name}, alpha={args.alpha}, ds={args.ds})"
    pr_title = f"Precision–Recall Curve ({model_name}, alpha={args.alpha}, ds={args.ds})"

    roc_auc = plot_roc(y_true, y_score, roc_path, roc_title)
    ap = plot_pr(y_true, y_score, pr_path, pr_title)


    # F1/IoU reporting (best threshold + fixed 0.5)
   
    best_thr, best_p, best_r, best_f1, best_iou = find_best_f1_threshold(
        y_true, y_score, n_thresholds=args.thresholds
    )

    p05, r05, f105, iou05 = f1_iou_from_scores(y_true, y_score, threshold=0.5)

    
    # Save metrics summary
   
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, f"METRICS_{tag}.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Pixel subsample per batch (max_pixels): {args.max_pixels}\n\n")

        f.write("Curve-based metrics (pixel-wise):\n")
        f.write(f"  ROC-AUC: {roc_auc:.6f}\n")
        f.write(f"  Average Precision (AP / APS): {ap:.6f}\n\n")

        f.write("Threshold-based metrics (pixel-wise):\n")
        f.write(f"  Best-F1 threshold search grid: {args.thresholds} points in [0,1]\n")
        f.write(f"  Best threshold: {best_thr:.4f}\n")
        f.write(f"    Precision: {best_p:.6f}\n")
        f.write(f"    Recall:    {best_r:.6f}\n")
        f.write(f"    F1:        {best_f1:.6f}\n")
        f.write(f"    IoU:       {best_iou:.6f}\n\n")

        f.write("At threshold = 0.5:\n")
        f.write(f"    Precision: {p05:.6f}\n")
        f.write(f"    Recall:    {r05:.6f}\n")
        f.write(f"    F1:        {f105:.6f}\n")
        f.write(f"    IoU:       {iou05:.6f}\n")

    print(f"[OK] Loaded checkpoint: {ckpt_path}")
    print(f"[OK] Saved ROC curve:    {roc_path}")
    print(f"[OK] Saved PR curve:     {pr_path}")
    print(f"[OK] Saved metrics:      {metrics_path}")
    print(f"[RESULT] ROC-AUC = {roc_auc:.6f} | AP(APS) = {ap:.6f}")
    print(f"[RESULT] Best F1 = {best_f1:.6f} @ thr={best_thr:.4f} | IoU={best_iou:.6f}")
    print(f"[RESULT] @ thr=0.5 -> F1={f105:.6f} | IoU={iou05:.6f}")


if __name__ == "__main__":
    main()
