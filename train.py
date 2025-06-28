'''
CHULA: Custom Heuristic Uncertainty-guided Loss for Accurate Land Title Deed Segmentation
Author: Teerapong Panboonyuen
'''

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.data_loader import ThaiDeedDataset, get_transforms
from utils.metrics import mIoU, F1Score
from chula_loss import CHULALoss

# Import your segmentation and detection models
from models.deeplabv3_plus import DeepLabV3Plus
from models.yolov12 import YOLOv12  # your custom YOLOv12 implementation with CHULA support

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Dataloader
    train_transforms, val_transforms = get_transforms()
    train_dataset = ThaiDeedDataset(args.train_dir, transforms=train_transforms)
    val_dataset = ThaiDeedDataset(args.val_dir, transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load class frequency stats for weighting in CHULA loss
    class_freqs = train_dataset.get_class_frequencies()

    # Initialize model
    if args.model == "deeplabv3plus":
        model = DeepLabV3Plus(num_classes=args.num_classes)
    elif args.model == "yolov12":
        model = YOLOv12(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.to(device)

    # Initialize CHULA Loss
    loss_fn = CHULALoss(
        class_freqs=class_freqs,
        lambda_ce=args.lambda_ce,
        lambda_unc=args.lambda_unc,
        lambda_heu=args.lambda_heu,
        device=device
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            optimizer.zero_grad()

            if args.model == "deeplabv3plus":
                # Predict segmentation logits and uncertainty map (aux output)
                logits, uncertainty_map = model(images)

                loss = loss_fn(logits, targets, uncertainty_map)
            else:  # YOLOv12 - multi-task output
                # YOLOv12 forward returns detection outputs and segmentation logits + uncertainty (if implemented)
                detection_out, seg_logits, uncertainty_map = model(images)

                loss = loss_fn(seg_logits, targets, uncertainty_map)
                # Add detection loss here if CHULA supports detection loss integration
                detection_loss = model.compute_detection_loss(detection_out, batch["detection_targets"].to(device))
                loss += detection_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            validate(model, val_loader, loss_fn, device, args)

def validate(model, val_loader, loss_fn, device, args):
    model.eval()
    total_loss = 0.0
    metric_miou = mIoU(num_classes=args.num_classes)
    metric_f1 = F1Score(num_classes=args.num_classes)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            if args.model == "deeplabv3plus":
                logits, uncertainty_map = model(images)
                loss = loss_fn(logits, targets, uncertainty_map)
            else:
                detection_out, seg_logits, uncertainty_map = model(images)
                loss = loss_fn(seg_logits, targets, uncertainty_map)

            total_loss += loss.item()

            preds = torch.argmax(seg_logits, dim=1)
            metric_miou.update(preds.cpu(), targets.cpu())
            metric_f1.update(preds.cpu(), targets.cpu())

    avg_loss = total_loss / len(val_loader)
    miou_score = metric_miou.compute()
    f1_score = metric_f1.compute()

    print(f"Validation Loss: {avg_loss:.4f}, mIoU: {miou_score:.4f}, F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CHULA model for Thai Land Title Deed segmentation")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--model", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "yolov12"], help="Model architecture")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of segmentation classes")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--lambda_ce", type=float, default=1.0, help="Weight for cross-entropy loss")
    parser.add_argument("--lambda_unc", type=float, default=0.4, help="Weight for uncertainty loss")
    parser.add_argument("--lambda_heu", type=float, default=0.6, help="Weight for heuristic loss")

    args = parser.parse_args()

    train(args)