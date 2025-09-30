import argparse, copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

from Dataloader import get_loaders
from Resnet import ResNet18, ResNet50
from imbalanced import make_class_weights_from_labels
from Densenet import DenseNet121


@torch.no_grad()
def _batch_counts(pred_labels: torch.Tensor, targets: torch.Tensor):
    p = pred_labels.int(); t = targets.int()
    tp = ((p == 1) & (t == 1)).sum().item()
    tn = ((p == 0) & (t == 0)).sum().item()
    fp = ((p == 1) & (t == 0)).sum().item()
    fn = ((p == 0) & (t == 1)).sum().item()
    return tp, tn, fp, fn

def f1_from_counts(tp, fp, fn, eps=1e-12):
    return (2 * tp) / max(2 * tp + fp + fn, eps)

def prf1_from_lists(y_true, y_pred, eps=1e-12):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = 0; tp = 0
    precision = tp / max(tp + fp, eps)
    recall    = tp / max(tp + fn, eps)
    f1        = (2 * precision * recall) / max(precision + recall, eps)
    acc       = (tp + tn) / max(tp + tn + fp + fn, eps) * 100.0
    return precision, recall, f1, acc

def save_ckpt_by_acc(model, save_dir: Path, tag: str, acc_float: float):
    fname = f"{tag}_{acc_float:.2f}%.pth"
    torch.save(model.state_dict(), save_dir / fname)
    return fname

def save_eval_artifacts(y_true, y_pred, cm_path: Path, report_path: Path, header_line=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0-NORMAL','1-PNEUMONIA'])
    disp.plot(cmap=plt.cm.Blues)
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path, bbox_inches="tight"); plt.close()

    report = classification_report(y_true, y_pred, target_names=['0 - NORMAL', '1 - PNEUMONIA'])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        if header_line:
            f.write(header_line + "\n\n")
        f.write(report + "\n")

def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, total_epochs: int):
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    TP = TN = FP = FN = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=True, ncols=125)
    for x, y in pbar:
        x, y = x.to(device), torch.as_tensor(y, device=device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            n += y.numel()
            tp, tn, fp, fn = _batch_counts(pred, y)
            TP += tp; TN += tn; FP += fp; FN += fn
            running_loss += loss.item() * y.size(0)

            acc_now = 100.0 * correct / max(n, 1)
            f1_now  = f1_from_counts(TP, FP, FN)
            loss_now = running_loss / max(n, 1)
            pbar.set_postfix_str(f"acc={acc_now:6.2f}% | f1={f1_now:7.4f} | loss={loss_now:7.4f}")

    acc = 100.0 * correct / max(n, 1)
    f1  = f1_from_counts(TP, FP, FN)
    avg_loss = running_loss / max(n, 1)
    return avg_loss, acc, f1

@torch.no_grad()
def evaluate(model, loader, criterion, device, phase: str = "eval"):
    model.eval()
    running_loss, correct, n = 0.0, 0, 0
    TP = TN = FP = FN = 0
    y_true, y_pred = [], []

    pbar = tqdm(loader, desc=f"{phase:<4}", leave=True, ncols=125)
    for x, y in pbar:
        x, y = x.to(device), torch.as_tensor(y, device=device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        n += y.numel()
        tp, tn, fp, fn = _batch_counts(pred, y)
        TP += tp; TN += tn; FP += fp; FN += fn
        running_loss += loss.item() * y.size(0)

        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

        acc_now = 100.0 * correct / max(n, 1)
        f1_now  = f1_from_counts(TP, FP, FN)
        loss_now = running_loss / max(n, 1)
        pbar.set_postfix_str(f"acc={acc_now:6.2f}% | f1={f1_now:7.4f} | loss={loss_now:7.4f}")

    acc = 100.0 * correct / max(n, 1)
    f1  = f1_from_counts(TP, FP, FN)
    avg_loss = running_loss / max(n, 1)
    return avg_loss, acc, f1, (y_true, y_pred)

def build_model(backbone: str, num_classes: int, in_ch: int, dropout: float, pretrained: bool = False):
    b = backbone.lower()
    if b == "resnet18":
        return ResNet18(num_classes=num_classes, in_ch=in_ch, dropout=dropout)
    elif b == "resnet50":
        return ResNet50(num_classes=num_classes, in_ch=in_ch, dropout=dropout)
    elif b == "densenet121":
        return DenseNet121(num_classes=num_classes, in_ch=in_ch, dropout=dropout, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

def plot_metrics(csv_path: Path, out_prefix: Path):
    df = pd.read_csv(csv_path)

    def save_plot(cols, suffix):
        ax = df[cols].plot()
        fig = ax.get_figure()
        fig.savefig(out_prefix.with_name(out_prefix.stem + f"_{suffix}.png"), bbox_inches="tight")
        plt.close(fig)

    if {"train_acc","test_acc"}.issubset(df.columns):
        save_plot(["train_acc","test_acc"], "acc")
    else:
        raise ValueError("CSV lack of train_acc or val_acc")

    if {"train_f1","test_f1"}.issubset(df.columns):
        save_plot(["train_f1","test_f1"], "f1")
    else:
        raise ValueError("CSV lack of train_f1 or val_f1")

    if {"train_loss","test_loss"}.issubset(df.columns):
        save_plot(["train_loss","test_loss"], "loss")
    else:
        raise ValueError("CSV lack of train_loss or val_loss")

def main():
    ap = argparse.ArgumentParser(description="Chest X-Ray Pneumonia — train/val/test")
    ap.add_argument("--mode", choices=["train","test","plot"], default="train")
    ap.add_argument("--data-root", type=str, default="dataset")
    ap.add_argument("--backbone", choices=["resnet18","resnet50","densenet121"], default="resnet18")
    ap.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained weights")
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.09)
    ap.add_argument("--use-imbalanced", action="store_true")
    ap.add_argument("--use-class-weights", action="store_true")
    ap.add_argument("--save-dir", type=str, default="result")
    ap.add_argument("--ckpt", type=str, default="checkpoints")
    ap.add_argument("--metrics-file", type=str, default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    tag = args.backbone

    if args.mode == "train":
        sampler_opt = "imbalanced" if args.use_imbalanced else None
        train_loader, val_loader, test_loader = get_loaders(
            data_root=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            train_aug=True,
            sampler=sampler_opt,
            to_3ch=(args.in_ch == 3),
        )

        model = build_model(args.backbone, args.num_classes, args.in_ch, args.dropout, args.pretrained).to(device)

        if args.use_class_weights:
            cw = make_class_weights_from_labels(train_loader.dataset.get_labels()).to(device)
            criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_test_acc = -1.0
        
        hist = {"train_loss":[], "train_acc":[], "train_f1":[],
                "test_loss":[],  "test_acc":[],  "test_f1":[]}

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)

            te_loss, te_acc, te_f1, (ty, tp) = evaluate(model, test_loader, criterion, device, phase="test")
            prec, rec, f1_chk, acc_chk = prf1_from_lists(ty, tp)

            print(f"[TEST] Epoch={epoch:03d} "
                  f"precision={prec:.4f} recall={rec:.4f} f1={f1_chk:.4f} acc={acc_chk:.2f}% "
                  f"(loss={te_loss:.4f})")

            hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc); hist["train_f1"].append(tr_f1)
            hist["test_loss"].append(te_loss);  hist["test_acc"].append(te_acc);  hist["test_f1"].append(te_f1)

            if te_acc > best_test_acc:
                best_test_acc = te_acc
                saved = save_ckpt_by_acc(model, save_dir, tag, te_acc)
                print(f"[CKPT] improved! saved -> {saved}")
                header = f"precision={prec:.4f}, recall={rec:.4f}, f1={f1_chk:.4f}, acc={acc_chk:.2f}% (loss={te_loss:.4f})"
                save_eval_artifacts(ty, tp,
                    save_dir / f"{tag}_confusion_test_best.png",
                    save_dir / f"{tag}_classification_report_test_best.txt",
                    header_line=header,
                )

        metrics_csv = save_dir / f"{tag}_metrics.csv"
        pd.DataFrame(hist).to_csv(metrics_csv, index=False)

    elif args.mode == "test":

        _, _, test_loader = get_loaders(
            data_root=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            train_aug=False,
            sampler=None,
            to_3ch=(args.in_ch == 3),
        )
        model = build_model(args.backbone, args.num_classes, args.in_ch, args.dropout, args.pretrained).to(device)

        cp = Path(args.ckpt)
        if cp.is_dir():
            pattern = f"{args.backbone}_*.pth"
            matches = sorted(cp.glob(pattern), key=lambda p: p.stat().st_mtime)
            ckpt_path = matches[-1]
        else:
            ckpt_path = cp

        print(f"[INFO] loading checkpoint: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

        criterion = nn.CrossEntropyLoss()
        te_loss, te_acc, te_f1, (ty, tp) = evaluate(model, test_loader, criterion, device, phase="test")
        prec, rec, f1_chk, acc_chk = prf1_from_lists(ty, tp)
        print(f"[TEST] precision= {prec:.4f}, recall= {rec:.4f}, f1= {f1_chk:.4f}, acc= {acc_chk:.2f}% (loss={te_loss:.4f})")
        header = f"precision={prec:.4f}, recall={rec:.4f}, f1={f1_chk:.4f}, acc={acc_chk:.2f}% (loss={te_loss:.4f})"
        save_eval_artifacts(ty, tp,
            Path(args.save_dir) / f"{tag}_confusion_test.png",
            Path(args.save_dir) / f"{tag}_classification_report_test.txt",
            header_line=header,
        )

    elif args.mode == "plot":
        csv = Path(args.metrics_file) if args.metrics_file else (Path(args.save_dir)/f"{tag}_metrics.csv")
        plot_metrics(csv, csv)

if __name__ == "__main__":
    main()
