import os, pandas as pd, numpy as np, torch, torch.nn as nn
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataloader import CXRCSV, CLASSES, get_loaders
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
torch.backends.cudnn.benchmark = True

def make_scheduler(optimizer, args):
    if args.sched == "plateau":
        mode = "max" if args.monitor == "val_f1" else "min"
        return ReduceLROnPlateau(
            optimizer, mode=mode,
            factor=args.plateau_factor, patience=args.plateau_patience,
            min_lr=args.min_lr
        )
    else:  # cosine
        return CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    
def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def make_model(num_classes=4, dropout=0.1, backbone="resnet18"):
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m

    if backbone == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m

    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        return m
    
    if backbone == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_f = m.classifier[2].in_features  
        m.classifier = nn.Sequential(
            m.classifier[0],  
            m.classifier[1],   
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        return m

    raise ValueError("unknown backbone")

@torch.no_grad()
def eval(model, loader, device, criterion=None, use_amp=True):
    model.eval()
    tot_loss, n_batches = 0.0, 0
    y_true, y_pred, all_probs = [], [], []
    autocast = torch.amp.autocast if hasattr(torch, "amp") else torch.cuda.amp.autocast

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with autocast('cuda', enabled=(use_amp and torch.cuda.is_available())):
            logits = model(x)
            probs = logits.softmax(1)
            if criterion is not None:
                loss = criterion(logits, y)
                tot_loss += float(loss.item())
                n_batches += 1
        all_probs.append(probs.cpu().numpy())
        y_true.append(y.cpu().numpy())
        y_pred.append(probs.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    avg_loss = (tot_loss / n_batches) if n_batches > 0 else None
    return avg_loss, y_true, y_pred

def save_confusion_matrix(figpath, y_true, y_pred, labels):
    title = os.path.splitext(os.path.basename(figpath))[0]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close(fig)

def train_one(csv_train, dir_train, csv_val, dir_val, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    tr, va, targets = get_loaders(csv_train, dir_train, csv_val, dir_val, bs=args.bs, img_size=args.img_size)

    model = make_model(num_classes=len(CLASSES), dropout=args.dropout, backbone=args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(opt, args)         

    # class weights
    cls_cnt = np.bincount(targets, minlength=len(CLASSES))
    w = (cls_cnt.sum() / (cls_cnt + 1e-6))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device)) # change

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())  

    best_f1, best_path = -1, f"best_{getattr(args, 'backbone', 'model')}.pt"
    train_losses, val_losses, train_f1s, val_f1s = [], [], [], []
    train_accs, val_accs = [], []
    patience_left = args.es_patience

    autocast = torch.amp.autocast

    for ep in range(1, args.epochs + 1):
        # ===== Train =====
        model.train()
        running_loss = 0.0
        progress = tqdm(tr, desc=f"Epoch {ep}/{args.epochs}", ncols=200, leave=True)

        for x, y in progress:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(x)
                loss = crit(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(tr)

        # --- Evaluate train & val ---
        _ , y_tr, yhat_tr = eval(model, tr, device, crit)
        va_loss, y_va, yhat_va = eval(model, va, device, crit)

        train_f1 = f1_score(y_tr, yhat_tr, average="macro")
        val_f1 = f1_score(y_va, yhat_va, average="macro")
        train_acc = accuracy_score(y_tr, yhat_tr)
        val_acc = accuracy_score(y_va, yhat_va)

        train_losses.append(avg_train_loss)
        val_losses.append(va_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        tqdm.write(
            f"[Epoch {ep}/{args.epochs}] "
            f"train_loss={avg_train_loss:.4f}, val_loss={va_loss:.4f}, "
            f"train_F1={train_f1:.4f}, val_F1={val_f1:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        # --- LR Scheduler step ---
        prev_lr = get_lr(opt)
        if args.sched == "plateau":
            metric = val_f1 if args.monitor == "val_f1" else va_loss
            scheduler.step(metric)
        else:
            scheduler.step()
        curr_lr = get_lr(opt)
        if curr_lr < prev_lr:
            tqdm.write(f"[LR] reduced: {prev_lr:.2e} -> {curr_lr:.2e}")
        else:
            tqdm.write(f"[LR] current: {curr_lr:.2e}")

        # --- Save best & Early Stopping ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_left = args.es_patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {ep} (best val_F1={best_f1:.4f})")
                break

    print(f"Best val macro-F1: {best_f1:.4f}")

    best_state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)

    _, y_tr, yhat_tr = eval(model, tr, device, criterion=None)
    _, y_va, yhat_va = eval(model, va, device, criterion=None)

    rep_tr = classification_report(y_tr, yhat_tr, target_names=CLASSES, digits=4)
    rep_va = classification_report(y_va, yhat_va, target_names=CLASSES, digits=4)

    save_confusion_matrix("Confusionmatrix_train.png", y_tr, yhat_tr, CLASSES)
    save_confusion_matrix("Confusionmatrix_val.png",   y_va, yhat_va, CLASSES)
    print("Saved confusion matrices → cm_train.png, cm_val.png")

    report_path = f"report_{args.backbone}.txt" if hasattr(args, "backbone") else "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Classification Report (Train) ===\n")
        f.write(rep_tr + "\n\n")
        f.write("=== Classification Report (Val) ===\n")
        f.write(rep_va + "\n")
    print("Saved reports.")

    # --- Plot Loss Curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Saved loss curve → loss_curve.png")

    # --- Plot F1 Curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_f1s, label="Train F1", marker="o")
    plt.plot(val_f1s, label="Val F1", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1-score")
    plt.title("F1-score Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("f1_curve.png", dpi=150)
    print("Saved F1-score curve → f1_curve.png")

    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(train_accs, label="Train Acc", marker="o")
    plt.plot(val_accs,   label="Val Acc",   marker="o")
    plt.xlabel("Epoch") 
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc_curve.png", dpi=150)
    print("Saved Accuracy curve → acc_curve.png")

    return best_path

@torch.no_grad()
def test(ckpt_path, csv_test, dir_test, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(num_classes=len(CLASSES), dropout=args.dropout, backbone=args.backbone).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True) 
    model.load_state_dict(state)
    model.eval()

    ds_te = CXRCSV(csv_test, dir_test, train=False, img_size=args.img_size)
    te = DataLoader(ds_te, batch_size=64, shuffle=False, pin_memory=True)

    rows = []
    for x, fns in tqdm(te, desc="Saving submission", ncols=150):
        x = x.to(device)
        probs = model(x).softmax(1).cpu().numpy()   # (B,4)
        pred_idx = probs.argmax(1)                  # (B,)
        one_hot = np.eye(len(CLASSES), dtype=int)[pred_idx]  # (B,4) 
        for fn, oh in zip(fns, one_hot):
            rows.append([fn] + list(oh))     

    out_csv = getattr(args, "out_csv", "submit.csv")
    pd.DataFrame(rows, columns=["new_filename"] + CLASSES).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} (one-hot submission)")
 

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # hyperparams
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--es-patience", type=int, default=8, help="early stopping patience based on val_F1")
    # dataset paths
    ap.add_argument("--train-csv", type=str, default=r"D:\master\114-1\AIMI\Lab3\train_data.csv")
    ap.add_argument("--train-dir", type=str, default=r"D:\master\114-1\AIMI\Lab3\train_images")
    ap.add_argument("--val-csv",   type=str, default=r"D:\master\114-1\AIMI\Lab3\val_data.csv")
    ap.add_argument("--val-dir",   type=str, default=r"D:\master\114-1\AIMI\Lab3\val_images")
    ap.add_argument("--test-dir",  type=str, default=r"D:\master\114-1\AIMI\Lab3\test_images")
    ap.add_argument("--test-csv",  type=str, default="test_data.csv")
    ap.add_argument("--out-csv",   type=str, default="Submission.csv")
    ap.add_argument("--dropout", type=float, default=0.05)

    # Scheduler
    ap.add_argument("--sched", type=str, default="plateau", choices=["plateau", "cosine"])
    ap.add_argument("--monitor", type=str, default="val_f1", choices=["val_f1", "val_loss"])
    ap.add_argument("--plateau-factor", type=float, default=0.1)
    ap.add_argument("--plateau-patience", type=int, default=5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","densenet121","efficientnet_b0","convnext_tiny"])

    args = ap.parse_args()

    # Step 1: Train model and save best weights
    ckpt = train_one(args.train_csv, args.train_dir, args.val_csv, args.val_dir, args)

    # Step 2: Create test csv if missing
    if not os.path.exists(args.test_csv):
        test_files = sorted(os.listdir(args.test_dir))
        pd.DataFrame({"new_filename": test_files}).to_csv(args.test_csv, index=False)

    # Step 3: Evaluate best model on test set & save submission
    test(ckpt, args.test_csv, args.test_dir, args)


