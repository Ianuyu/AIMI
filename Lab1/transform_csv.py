from pathlib import Path
import pandas as pd

DATA_ROOT = Path("dataset")
DS_SUBDIR  = "chest_xray"   
SPLITS = ["train", "val", "test"]
CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}
EXTS = (".jpeg", ".jpg", ".png")

def generate_csv(split: str):
    img_root = DATA_ROOT / DS_SUBDIR / split  # dataset/chest_xray/train
    rows = []
    for cls_name, label in CLASS_TO_LABEL.items():
        cls_dir = img_root / cls_name
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in EXTS:
                rows.append((str(p.relative_to(DATA_ROOT)), label))

    if not rows:
        print(f"[warn] no images found in {img_root}")
        return

    df = pd.DataFrame(rows, columns=["path", "label"])
    out_csv = DATA_ROOT / f"{split}.csv"   
    df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    for s in SPLITS:
        generate_csv(s)
