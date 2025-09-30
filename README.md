# AIMI

## Lab 1 : Pneumonia Classification
> **Task** : Detect Pneumonia from Chest X-Ray Images  
> **Dataset** : Chest X-Ray Images (Pneumonia)  
> **Report** : [Report](https://github.com/Ianuyu/AIMI/blob/main/Lab1/LAB1_314553020_%E8%A8%B1%E8%89%AF%E4%BA%A6.pdf)
## Abstract
本實作以 ResNet-18、ResNet-50 與 DenseNet-121（可選 ImageNet 預訓練）對胸腔 X 光進行二分類（NORMAL / PNEUMONIA）。
使用自訂 DataLoader 做等比縮放與基本強度增強，並以不平衡取樣降低類別偏差。結果顯示三模型整體表現接近，ResNet-18 在成本/效益上最佳。

## Results 
| Model        | Acc (%) | Precision | Recall | F1  |
|--------------|:-------:|:---------:|:------:|:---:|
| ResNet-18    | 91.19   | 0.88      | 0.98   | 0.93 |
| ResNet-50    | 90.06   | 0.89      | 0.95   | 0.92 |
| DenseNet-121 | 90.22   | 0.88      | 0.96   | 0.92 |

## Command
```text
# Training
python main.py --mode train --backbone resnet18 --in-ch 3 --use-imbalanced --use-class-weights \
  --epochs 50 --batch-size 64 --lr 1e-4 --weight-decay 1e-4 --dropout 0.09
# Plot
python main.py --mode plot --backbone resnet18
# Testing
python main.py --mode test --backbone resnet18 --ckpt checkpoints
```

