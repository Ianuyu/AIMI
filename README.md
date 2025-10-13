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
## Lab 2 : EEGNet & DeepConvNet for BCI Classification
> **Task** ：Classify EEG signals for motor imagery tasks  
> **Dataset** ：BCI Competition Dataset (EEG signals)  
> **Report** ：[Report](https://github.com/Ianuyu/AIMI/blob/main/Lab2/LAB2_314553020_%E8%A8%B1%E8%89%AF%E4%BA%A6.pdf)    
## Abstract
本實驗利用 **PyTorch** 實作了 **EEGNet** 與 **DeepConvNet** 兩種模型，用於對 **BCI Competition** 腦波資料集進行分類任務。
同時，本專案比較了不同的 **啟發函數（ELU、ReLU、LeakyReLU）** 對模型效能的影響，並針對 **ELU 的 α（alpha）參數** 進行實驗分析。
此外，訓練過程中結合了 **Warmup** 與 **Cosine Annealing**，以提升模型的收斂穩定性與最終準確率。

## Results 
| Model           | Activation | Test Acc (%) | Train Acc (%) |
| --------------- | :--------: | :----------: | :-----------: |
| **EEGNet**      |  LeakyReLU |   **87.78**  |     99.17     |
| EEGNet          |    ReLU    |     87.22    |     98.80     |
| EEGNet          |     ELU    |     85.19    |     94.17     |
| **DeepConvNet** |  LeakyReLU |   **85.09**  |     97.22     |
| DeepConvNet     |    ReLU    |     84.17    |     97.04     |
| DeepConvNet     |     ELU    |     81.94    |     99.17     |

## Command
```text
# Training and Testing
python main.py 
```
