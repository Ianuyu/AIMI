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

| Model               |  Accuracy | Precision |   Recall  | F1-score (Macro) |
| :------------------ | :-------: | :-------: | :-------: | :--------------: |
| **ResNet-18**       |   0.818   |   0.827   |   0.871   |       0.846      |
| **DenseNet-121**    |   0.838   |   0.839   |   0.833   |       0.836      |
| **EfficientNet-B0** |   0.771   |   0.773   |   0.843   |       0.798      |
| **ConvNeXt-Tiny**   | **0.839** | **0.837** | **0.841** |     **0.837**    |

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
## Lab 3 : Multi-class Classification
> **Task** : classify X-ray images into four categories:  **Normal**, **Bacteria**, **Virus**, and **COVID-19**.  
> **Dataset** : Multi-class Classification with CXR dataset  
> **Report** : [Report]()
## Abstract
本實作以 ResNet-18、DenseNet-121、EfficientNet-B0 與 ConvNeXt-Tiny 四種深度卷積神經網路（均可選用 ImageNet 預訓練權重）進行 胸腔 X 光影像四分類任務（Normal / Bacteria / Virus / COVID-19）。
資料前處理包含等比例縮放至 256×256、隨機水平翻轉、亮度/對比度調整與輕度旋轉增強，並以 類別加權損失（Weighted Cross Entropy） 與 不平衡取樣（Imbalanced Sampling） 緩解資料分佈不均問題。
實驗結果顯示各模型整體表現穩定，其中 ConvNeXt-Tiny 具最佳宏平均 F1-score（約 0.84），能有效捕捉高階語意特徵；ResNet-18 則在訓練效率與準確率間取得良好平衡，為具代表性的基準模型。

## Results 
| Model               |  Accuracy | Precision |   Recall  | F1-score (Macro) |
| ------------------ | :-------: | :-------: | :-------: | :--------------: |
| **ResNet-18**       |   0.818   |   0.827   |   0.871   |       0.846      |
| **DenseNet-121**    |   0.838   |   0.839   |   0.833   |       0.836      |
| **EfficientNet-B0** |   0.771   |   0.773   |   0.843   |       0.798      |
| **ConvNeXt-Tiny**   | **0.839** | **0.837** | **0.841** |     **0.837**    |

## Command
```text
# Training and Testing
resnet18
python main.py --backbon resnet18 --epochs 50 --es-patience 10 --lr 5e-5 --bs 16 --img-size 256 --sched plateau --monitor val_f1
--plateau-factor 0.5 --plateau-patience 3 --min-lr 1e-6 --weight-decay 2e-4 --dropout 0.10
densenet121
python main.py --backbone densenet121 --epochs 50 --es-patience 10 --lr 5e-5 --bs 16 --img-size 256 --sched plateau --monitor val_f1
--plateau-factor 0.5 --plateau-patience 3 --min-lr 1e-6 --weight-decay 2e-4 --dropout 0.10
EfficientNet-B0
python main.py --backbone efficientnet_b0 --epochs 50 --es-patience 10 --lr 5e-5 --bs 16 --img-size 256 --sched plateau --monitor val_f1
--plateau-factor 0.5 --plateau-patience 3 --min-lr 1e-6 --weight-decay 2e-4 --dropout 0.10
ConvNeXt-Tiny
python main.py --backbone convnext_tiny --epochs 50 --es-patience 10 --lr 5e-5 --bs 16 --img-size 256 --sched plateau --monitor val_f1
--plateau-factor 0.5 --plateau-patience 3 --min-lr 1e-6 --weight-decay 2e-4 --dropout 0.10
```
