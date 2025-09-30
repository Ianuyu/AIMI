# AIMI

## Lab 1: Pneumonia Classification
> **Task**: Detect Pneumonia from Chest X-Ray Images  
> **Dataset**: Chest X-Ray Images (Pneumonia)
---
## Abstract
本實作以 ResNet-18、ResNet-50 與 DenseNet-121（可選 ImageNet 預訓練）對胸腔 X 光進行二分類（NORMAL / PNEUMONIA）。  
使用自訂 DataLoader 做等比縮放與基本強度增強，並以不平衡取樣降低類別偏差。結果顯示三模型整體表現接近，ResNet-18 在成本/效益上最佳（詳見下表與曲線）。
---
