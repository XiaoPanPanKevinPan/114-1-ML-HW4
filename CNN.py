import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        # 使用 ResNet152 搭配 ImageNet 預訓練權重
        self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

        # 將 ResNet152 的最後一層改為兩個類別
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    model.train() # 設定模型為訓練模式
    running_loss = 0.0
    correct_predictions = 0

    # 使用 tqdm 包裹 train_loader 以顯示進度條
    for X, y in tqdm(train_loader, desc="Training"):
        # 將數據移到指定的設備 (CPU 或 GPU)
        X, y = X.to(device), y.to(device)

        # 前向傳播
        pred = model(X)

        # 計算損失
        loss = criterion(pred, y)

        # 清零優化器的梯度
        optimizer.zero_grad()

        # 反向傳播
        loss.backward()

        # 更新模型權重
        optimizer.step()

        # 累加損失
        running_loss += loss.item() * X.size(0) # 乘以批次大小以獲得總損失

        # 計算準確率
        _, predicted_classes = torch.max(pred, 1) # 找到預測機率最高的類別索引
        correct_predictions += (predicted_classes == y).sum().item() # 累計正確預測的數量

    # 計算平均損失
    total_samples = len(train_loader.dataset)
    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy

def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    model.eval() # 設定模型為評估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # torch.no_grad() 確保在評估過程中不計算梯度，節省記憶體和計算
    with torch.no_grad():
        # 使用 tqdm 包裹 val_loader 以顯示進度條
        for X, y in tqdm(val_loader, desc="Validating"):
            # 將數據移到指定的設備 (CPU 或 GPU)
            X, y = X.to(device), y.to(device)

            # 前向傳播
            pred = model(X)

            # 計算損失
            loss = criterion(pred, y)

            # 累計損失
            running_loss += loss.item() * X.size(0) # 乘以批次大小以獲得總損失
            total_samples += X.size(0) # 累計總樣本數

            # 計算準確率
            _, predicted_classes = torch.max(pred, 1) # 找到預測機率最高的類別索引
            correct_predictions += (predicted_classes == y).sum().item() # 累計正確預測的數量

    # 計算平均損失和準確率
    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device, save_suffix=""):
    model.eval() # 設定模型為評估模式
    predictions = []
    filenames = []

    with torch.no_grad():
        # 使用 tqdm 包裹 test_loader 以顯示進度條
        for X, fnames in tqdm(test_loader, desc="Testing"):
            # 將數據移到指定的設備 (CPU 或 GPU)
            X = X.to(device)

            # 前向傳播
            pred = model(X)

            # 獲取預測類別 (索引)
            _, predicted_classes = torch.max(pred, 1)

            # 將預測結果和檔名添加到列表中
            predictions.extend(predicted_classes.cpu().numpy()) # 將結果移回 CPU 並轉為 NumPy 陣列
            filenames.extend(fnames)

    # 創建 pandas DataFrame
    # 根據作業要求，Header 應為: filename,label
    results_df = pd.DataFrame({'filename': filenames, 'label': predictions})

    # 存檔到 submission.csv
    results_df.to_csv("submission%s.csv" % save_suffix, index=False, header=True)
    print(f"Predictions saved to 'submission%s.csv'" % save_suffix)

    return results_df
