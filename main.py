import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger

from CNN import CNN, train, validate, test
from utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset, plot
from run_trainning import run_training

def main():
    """
    載入資料
    """
    logger.info("Start loading data")

    # 載入 data/train
    images, labels = load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)
    
    ## 將 data/train 分成 train 與 val (8 : 2)
    train_len = int(0.8 * len(images))
    train_images, val_images = images[:train_len], images[train_len:]
    train_labels, val_labels = labels[:train_len], labels[train_len:]
    
    ## 建立 dataset
    train_dataset = TrainDataset(train_images, train_labels)
    val_dataset = TrainDataset(val_images, val_labels, with_augmentation=False)
    
    # 載入 data/test
    test_images = load_test_dataset()
    test_dataset = TestDataset(test_images)
    
    """
    CNN - 訓練與驗證
    """
    logger.info("Start training CNN")
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # 優化器設定
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=1e-3)

    # lr scheduler: 1e-3 to 1e-7
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

    run_training(train_loader, val_loader, model, device, criterion, optimizer, epochs=10, save_suffix="", scheduler=scheduler)

    """
    CNN - 測試
    """
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
    test(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()
