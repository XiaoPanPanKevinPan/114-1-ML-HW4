import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger

from CNN import CNN, test
from utils import TrainDataset, load_train_dataset, TestDataset, load_test_dataset
from run_trainning import run_training

def main():
    """
    載入資料
    """
    # 1. 檢查是否有偽標籤檔案
    if not os.path.exists('pseudo_labels.csv'):
        logger.warning("'pseudo_labels.csv' not found.")
        logger.info("Running inference.py --preset 2 to generate pseudo-labels...")
        exit_code = os.system("python3 inference.py --preset 2")
        if exit_code != 0:
            logger.error("Failed to generate pseudo-labels. Exiting.")
            return

    # 2. 讀取偽標籤
    logger.info("Loading pseudo-labels from 'pseudo_labels.csv'...")
    df = pd.read_csv('pseudo_labels.csv')
    
    # 3. 從 data/unlabeled 讀取檔案，建立 'filename' -> 'path/to/filename.ext' 的映射
    unlabeled_dir = 'data/unlabeled'
    filename_to_path = {}
    for f in os.listdir(unlabeled_dir):
        base, _ = os.path.splitext(f)
        filename_to_path[base] = os.path.join(unlabeled_dir, f)
        
    p_images = []
    p_labels = []
    
    # 4. 從 3. 的映射中，取出對應的檔案路徑
    for idx, row in df.iterrows():
        fname = str(row['filename'])
        label = int(row['label'])
        
        if fname in filename_to_path:
            p_images.append(filename_to_path[fname])
            p_labels.append(label)

    p_images, p_labels = shuffle(p_images, p_labels, random_state=777)
            
    logger.info(f"Loaded {len(p_images)} pseudo-labeled images.")

    # 5. 準備訓練資料 (Original + Pseudo)
    
    # 5.1. Original（與 main.py 一致）
    # 載入 data/train
    o_images, o_labels = load_train_dataset()
    o_images, o_labels = shuffle(o_images, o_labels, random_state=777)
    
    ## 將 data/train 分成 train 與 val (8 : 2)
    train_len = int(0.8 * len(o_images))
    o_train_images, val_images = o_images[:train_len], o_images[train_len:]
    o_train_labels, val_labels = o_labels[:train_len], o_labels[train_len:]

    # 5.2. 將偽標籤資料加入到 Training Set
    new_train_images = o_train_images + p_images
    new_train_labels = o_train_labels + p_labels
    
    logger.info(f"New Training Set (Original + Pseudo): {len(new_train_images)}")
    
    ## 建立 dataset: main 遇到 overfitting 了，所以
    ## Train 使用 Augmentation 以減緩 overfitting
    train_dataset = TrainDataset(new_train_images, new_train_labels)
    val_dataset = TrainDataset(val_images, val_labels) 
    
    # 載入 data/test
    test_images = load_test_dataset()
    test_dataset = TestDataset(test_images)
    
    """
    CNN - 訓練與驗證
    """
    logger.info("Start training CNN (semi-supervised, fine-tuning)")
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    
    # 載入 Teacher Model 的權重作為起點 (Fine-tuning)
    if os.path.exists('best_model.pth'):
        logger.info("Loading teacher model weights from 'best_model.pth' for fine-tuning...")
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    else:
        logger.warning("'best_model.pth' not found. Starting from ImageNet weights.")

    criterion = nn.CrossEntropyLoss()
 
    # 優化器設定
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(base_params, lr=5e-5) # 從 main.py 的斷點繼續（依記錄 last best 是 5th epoch => 下一個是 6, LR = 8.12e-4）
    
    # lr scheduler: 5e-5 to 1e-7
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

    run_training(train_loader, val_loader, model, device, criterion, optimizer, epochs=10, save_suffix="_semisupervised", scheduler=scheduler)

    """
    CNN - 測試
    """
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
    test(model, test_loader, criterion, device, save_suffix="_semisupervised")

if __name__ == '__main__':
    main()
