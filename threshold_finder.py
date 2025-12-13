#!/usr/bin/env python3
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2
from sklearn.utils import shuffle
from PIL import Image
from loguru import logger
import numpy as np

from CNN import CNN
from utils import load_train_dataset, TrainDataset

def get_args():
    parser = argparse.ArgumentParser(description="Find optimal threshold using Validation Set")
    parser.add_argument("--threshold", type=float, default=0.99, help="Confidence threshold")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Ratio of validation set to use (0.0-1.0)")
    parser.add_argument('--model', type=str, default='best_model.pth', help="Path to model weights")
    parser.add_argument("--ratio-seed", type=int, default=999, help="Seed for ratio sampling")
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()

def main(args):
    # 1. 載入 data/train 與分割 Validation Set
    # （與 main.py 一致）
    logger.info("Start loading data")
    images, labels = load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)

    ## 將 data/train 分成 train 與 val (8 : 2)
    train_len = int(0.8 * len(images))
    val_images = images[train_len:]
    val_labels = labels[train_len:]
    
    total_val_len = len(val_images)
    logger.info(f"Total Validation Set size: {total_val_len}")
    
    # 2. 為求速度，隨機抽樣 (如果 ratio < 1.0)
    if args.validation_ratio < 1.0:
        sample_size = int(total_val_len * args.validation_ratio)
        # 再次 shuffle 確保隨機性，但用固定 seed 方便重現
        val_images, val_labels = shuffle(val_images, val_labels, random_state=args.ratio_seed) 
        val_images = val_images[:sample_size]
        val_labels = val_labels[:sample_size]
        logger.info(f"Sampled {sample_size} images for analysis (Ratio: {args.validation_ratio})")
    
    ## 建立 dataset
    val_dataset = TrainDataset(val_images, val_labels, with_augmentation=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 3. 載入模型
    # 初始化模型
    logger.info("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    if os.path.exists(args.model):
        logger.info(f"Loading model weights from '{args.model}'...")
        model.load_state_dict(torch.load(args.model, map_location=device))
    else:
        logger.error(f"Model '{args.model}' not found! Please train the model first.")
        return

    # 4. 執行推論與分析
    model.eval()
    total_eval_samples = 0
    kept_samples = 0
    correct_kept_samples = 0
    
    logger.info(f"Running inference with threshold {args.threshold}...")
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            # Forward: 使用 Softmax 取得機率
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            # Threshold 過濾
            mask = max_probs >= args.threshold
            
            total_eval_samples += X.size(0)
            batch_kept_count = mask.sum().item()
            kept_samples += batch_kept_count
            
            # 對通過門檻的樣本，統計正確率
            if batch_kept_count > 0:
                kept_preds = preds[mask]
                kept_labels = y[mask]
                correct_count = (kept_preds == kept_labels).sum().item()
                correct_kept_samples += correct_count
    
    # 5. 輸出報告
    if total_eval_samples == 0:
        logger.warning("No samples evaluated.")
        return

    kept_rate = (kept_samples / total_eval_samples) * 100
    precision = (correct_kept_samples / kept_samples) * 100 if kept_samples > 0 else 0.0
    
    logger.success(f" Threshold Analysis Report")
    print("\n" + "=" * 40)
    print(f" Target Model     : {args.model}")
    print(f" Confidence Thresh: {args.threshold}")
    print(f" Sample Ratio     : {args.validation_ratio} ({total_eval_samples} images)")
    print("-" * 40)
    print(f" [Retention] Kept : {kept_samples}/{total_eval_samples} ({kept_rate:.2f}%)")
    print(f" [Precision] Acc  : {precision:.4f}% (Correct: {correct_kept_samples})")
    print("=" * 40 + "\n")
    
    # 簡單分析
    if precision > 99.5:
        logger.success(f"Excellent! Threshold {args.threshold} provides high purity.")
    elif precision > 99.0:
        logger.info(f"Good. Threshold {args.threshold} is acceptable but could be cleaner.")
    else:
        logger.warning(f"Risk High! Threshold {args.threshold} has too many errors ({100-precision:.2f}% error rate).")

if __name__ == "__main__":
    args = get_args()
    main(args)
