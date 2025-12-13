#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CNN import CNN
from utils import TestDataset, load_test_dataset
from loguru import logger
import os
from tqdm import tqdm

import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for HW4")
    parser.add_argument('--preset', type=int, choices=[1, 2, 3], help="Start from a preset configuration: [with best_model.pth for data/test, with best_model.pth for data/unlabeled, with best_model_semisupervised.pth for data/test]")
    parser.add_argument('--model', type=str, help="Path to model weights")
    parser.add_argument('--input', '-i', type=str, help="Input directory path")
    parser.add_argument('--output', '-o', type=str, help="Output CSV filename")
    parser.add_argument('--threshold', type=float, default=None, help="Confidence threshold for filtering (e.g. 0.99). If set, only predictions with > threshold confidence are saved.")
    parser.add_argument('--from-checkpoint', type=bool, default=False, help="Is '--model' a checkpoint file?")
    return parser.parse_args()

def inference(model_path, input_dir, output_file, threshold=None, from_checkpoint=False):
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 檢查模型 model_path 是否存在
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' not found!")
        return

    # 載入資料
    logger.info(f"Loading data from {input_dir}...")
    if not os.path.exists(input_dir): # 若 input_dir 不存在，提醒使用者
        logger.error(f"Input directory '{input_dir}' not found!")
        return

    # 使用 utils.load_test_dataset
    image_paths = load_test_dataset(input_dir)
    if not image_paths:
        logger.warning(f"No images found in {input_dir}")
        return

    test_dataset = TestDataset(image_paths)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
    
    # 初始化模型
    logger.info("Initializing model...")
    model = CNN().to(device)
    
    logger.info(f"Loading weights from {model_path} {"" if not from_checkpoint else "(checkpoint)"}")

    if from_checkpoint:
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 執行推論
    model.eval()
    logger.info("Starting inference...")
    
    filenames = []
    predictions = []
    
    with torch.no_grad():
        for X, _ in tqdm(test_loader, desc="Testing"):
            X = X.to(device)
            outputs = model(X)
            
            # 使用 Softmax 取得機率
            probs = torch.softmax(outputs, dim=1)
            max_probs, class_opt = torch.max(probs, 1)
            
            # 轉換為 list
            batch_probs = max_probs.tolist()
            batch_preds = class_opt.tolist()
            
            # 暫存
            predictions.extend(zip(batch_preds, batch_probs))
    
    # 處理檔名
    all_filenames = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    
    # 根據 Threshold 過濾
    final_filenames = []
    final_labels = []
    
    skipped_count = 0
    
    for i, (pred_label, pred_prob) in enumerate(predictions):
        if threshold is not None and pred_prob < threshold:
            skipped_count += 1
            continue
        
        final_filenames.append(all_filenames[i])
        final_labels.append(pred_label)
        
    if threshold is not None:
        logger.info(f"Threshold {threshold}: Kept {len(final_labels)}/{len(all_filenames)} images. (Skipped {skipped_count})")
    
    # 創建 DataFrame
    import pandas as pd
    
    results_df = pd.DataFrame({'filename': final_filenames, 'label': final_labels})
    results_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to '{output_file}'")

if __name__ == '__main__':
    args = get_args()
    
    # 預設設定
    model_path = 'best_model.pth'
    input_dir = 'data/test/'
    output_file = 'submission.csv'
    threshold = args.threshold
    from_checkpoint = args.from_checkpoint
    
    # preset 設定集
    if args.preset == 1:
        # 第一階段訓練後，推論測試集
        pass 
    elif args.preset == 2:
        # 第二階段訓練前，推論未標記集（用於半監督學習／微調）
        input_dir = 'data/unlabeled/'
        output_file = 'pseudo_labels.csv'
        if threshold is None:
            threshold = 0.99 # Default threshold for preset 2
    elif args.preset == 3:
        # 第二階段訓練後，推論測試集
        model_path = 'best_model_semisupervised.pth'
        output_file = 'submission_semisupervised.csv'
        
    # args 可以覆蓋設定集
    if args.model: model_path = args.model
    if args.input: input_dir = args.input
    if args.output: output_file = args.output
    
    logger.info(f"Inference Config: Model={model_path}, Input={input_dir}, Output={output_file}, Threshold={threshold}")
    
    inference(model_path, input_dir, output_file, threshold, from_checkpoint)
