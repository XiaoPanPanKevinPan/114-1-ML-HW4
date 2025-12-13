import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger

from CNN import CNN, train, validate, test
from utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset, plot

def run_training(train_loader, val_loader, model, device, criterion, optimizer, epochs, save_suffix="", scheduler=None):
    """
    通用訓練迴圈函數。從原本的 main() 提取出來，以供 main.py 與 semi_supervised.py 使用
    功能包含：
    - 斷點恢復
    - 訓練
    - 驗證
    - 儲存最佳模型、檢查點
    - 繪製損失曲線
    """
    train_losses = []
    val_losses = []
    max_acc = 0

    # 斷點恢復
    start_epoch = 0
    checkpoint_filename = f'checkpoint{save_suffix}.pth'
    best_model_filename = f'best_model{save_suffix}.pth'
    
    if os.path.exists(checkpoint_filename):
        try:
            logger.info(f"Found checkpoint '{checkpoint_filename}', trying to recover...")
            checkpoint = torch.load(checkpoint_filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            max_acc = checkpoint['max_acc']

            # 載入歷史 Loss 記錄，確保 plot 帶歷史記錄
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']

            # 載入學習率调度器狀態
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Successfully recovered! Continuing from Epoch {start_epoch + 1}")

        except Exception as e:
            logger.error(f"Failed to recover: {e}, starting from scratch")

    # 訓練迴圈
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 儲存最佳模型（原子存檔，以免存到一半斷電）
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), best_model_filename + '.tmp')
            os.replace(best_model_filename + '.tmp', best_model_filename)
            logger.info(f"Saved new best model with acc: {val_acc:.4f}")

        # 更新學習率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
                logger.info(f"LR Status: {optimizer.param_groups[0]['lr']}")
            else:
                scheduler.step()
                logger.info(f"LR Status: {scheduler.get_last_lr()[0]}")

        # 儲存檢查點（也是原子存檔）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'max_acc': max_acc,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, checkpoint_filename + '.tmp')
        os.replace(checkpoint_filename + '.tmp', checkpoint_filename)
        logger.info(f"Checkpoint saved to {checkpoint_filename}")

        # 訓練日誌
        logger.info(f"Epoch {epoch + 1}/{epochs} is done")
        logger.info(f"- Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        logger.info(f"- Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
    logger.info(f"Best Accuracy: {max_acc:.4f}")

    """
    CNN - 繪製訓練過程損失曲線
    """
    plot(train_losses, val_losses, filename=f'plot{save_suffix}.png')
    logger.info(f"Plot of loss saved to plot{save_suffix}.png")
