import torch
from torchvision.transforms import v2 as transforms_v2
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    transform: transforms_v2.Compose
    images: List[str]
    labels: List[int]
    def __init__(self, images, labels, with_augmentation=True):
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RGB(),
                transforms_v2.Resize((224, 224)),
            ] + (
                [] if not with_augmentation else 
                [ transforms_v2.RandomHorizontalFlip(p=0.5) ]
            ) + [
                transforms_v2.ToImage(), 
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # ref: suggested in https://docs.pytorch.org/vision/0.8/models.html
            ]
        )
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms_v2.Compose([
            transforms_v2.RGB(),
            transforms_v2.Resize((224, 224)),
            transforms_v2.ToImage(), 
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name
    
def load_train_dataset(path: str='data/train/')->Tuple[List, List]:
    # 從給定路徑載入訓練資料集，回傳影像路徑與標籤
    images = []
    labels = []

    # 資料夾名稱就是 label。轉成數字表示
    label_map = { "real": 0, "generated": 1}

    for label in sorted(os.listdir(path)):
        fullPathForLabel = os.path.join(path, label)
        if not os.path.isdir(fullPathForLabel):
            continue
        
        if label not in label_map:
            continue
            
        current_label_id = label_map[label]
        
        for imgPath in sorted(os.listdir(fullPathForLabel)):
            fullPathForImg = os.path.join(fullPathForLabel, imgPath)
            if os.path.isdir(fullPathForImg):
                continue
            images.append(fullPathForImg)
            labels.append(current_label_id)

    return images, labels

def load_test_dataset(path: str='data/test/')->List:
    # 從給定路徑載入測試資料集，回傳影像路徑
    images = []
    for imgPath in sorted(os.listdir(path)):
        fullPathForImg = os.path.join(path, imgPath)
        if os.path.isdir(fullPathForImg):
            continue
        images.append(fullPathForImg)
    return images

def plot(train_losses: List, val_losses: List, filename: str = 'loss.png'):
    # (TODO) 繪製 CNN 的訓練損失和驗證損失，並將圖片儲存為 'loss.png'
    # x軸: 'Epoch', y軸: 'Loss'

    epochs = range(1, len(train_losses) + 1) # epoch 從 1 開始

    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    plt.close()