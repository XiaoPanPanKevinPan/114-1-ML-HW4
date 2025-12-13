# ML HW4

學號姓名：112550169 潘仰祐

[English Ver.](README.md)

我使用 ResNet-152 作為訓練起點，經過 10 Epochs 後，使用生出的模型對未標籤資料進行標記，再使用 Semi-Supervised Learning 進行 10 Epochs 的訓練（Fine-Tune）。

## 環境設定
這份作業主要在 Python3.14 (w/Cuda 13.1) 的虛擬環境上測試過（GPU: Nvidia GeForce RTX 3060 Laptop）。在 Google Colab 中，使用 Python3.12 與 3.9 (w/Cuda 12.4) 的虛擬環境，也能執行。可以參考 `requirements_*.txt` 中記錄，查看虛擬環境使用的套件列表與版本。

由於我無法建立一個 Cuda 11.3 的環境（它已經是 ChatGPT 時代前的產物了），所以 **建議用 Google Colab 測試 `inference.ipynb`**。

### 必要套件
可以用這個指令安裝：
```bash
pip install torch torchvision pandas loguru tqdm matplotlib scikit-learn pillow numpy
```
*註：可能需要按照 CUDA 版本安裝對應的 torch (e.g. `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124`)。
*註2：要使用 CUDA 可能還需要安裝 Nvidia 系列 Driver。

## 檔案結構
在復現作業的時候，會使用到的入口檔案：
- `main.py`：訓練第一階段。只使用有標籤的 `data/train` 訓練。
- `semi_supervised.py`：訓練第二階段（Semi-Supervised）。混合真實資料 + `data/unlabeled` 的推測標記進行訓練。
- `inference.py`：推論腳本。用來幫 `data/test` 和 `data/unlabeled` 做標記。可以給定信心門檻，過濾掉不確定的預測。（可以使用 `--help` 查看選項）
    - `--preset 1`：在第一階段訓練後，使用 `best_model.pth` 推論 `data/test`（以防萬一不小心刪掉 `submission.csv`，或者最後一個 epoch 不是 best_model 而把它覆蓋掉）。
    - `--preset 2`：在第二階段訓練前，使用 `best_model.pth` 推論 `data/unlabeled`（用於半監督學習／微調）。這邊會用 `--threshold 0.99` 過濾掉不確定的資料。
    - `--preset 3`：在第二階段訓練後，使用 `best_model_semisupervised.pth` 推論 `data/test`（以防萬一不小心刪掉 `submission_semisupervised.csv`，或者最後一個 epoch 不是 best_model_semisupervised 而把它覆蓋掉）。

這個入口檔案用來輔助訓練：
- `threshold_finder.py`：輔助 `inference.py` 的腳本。給定信心門檻後，會回傳正確預測的比例，用以尋找一個好的信心門檻。（可以使用 `--help` 查看選項）

上述他們倚賴的小模組：
- `run_trainning.py`：通用訓練迴圈函數，功能有斷點恢復、訓練、驗證、儲存最佳模型、儲存檢查點、繪製 loss 曲線。
- `utils.py`：資料預處理、Dataset 定義與繪圖工具。
- `CNN.py`：模型定義（起點是 ResNet-152 + ImageNet 預訓練權重），以及用於操作模型的 `train`, `validate`, `test` 函數。

## 訓練與執行流程
以下是我設計的訓練流程。

### TL;DR
```bash
# 如果只想復現我的訓練結果，產生模型 + submission*.csv
python main.py                  # 1. 第一階段訓練
python inference.py --preset 2  # 3. 產生偽標籤
python semi_supervised.py       # 4. 第二階段訓練

# 如果只想確認我的模型預測，產生 submission*.csv
python inference.py --preset 1  # 5. 第一階段預測
python inference.py --preset 3  # 5. 第二階段預測
```

### 1. 訓練第一階段
前情提要：我把 `data/train` 的資料，透過 sklearn 用固定亂數種子 777 打亂，然後把它的後面 20% 用作 Validation Set。

在這個階段，我們只使用 `data/train` 的已知資料（777 打亂後的前 80%），訓練第一版。
```bash
python main.py
```
- 輸出 1：`best_model.pth`，是訓練過程中，Validation 的 Accuracy 最高的模型。
- 輸出 2：`submission.csv`，是對 `data/test` 的預測結果。

- 結果：第 10 Epoch 時產生了 best_model，它的 Validation Accuracy 是 99.45%。

### 2. 決定信心門檻（復現不需要）
為了下一步，我們要知道定義怎樣叫做「模型很有信心」，這個腳本可以用來測試上一步訓練出的 model，在只看它的信心高於門檻的預測時，在 validation set 上的正確率。預設值如下：
```bash
python threshold_finder.py
# ^^^^ 相當於 vvvv
python threshold_finder.py \
    --threshold 0.99 \
    --validation-ratio 0.1 \
    --model best_model.pth \
    --ratio-seed 999 \
    --batch-size 24
```

- 結果：我使用 `python threshold_finder.py --validation-ratio 1` 的結果是 99.8749% 的精準度，我認為足夠了，因此我選擇 0.99 作為信心門檻。

### 3. 產生偽標籤
我想善用資料集 `data/unlabeled`，因此我使用第一步訓練出的 model，配合第二步的信心門檻，對無標籤資料進行預測。

```bash
python inference.py --preset 2
# ^^^^ 相當於 vvvv
python inference.py \
    --model best_model.pth \
    --input data/unlabeled \
    --output pseudo_labels.csv \
    --threshold 0.99 \
    --from-checkpoint False
```

- 輸出：`pseudo_labels.csv`

### 4. 訓練第二階段 (Semi-Supervised)
使用 `data/train` + `data/unlabeled` 的推測標記進行訓練。後者的標記從上一步產生的 `pseudo_labels.csv` 中取得。

這一步的模型權重，基於第一階段產生的 `best_model.pth` 進行微調。
```bash
python semi_supervised.py
```
- 輸出 1：`best_model_semisupervised.pth`，是訓練過程中，Validation 的 Accuracy 最高的模型。
- 輸出 2：`submission_semisupervised.csv`，是對 `data/test` 的預測結果。

- 結果：第 10 Epoch 時產生了 best_model_semisupervised，它的 Validation Accuracy 是 99.51%。（其實沒有多很多）

### 5. 重新產生預測結果
雖然 `main.py` 與 `semi_supervised.py` 會產生 `submission*.csv`。但是如果需要，我們也是可以使用 `inference.py` 來再次產生這兩個檔案，以便上傳 Kaggle。

另外，說不定運氣不好，best_model 的效果不比訓練終點（最後一個 checkpoint）高，所以我們也可以用它來產生預測結果，再看看效果。

```bash
python inference.py --preset 1
# ^^^^ 相當於 vvvv
python inference.py \
    --model best_model.pth \
    --input data/test \
    --output submission.csv \
    --threshold 1 \
    --from-checkpoint False
```
```bash
python inference.py --preset 3
# ^^^^ 相當於 vvvv
python inference.py \
    --model best_model_semisupervised.pth \
    --input data/test \
    --output submission_semisupervised.csv \
    --threshold 1 \
    --from-checkpoint False
```
- 輸出：依照 `--output` 而定。

## 訓練設計 & Hyper Parameters
以下是各部分的設定：

### 資料預先處理
- 分類與數值對應
    - 第 0 類：real
    - 第 1 類：generated
- 資料載入與標準化
    - 使用 RGB 色彩空間數值
    - 直接縮放成 (224, 224) 大小
    - 如果要資料增強
        - 隨機水平翻轉（50% 的機率）
        - 沒了
    - 轉成 torchvision Image
    - 使用精度 torch.float32
    - 使用 ImageNet 的均值與標準差進行標準化

### 訓練第一階段
- 模型初始化：ResNet-152 + ImageNet 預訓練權重
- 資料來源：`data/train`
  - 亂數排序之固定種子：777
  - Training Set 大小：80%
  - Validation Set 大小：20%
- Batch Size：24
- Epoch：10
- Learning Rate：
    - 始於：1e-3
    - 學習率排程：Cosine Annealing
    - 終於：1e-7
- 優化器：Adam
- 損失函數：Cross Entropy

### 產生偽標籤
如前面「訓練與執行流程」的「2. 決定信心門檻（復現不需要）」所述
- 資料來源：`data/unlabeled`
- 信心門檻：0.99
- Batch Size：24
- 預測分數轉成信心（機率）：使用 Softmax

### 訓練第二階段
- 模型初始化：ResNet-152 + 第一階段 best_model 權重
- 資料來源 1：`data/train`
    - 亂數排序之固定種子：777
    - Training Set 大小：80%
    - Validation Set 大小：20%
- 資料來源 2：`data/unlabeled`
    - 只取 `pseudo_labels.csv` 有標記的部分
    - 加入 Training Set
- Batch Size：24
- Epoch：10
- Learning Rate：
    - 始於：5e-5
    - 學習率排程：Cosine Annealing
    - 終於：1e-7
- 優化器：Adam
- 損失函數：Cross Entropy