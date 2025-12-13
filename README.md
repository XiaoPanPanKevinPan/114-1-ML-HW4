# ML HW4

Student: 112550169 潘仰祐

[中文版本](README_zh.md)

I used ResNet-152 as the starting point for training. After 10 epochs (Stage 1), I used the trained model to label the unlabeled data, and then performed another 10 epochs of Semi-Supervised Learning (Fine-Tune).

## Environment Setup
This assignment was mainly developed on Python 3.14 (w/Cuda 13.1) using an Nvidia GeForce RTX 3060 Laptop GPU. It can also run on Google Colab with Python 3.12 and 3.9 (w/Cuda 12.4).
Please refer to `requirements_*.txt` for the list of packages and versions used in the virtual environments.

Since I couldn't make it to build up a development environment with Cuda 11.3 (which is already a pre-ChatGPT era product), so it is suggested to **test my inference.ipynb** on Google Colab instead.

### Required Packages
You can install the necessary packages using the following command:
```bash
pip install torch torchvision pandas loguru tqdm matplotlib scikit-learn pillow numpy
```
*Note 1: You may need to install the specific version of PyTorch compatible with your CUDA version (e.g., `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124`).*
*Note 2: Using CUDA may require installing the appropriate Nvidia drivers.*

## File Structure
Entry points used for reproducing the assignment:
- `main.py`: Stage 1 Training. Trains using only the labeled `data/train`.
- `semi_supervised.py`: Stage 2 Training (Semi-Supervised). Trains using a mix of real data and pseudo-labeled `data/unlabeled`.
- `inference.py`: Inference script. Labels `data/test` and `data/unlabeled`. Capable of filtering uncertain predictions using a confidence threshold. (Use `--help` to see options)
    - `--preset 1`: Infers `data/test` using `best_model.pth` after Stage 1 (in case `submission.csv` is lost, or is replaced when the last epoch is not the best_model).
    - `--preset 2`: Infers `data/unlabeled` using `best_model.pth` before Stage 2 (for generating pseudo-labels). Filters uncertain data with `--threshold 0.99`.
    - `--preset 3`: Infers `data/test` using `best_model_semisupervised.pth` after Stage 2 (in case `submission_semisupervised.csv` is lost, or is replaced when the last epoch is not the best_model_semisupervised).

Helper script:
- `threshold_finder.py`: Helper for `inference.py`. Give it a confidence threshold, and it reports the precision on the validation set, to help find an optimal threshold. (Use `--help` to see options)

Dependencies and modules:
- `run_trainning.py`: Generic training loop function, handling checkpoint resumption, training, validation, model saving, logging, and loss plotting.
- `utils.py`: Data preprocessing, Dataset definitions, and plotting tools.
- `CNN.py`: Model definition (ResNet-152 + ImageNet pretrained weights), and `train`, `validate`, `test` functions.

## Training & Execution Workflow
Below is the designed training workflow.

### TL;DR
```bash
# To reproduce training results (Model + submission*.csv)
python main.py                  # 1. Stage 1 Training
python inference.py --preset 2  # 3. Generate Pseudo-labels
python semi_supervised.py       # 4. Stage 2 Training

# To reproduce model predictions only (submission*.csv)
python inference.py --preset 1  # 5. Stage 1 Prediction
python inference.py --preset 3  # 5. Stage 2 Prediction
```

### 1. Stage 1 Training
Context: I shuffled the `data/train` using `sklearn` with a fixed random seed (777) and used the last 20% as the Validation Set.

In this stage, we only use the known data from `data/train` (the first 80% after shuffling) to train the first version.
```bash
python main.py
```
- Output 1: `best_model.pth`, the model with the highest Validation Accuracy during training.
- Output 2: `submission.csv`, the prediction results for `data/test`.
- Result: `best_model` was generated at Epoch 10 with a Validation Accuracy of 99.45%.

### 2. Determine Confidence Threshold (Not needed for reproduction)
To define "high confidence", this script tests the model trained in the previous step. It checks the accuracy on the validation set when considering only predictions with confidence above a certain threshold. Default values:
```bash
python threshold_finder.py
# ^^^^ Equivalent to vvvv
python threshold_finder.py \
    --threshold 0.99 \
    --validation-ratio 0.1 \
    --model best_model.pth \
    --ratio-seed 999 \
    --batch-size 24
```
- Result: using `python threshold_finder.py --validation-ratio 1` yielded a precision of 99.8749%. I deemed this sufficient and chose 0.99 as the threshold.

### 3. Generate Pseudo-labels
To utilize the `data/unlabeled` dataset, I used the model trained in step 1 and the threshold determined in step 2 to predict labels for the unlabeled data.
```bash
python inference.py --preset 2
# ^^^^ Equivalent to vvvv
python inference.py \
    --model best_model.pth \
    --input data/unlabeled \
    --output pseudo_labels.csv \
    --threshold 0.99 \
    --from-checkpoint False
```
- Output: `pseudo_labels.csv`

### 4. Stage 2 Training (Semi-Supervised)
Trains using `data/train` + pseudo-labeled `data/unlabeled`. The pseudo-labels come from `pseudo_labels.csv` generated in the previous step.
The model weights are initialized (fine-tuned) from `best_model.pth` generated in Stage 1.
```bash
python semi_supervised.py
```
- Output 1: `best_model_semisupervised.pth`, the model with the highest Validation Accuracy during training.
- Output 2: `submission_semisupervised.csv`, the prediction results for `data/test`.
- Result: `best_model_semisupervised` was generated at Epoch 10 with a Validation Accuracy of 99.51%.

### 5. Regenerate Predictions
Although `main.py` and `semi_supervised.py` generate `submission*.csv` automatically, we can use `inference.py` to regenerate them if needed.
Additionally, if the `best_model` performance is unexpectedly lower than the final checkpoint, we can use `inference.py` to generate predictions from checkpoints to compare.

```bash
python inference.py --preset 1
# ^^^^ Equivalent to vvvv
python inference.py \
    --model best_model.pth \
    --input data/test \
    --output submission.csv \
    --threshold 1 \
    --from-checkpoint False
```
```bash
python inference.py --preset 3
# ^^^^ Equivalent to vvvv
python inference.py \
    --model best_model_semisupervised.pth \
    --input data/test \
    --output submission_semisupervised.csv \
    --threshold 1 \
    --from-checkpoint False
```
- Output: Depends on `--output`.

## Training Design & Hyperparameters
Configuration details:

### Data Preprocessing
- Mapping:
    - Class 0: real
    - Class 1: generated
- Loading & Normalization:
    - Reads in RGB color space.
    - Resizes directly to (224, 224).
    - Augmentation: Random Horizontal Flip (p=0.5). No other augmentations.
    - Converts to torchvision Image.
    - Uses `torch.float32` precision.
    - Normalizes using standard ImageNet mean and std.

### Stage 1 Training
- Initialization: ResNet-152 + ImageNet Pretrained Weights
- Data Source: `data/train`
    - Shuffle Seed: 777
    - Train/Val Split: 80% / 20%
- Batch Size: 24
- Epochs: 10
- Learning Rate:
    - Start: 1e-3
    - Scheduler: Cosine Annealing
    - End: 1e-7
- Optimizer: Adam
- Loss Function: Cross Entropy

### Pseudo-label Generation
As described in "2. Determine Confidence Threshold":
- Source: `data/unlabeled`
- Threshold: 0.99
- Batch Size: 24
- Score to Probability: Softmax

### Stage 2 Training
- Initialization: ResNet-152 + Stage 1 `best_model` weights (Fine-tuning)
- Data Source 1: `data/train`
    - Shuffle Seed: 777
    - Train/Val Split: 80% / 20%
- Data Source 2: `data/unlabeled`
    - Includes Only pseudo-labels that passed the confidence threshold.
    - Added to the Training Set.
- Batch Size: 24
- Epochs: 10
- Learning Rate:
    - Start: 5e-5
    - Scheduler: Cosine Annealing
    - End: 1e-7
- Optimizer: Adam
- Loss Function: Cross Entropy
