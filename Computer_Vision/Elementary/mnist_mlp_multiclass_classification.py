import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gzip
import numpy as np
import os

class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        super().__init__()

        # 이미지 데이터 로드
        with gzip.open(images_path, 'rb') as f:
            self.images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

        # 라벨 데이터 로드
        with gzip.open(labels_path, 'rb') as f:
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8)
            
        # 정규화 및 Tensor 변환
        self.images = torch.tensor(self.images, dtype=torch.float32) / 255.0  # Normalize
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# 2-layer MLP 모델 정의 (Base)
class BaseMLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 개선된 2-layer MLP 모델 정의 (BatchNorm + Dropout 추가)
class ImprovedMLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Accuracy 계산 함수
def accuracy(y_pred, y_true):
    _, preds = torch.max(y_pred, dim=1)
    return torch.sum(preds == y_true).item() / y_true.size(0)

# 모델 학습 함수
def train(model, train_loader, loss_fn, optimizer, accuracy, epochs=5):
    model.train()
    device = next(model.parameters()).device
    for epoch in range(1, epochs + 1):
        total_loss, total_acc = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

        print(f"Epoch[{epoch:2d}/{epochs}] "
              f"loss: {total_loss / len(train_loader):.3f}, "
              f"acc: {total_acc / len(train_loader):.3f}")
        
# 모델 평가 함수
def evaluate(model, test_loader, loss_fn, accuracy):
    model.eval()
    device = next(model.parameters()).device

    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

    print(f"Evaluation loss: {total_loss / len(test_loader):.3f}, "
          f"acc: {total_acc / len(test_loader):.3f}")


if __main__ == "__main__":

  data_dir = r"D:\Non_Documents\2025\datasets\mnist"

  # 데이터셋 로드
  train_dataset = MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                               os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))

  test_dataset = MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                              os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

  # DataLoader 정의
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

  # 모델, 손실 함수, 옵티마이저 정의
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = BaseMLPModel().to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  # 모델 학습 및 평가 실행
  train(model, train_loader, loss_fn, optimizer, accuracy, epochs=10)
  evaluate(model, test_loader, loss_fn, accuracy)
