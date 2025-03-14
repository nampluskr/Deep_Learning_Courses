import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gzip
import numpy as np
import os


# MNISTDataset 클래스 정의 (Autoencoder용)
class MNISTDataset(Dataset):
    def __init__(self, images_path):
        # 이미지 데이터 로드
        with gzip.open(images_path, 'rb') as f:
            self.images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

        # 정규화 및 Tensor 변환
        self.images = torch.tensor(self.images, dtype=torch.float32) / 255.0  # Normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.images[idx]  # Autoencoder는 입력과 출력이 동일함


# 3-layer MLP Autoencoder 모델 정의
class Autoencoder(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()  # 출력 범위를 0~1로 제한
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 3-layer MLP Autoencoder (BatchNorm + Dropout 추가)
class OptimizedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 28*28),
            nn.Sigmoid()  # 출력 범위를 0~1로 제한
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 모델 학습 함수
def train(model, train_loader, loss_fn, optimizer, epochs=10):
    model.train()
    device = next(model.parameters()).device  # model 로부터 device 계산
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch:2d}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 모델 평가 함수
def evaluate(model, test_loader, loss_fn):
    model.eval()
    device = next(model.parameters()).device  # model 로부터 device 계산
    total_loss = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, images)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")


if __name__ == "__main__":
    data_dir = r"D:\Non_Documents\2025\datasets\mnist"

    # 데이터셋 로드
    train_dataset = MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    test_dataset = MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))

    # DataLoader 정의
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss()  # Autoencoder는 MSE 손실 사용
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습 및 평가 실행
    train(model, train_loader, loss_fn, optimizer, epochs=10)
    evaluate(model, test_loader, loss_fn)
