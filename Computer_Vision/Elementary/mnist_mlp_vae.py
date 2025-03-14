import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gzip
import numpy as np
import os


# MNISTDataset 클래스 정의 (VAE용)
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
        return self.images[idx], self.images[idx]  # 입력과 출력이 동일함 (VAE는 재구성 목표)

# VAE 모델 정의
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()  # 출력 범위를 0~1로 제한
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# VAE 손실 함수 정의
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


# 모델 학습 함수
def train(model, train_loader, optimizer, epochs=10):
    model.train()
    device = next(model.parameters()).device  # model 로부터 device 계산
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs, mu, logvar = model(images)
            loss = vae_loss(outputs, images, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
        
def evaluate(model, test_loader):
    model.train()
    device = next(model.parameters()).device  # model 로부터 device 계산
    total_loss = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs, mu, logvar = model(images)
            loss = vae_loss(outputs, images, mu, logvar)
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
    latent_dim = 16
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습 및 평가 실행
    train(model, train_loader, optimizer, epochs=10)
    evaluate(model, test_loader)
