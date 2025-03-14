import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gzip
import numpy as np
import os

# MNISTDataset 클래스 정의 (GAN 용)
class MNISTDataset(Dataset):
    def __init__(self, images_path):
        # 이미지 데이터 로드
        with gzip.open(images_path, 'rb') as f:
            self.images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
        
        # 정규화 및 Tensor 변환
        self.images = torch.tensor(self.images, dtype=torch.float32) / 255.0 * 2 - 1  # Normalize to [-1, 1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    

# 3-layer MLP Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()  # Output normalized to [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)


# 3-layer MLP Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, x):
        return self.model(x)

    
# 모델 학습 함수
def train(generator, discriminator, train_loader, loss_fn, optimizer_G, optimizer_D, epochs=50, latent_dim=100):
    generator.train()
    discriminator.train()
    device = next(generator.parameters()).device  # model 로부터 device 계산
    
    for epoch in range(epochs):
        total_loss_G = 0
        total_loss_D = 0
        
        for real_images in train_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # 실제 데이터 레이블 (1), 가짜 데이터 레이블 (0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # 1. Discriminator 학습
            optimizer_D.zero_grad()
            
            outputs_real = discriminator(real_images)
            loss_real = loss_fn(outputs_real, real_labels)
            
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z)
            outputs_fake = discriminator(fake_images.detach())
            loss_fake = loss_fn(outputs_fake, fake_labels)
            
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()
            total_loss_D += loss_D.item()
            
            # 2. Generator 학습
            optimizer_G.zero_grad()
            outputs_fake = discriminator(fake_images)
            loss_G = loss_fn(outputs_fake, real_labels)  # Generator는 실제 데이터처럼 속이도록 학습
            loss_G.backward()
            optimizer_G.step()
            total_loss_G += loss_G.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss D: {total_loss_D / len(train_loader):.4f}, Loss G: {total_loss_G / len(train_loader):.4f}")

if __name__ == "__main__":

      data_dir = r"D:\Non_Documents\2025\datasets\mnist"
    
    # 데이터셋 로드
    train_dataset = MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 모델, 손실 함수, 옵티마이저 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    loss_fn = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 모델 학습 실행
    train(generator, discriminator, train_loader, loss_fn, optimizer_G, optimizer_D, epochs=50, latent_dim=latent_dim)
