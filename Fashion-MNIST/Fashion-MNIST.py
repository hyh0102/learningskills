
import torch
import torchvision

from torch import nn

torchvision.disable_beta_transforms_warning()  # 0.15.1 的 v2 还是 beta，屏蔽提示

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import decode_image

import torch.nn.functional as F

import torchvision.models as models

transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype(torch.float32)])

#下载并加载 60000 / 10000 张 28×28 灰度图
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

# 通用"CSV + 图片目录"数据集模板
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# 按 64 张一批切分
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Get Device for Training 选设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# MLP：784→512→512→10（Flatten + 3 个 Linear + ReLU）
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 创建例子验证
model = NeuralNetwork().to(device)
print(model)

# 损失函数与优化器
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练 / 测试循环
# 前向→loss→backward→step→zero_grad
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()                                     # 切到训练模式（影响 Dropout/BatchNorm）
    for batch, (X, y) in enumerate(dataloader):
        # [修正] 数据要搬到和模型相同的设备，否则 cuda 下会报 device mismatch
        X, y = X.to(device), y.to(device)

        pred = model(X)                               # 前向
        loss = loss_fn(pred, y)                       # 计算损失

        loss.backward()                               # 反向：求梯度
        optimizer.step()                              # 用梯度更新参数
        optimizer.zero_grad()                         # 清空梯度，避免累加

        if batch % 100 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


# 评估准确率与平均损失
def test_loop(dataloader, model, loss_fn):
    model.eval()                                      # 切到评估模式
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():                             # 评估不需要计算图，省显存/时间
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)         # [修正] 同上
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

torch.save(model, 'model.pth')