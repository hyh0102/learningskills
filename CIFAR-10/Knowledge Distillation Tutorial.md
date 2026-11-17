# 知识蒸馏教程

知识蒸馏（Knowledge Distillation）是一种把知识从大型、计算代价高昂的模型迁移到小型模型的技术，且不损失模型的性能。这样就能把模型部署到算力较弱的硬件上，使推理与评估更快、更高效。

在本教程中，我们将进行一系列实验，目标是提升一个轻量神经网络的准确率，做法是让一个更强的网络充当“教师”（teacher）。轻量网络的计算开销与前向传播速度保持不变——我们的干预只作用于它的权重，而不改变它的前向传播。这类技术常见于无人机、手机等设备。本教程不依赖任何外部库，所需的一切都在 `torch` 与 `torchvision` 中。

通过本教程，你将学会：

- 如何改造模型类，以提取隐藏表示并用于后续计算；
- 如何改造 PyTorch 的常规训练循环，在（例如）分类所用的交叉熵之外再叠加额外的损失；
- 如何借助更复杂的“教师”模型提升轻量模型的性能。

## 前置条件

- 1 块 4GB 显存的 GPU
- PyTorch 2.6 或更高版本（`torch.accelerator` 自 2.6 起才提供；原文标注的 v2.0 不足以运行下面的代码）
- CIFAR-10 数据集（脚本会自动下载，并保存到名为 /data 的目录）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 检查当前环境的加速器是否可用；若不可用则退回 CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
```

> 说明：这段代码自动检测当前运行环境中最合适的硬件（GPU / Mac MPS / CPU），并将其作为后续深度学习计算的默认设备。

**输出**

```text
Using cuda device
```

## 加载 CIFAR-10

- CIFAR-10 是一个常用的图像数据集，包含 10 个类别。我们的目标是：对每张输入图片预测它属于以下哪一类：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。

输入图片是 **RGB** 彩色图，因此有 *3 个通道*，尺寸为 32×32 像素。基本上，每张图片由 3 × 32 × 32 = 3072 个取值在 0～255 之间的数字来描述。

我们的归一化方式是：对每个通道减去均值，再除以标准差。张量 `mean=[0.485, 0.456, 0.406]` 与 `std=[0.229, 0.224, 0.225]` 是预先算好的数值，分别代表 CIFAR-10 预定训练子集中各通道的均值与标准差。请注意，我们对测试集也使用同一组数值，而不重新计算均值与标准差。原因是：网络是在“减去并除以上述数值”之后得到的特征上训练的，我们需要保持一致性。再者，现实场景中我们也无法计算测试集的均值与标准差，因为按我们的假设，那时这批数据尚不可得。

### 1. 定义数据预处理流程

```python
# 下面为 CIFAR-10 做数据预处理。这里取 128 作为批大小（任意取值）。
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**代码解读**

- `transforms.Compose([...])`：把多个图像变换操作“打包”成一条流水线。数据进来后，会按顺序依次经过其中的处理。
- `transforms.ToTensor()`：将原本的 PIL 图片（像素值 0～255，形状为“高×宽×通道”）转换成 PyTorch 张量（像素值缩放到 0.0～1.0 之间，形状变为“通道×高×宽”）。这是把图片喂给神经网络之前必须做的一步。
- `transforms.Normalize(mean=..., std=...)`：对张量做标准化处理，计算公式为 `(原值 - mean) / std`。

### 2. 加载训练集

```python
# 加载 CIFAR-10 数据集：
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
```

**代码解读**

- `datasets.CIFAR10(...)`：调用 torchvision 内置的 CIFAR-10 数据集类。
- `root='./data'`：指定数据集下载与存放的文件夹路径（当前目录下的 data 文件夹）。
- `train=True`：告诉代码现在加载的是训练集（包含 50,000 张图片）。
- `download=True`：如果 ./data 目录下没有数据，会自动从网上下载；如果已经下载好，则直接读取。
- `transform=transforms_cifar`：在读取每一张图片时，自动应用上面定义好的预处理流水线（转 Tensor 与标准化）。

### 3. 加载测试集

```python
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)
```

**代码解读**

与训练集几乎完全相同，唯一的区别是 `train=False`：加载测试集（包含 10,000 张图片），用于最终评估模型的真实准确率。同样，它在读取时也会应用相同的预处理。

### 4. 数据加载器（DataLoader）

```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
```

**代码解读**

- `batch_size=128`：每次打包 128 张图片作为一个批次送进模型。模型的前向传播、损失计算、反向传播与权重更新，都是基于这 128 张图片进行的。
- `shuffle=True`（仅训练集）：每个 epoch（一轮训练）开始时，把 50,000 张图片完全打乱顺序。打乱顺序能打破样本之间的相关性，让模型真正学到图像特征，防止过拟合。
- `shuffle=False`（仅测试集）：测试时按原顺序加载，不打乱。
- `num_workers=2`：使用 2 个子进程在后台并行加载数据。

## 定义模型类与工具函数

接下来需要**定义模型类**。这里有几个由我们自主设定的参数。我们使用两种不同的架构，并在所有实验中保持滤波器数量固定，以保证比较的公平性。两个架构都是卷积神经网络（CNN），卷积层数量不同，用作特征提取器，之后接一个 10 类的分类器。学生网络的滤波器数量与神经元数量都更少。

```python
# 更深的神经网络，用作教师模型：
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 轻量的神经网络，用作学生模型：
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

**教师模型 `DeepNN`**

这是一个参数更多、结构更深的网络。

- **卷积层（`features`）**
  - 两个连续的 `Conv2d + ReLU`（通道数 3→128→64），再接一个 `MaxPool2d` 下采样（图片尺寸 32→16）。
  - 接着又是两个连续的 `Conv2d + ReLU`（通道数 64→64→32），再接一个 `MaxPool2d` 下采样（图片尺寸 16→8）。
  - 注意：这里没有使用 BatchNorm（BN 层）；而且在进入全连接层之前，特征图尺寸已变成 8×8，通道数为 32。
- **分类头（`classifier`）**
  - 到这里特征图是 32 通道 × 8 × 8，因此展平（flatten）后的维度是 32 × 8 × 8 = 2048。
  - 经 `nn.Linear(2048, 512)` 降维，再用 `Dropout(0.1)` 防止过拟合，最后输出 `num_classes`（即 10 类）。

**学生模型 `LightNN`**

这是一个参数很少的轻量网络，目标是模仿教师的输出。

- **卷积层（`features`）**
  - 通道数大幅减少（3→16→16），只用了两层卷积和两个池化层（图片尺寸同样是 32→16→8）。
- **分类头（`classifier`）**
  - 特征图是 16 通道 × 8 × 8，展平后的维度是 16 × 8 × 8 = 1024。
  - 经 `nn.Linear(1024, 256)`、`Dropout`，最后输出 10 类。

**展平操作 `torch.flatten(x, 1)`**

在 `forward` 函数里，经过 `self.features(x)` 之后，数据是一个四维张量 `(batch_size, channels, height, width)`。全连接层只接受二维数据，所以必须展平。

`torch.flatten(x, 1)` 的含义是：保留第 0 维（也就是 `batch_size`），把后面的通道、高、宽全部拉平成一维。

- DeepNN 展平后的长度是 2048。
- LightNN 展平后的长度是 1024。

**为什么两个网络要保持一致**

两个网络使用了完全相同的激活函数和分类逻辑，这也是为后续做知识蒸馏做准备：教师教学生时，通常需要对比两者在最终输出层（logits）上的差异。

我们使用 2 个函数来产生并评估原始分类任务的结果。

其中一个函数名为 **`train`**，接受以下参数：

1. **`model`**：要通过该函数训练（更新其权重）的模型实例。
2. **`train_loader`**：前面定义的 `train_loader`，其作用是把数据送入模型。
3. **`epochs`**：在数据集上循环的轮数。
4. **`learning_rate`**：学习率决定朝收敛方向迈出的步子有多大。步子过大或过小都可能有害。
5. **`device`**：决定在哪个设备上运行负载。视可用情况，可以是 CPU 或 GPU。

```python
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: 一个批次（batch_size 张）图像的集合
            # labels: 维度为 batch_size 的向量，其中的整数表示每张图像的类别
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: 网络对该批图像的输出，维度为 batch_size × num_classes 的张量
            # labels: 图像的真实标签，维度为 batch_size 的向量
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
```

## 交叉熵方法

为保证可复现性，需要设置 torch 的随机种子。我们用不同方法训练网络，为了公平比较，用**相同的权重**初始化网络是合理的。先以交叉熵训练教师网络：

```python
torch.manual_seed(42)
nn_deep = DeepNN(num_classes=10).to(device)
train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_deep = test(nn_deep, test_loader, device)

# 实例化轻量网络：
torch.manual_seed(42)
nn_light = LightNN(num_classes=10).to(device)
```

**输出**

```text
Epoch 1/10, Loss: 1.3324644655522788
Epoch 2/10, Loss: 0.8708866857506735
Epoch 3/10, Loss: 0.6903234038816388
Epoch 4/10, Loss: 0.5379468149998609
Epoch 5/10, Loss: 0.4173577759424439
Epoch 6/10, Loss: 0.31721441809783507
Epoch 7/10, Loss: 0.2246860868828681
Epoch 8/10, Loss: 0.16713004645979618
Epoch 9/10, Loss: 0.14741274812604155
Epoch 10/10, Loss: 0.11847115276605272
Test Accuracy: 74.54%
```

### 为什么实际结果与原文不同？

在深度学习里充满了随机性（stochasticity）。即使代码完全没改，每次跑出来的准确率通常也会上下浮动 1%～3% 左右。主要原因有四个：

1. **权重初始化是随机的**：神经网络刚创建时，所有参数（权重与偏置）都是随机生成的（例如默认的正态分布）。起点不同，模型最终落入的“局部最优解”也不同。
2. **数据打乱（`shuffle=True`）**：DataLoader 每个 epoch 都会重新随机打乱 5 万张图片。模型每次看到的顺序不同，学到的特征路径也会不同。
3. **Dropout 层的随机性**：模型里有 `nn.Dropout(0.1)`。每次前向传播时，它会随机“丢弃”10% 的神经元，相当于每次都在训练一个结构略有不同的子网络。
4. **GPU 的底层计算误差**：如果用 GPU 训练，CUDA 底层的并行计算（例如矩阵乘法）在某些情况下无法保证绝对的数学确定性，可能引入极微小的浮点误差，最终被放大。

我们再实例化一个轻量网络模型来比较它们的表现。反向传播对权重初始化很敏感，因此需要确保这两个网络拥有**完全相同的初始化**。

```python
torch.manual_seed(42)
new_nn_light = LightNN(num_classes=10).to(device)
```

为确认我们创建的是第一个网络的副本，我们检查其第一层权重的范数。如果两者一致，就可以安全地认为这两个网络确实相同。

```python
# 打印初始轻量网络第一层的范数
print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
# 打印新轻量网络第一层的范数
print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())
```

**输出**

```text
Norm of 1st layer of nn_light: 2.327361822128296
Norm of 1st layer of new_nn_light: 2.327361822128296
```

**统计每个模型的参数总量**

```python
total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
print(f"DeepNN parameters: {total_params_deep}")
total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
print(f"LightNN parameters: {total_params_light}")
```

**输出**

```text
DeepNN parameters: 1,186,986
LightNN parameters: 267,738
```

**用交叉熵训练并测试轻量网络**

```python
train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_light_ce = test(nn_light, test_loader, device)
```

**输出**

```text
Epoch 1/10, Loss: 1.4684629473844757
Epoch 2/10, Loss: 1.1556517544304927
Epoch 3/10, Loss: 1.0252509245177364
Epoch 4/10, Loss: 0.9241082334457456
Epoch 5/10, Loss: 0.8479285933782378
Epoch 6/10, Loss: 0.7787943416849121
Epoch 7/10, Loss: 0.7136897055236885
Epoch 8/10, Loss: 0.6535112059024899
Epoch 9/10, Loss: 0.6033801174224795
Epoch 10/10, Loss: 0.5528530050878939
Test Accuracy: 69.96%
```

如我们所见，根据测试准确率，现在可以把用作教师的更深网络与作为学生的轻量网络做个比较。到目前为止，学生尚未接受教师的任何干预，因此这一成绩是学生自身取得的。可用下面几行代码查看目前的指标：

**指标对比（学生尚未接受教师干预）**

```python
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")
```

**输出**

```text
Teacher accuracy: 74.54%
Student accuracy: 69.96%
```

## 知识蒸馏实验

现在尝试引入教师来提升学生网络的测试准确率。知识蒸馏正是实现这一点的直接技术，其依据是：两个网络都会在各类别上输出一个概率分布，因此两者具有相同数量的输出神经元。该方法的做法是：在传统交叉熵损失之外，再加入一项基于教师网络 softmax 输出的额外损失。

其假设是：一个训练得当的教师网络，其输出激活值携带了额外信息，学生网络可以在训练期间加以利用。原始论文指出，利用软目标中较小概率之间的比值，有助于实现深度神经网络的根本目标——在数据上构造一种相似性结构，使相似的对象被映射得更接近。例如在 CIFAR-10 中，一辆卡车如果露出轮子，可能被误认为汽车或飞机，但不太可能被误认为狗。因此可以合理假设：有价值的信息不仅存在于训练得当的模型给出的最高预测中，而是存在于整个输出分布里。

然而，仅靠交叉熵并不能充分利用这些信息，因为非预测类别的激活值往往极小，回传的梯度无法有意义地改变权重，从而构造出这个理想的向量空间。

继续定义第一个引入“教师—学生”动态的辅助函数时，还需要增加几个参数：

1. **`T`**：温度，控制输出分布的平滑程度。T 越大，分布越平滑，小概率值获得的提升也越大。
2. **`soft_target_loss_weight`**：赋予即将引入的额外目标的权重。
3. **`ce_loss_weight`**：赋予交叉熵的权重。调整这两个权重，会把网络推向优化其中某一个目标。

```python
def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # 教师网络切换到评估模式
    student.train() # 学生网络切换到训练模式

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 教师网络前向传播：这里不保存梯度，因为我们不改变教师的权重
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # 学生网络前向传播
            student_logits = student(inputs)

            # 先做 softmax 再取 log，使学生 logits 变"软"
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # 计算软目标损失。按论文《Distilling the knowledge in a neural network》作者的建议乘以 T**2
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # 计算真实标签损失
            label_loss = ce_loss(student_logits, labels)

            # 两项损失的加权和
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# 以温度 T=2 运行 train_knowledge_distillation。交叉熵与蒸馏损失的权重任意设为 0.75 与 0.25。
train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

# 对比蒸馏后学生在有/无教师两种情况下的测试准确率
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
```

**输出**

```text
Epoch 1/10, Loss: 2.368525334636269
Epoch 2/10, Loss: 1.8519720012879433
Epoch 3/10, Loss: 1.6284415977995108
Epoch 4/10, Loss: 1.468683267188499
Epoch 5/10, Loss: 1.3428366185759035
Epoch 6/10, Loss: 1.2304401574537271
Epoch 7/10, Loss: 1.135715493918075
Epoch 8/10, Loss: 1.048583413938732
Epoch 9/10, Loss: 0.9742893933335228
Epoch 10/10, Loss: 0.90246815708897
Test Accuracy: 70.63%
Teacher accuracy: 74.54%
Student accuracy without teacher: 69.96%
Student accuracy with CE + KD: 70.63%
```

## 余弦损失最小化实验

欢迎自行调参：控制 softmax “软化”程度的温度参数，以及各项损失的系数，都可以随意尝试。在神经网络中，很容易在主目标之外加入额外的损失函数，以实现更好的泛化等目标。

这次我们为学生网络引入一个目标，但关注的不再是输出层，而是它们的隐藏状态。我们的目标是：通过加入一个朴素的损失函数，把信息从教师的表示传递给学生——该损失的最小化意味着，随后送入分类器的展平向量会随损失下降而变得更加相似。当然，教师不会更新自己的权重，因此最小化只取决于学生的权重。

这一方法背后的思路是：我们假设教师模型拥有更好的内部表示，而这种表示若不加外部干预，学生很难自己达到；因此我们人为地把学生推向模仿教师的内部表示。不过，这样做最终是否真的帮到学生，并不容易判断：一方面，把轻量网络推到这一状态可能是好事（前提是我们找到了能带来更高测试准确率的内部表示）；另一方面也可能有害，因为两个网络架构不同，学生的学习能力不等于教师。换言之，学生的向量与教师的向量没有任何理由逐分量地相同——学生完全可能达到一种只是教师表示“重排”（permutation）的内部表示，而效果同样好。尽管如此，我们仍可以做个快速实验，看看这个方法的影响。

我们将使用 `CosineEmbeddingLoss`，其公式如下：

$$
\text{loss}(x_1, x_2, y) =
\begin{cases}
1 - \cos(x_1, x_2), & y = 1 \\
\max\left(0,\ \cos(x_1, x_2) - \text{margin}\right), & y = -1
\end{cases}
$$

当然，有一件事需要先解决。当我们把蒸馏用于输出层时，提到过两个网络的神经元数量相同，都等于类别数。但在卷积层之后的那一层，情况并非如此：展平最后一层卷积的输出后，教师的神经元比学生多。我们的损失函数要求输入两个维度相同的向量，因此必须想办法对齐。解决办法是：在教师的卷积层之后加入一个平均池化层，把它的维度降到与学生相同。

为此，我们要修改模型类，或者新建模型类。现在 `forward` 函数不仅返回网络的 logits，还返回卷积层之后展平的隐藏表示。对改造后的教师，我们加入前面提到的池化。

```python
class ModifiedDeepNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, 2)
        return x, flattened_conv_output_after_pooling

# 类似地创建学生类，返回一个元组。展平后我们不做池化。
class ModifiedLightNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        return x, flattened_conv_output

# 当然不必从零训练改造后的深层网络，直接从已训练的实例加载权重即可
modified_nn_deep = ModifiedDeepNNCosine(num_classes=10).to(device)
modified_nn_deep.load_state_dict(nn_deep.state_dict())

# 再次确认两个网络第一层的范数相同
print("Norm of 1st layer for deep_nn:", torch.norm(nn_deep.features[0].weight).item())
print("Norm of 1st layer for modified_deep_nn:", torch.norm(modified_nn_deep.features[0].weight).item())

# 用与其他轻量实例相同的随机种子初始化改造后的轻量网络。它将从头训练，以检验余弦损失最小化的效果。
torch.manual_seed(42)
modified_nn_light = ModifiedLightNNCosine(num_classes=10).to(device)
print("Norm of 1st layer:", torch.norm(modified_nn_light.features[0].weight).item())
```

**输出**

```text
Norm of 1st layer for deep_nn: 7.493465423583984
Norm of 1st layer for modified_deep_nn: 7.493465423583984
Norm of 1st layer: 2.327361822128296
```

自然地，训练循环也需要改动，因为现在模型返回的是一个元组 `(logits, hidden_representation)`。用一个示例输入张量，可以打印它们的形状。

```python
# 创建一个示例输入张量
sample_input = torch.randn(128, 3, 32, 32).to(device) # 批大小：128，通道数：3，图像尺寸：32x32

# 让输入经过学生网络
logits, hidden_representation = modified_nn_light(sample_input)

# 打印张量的形状
print("Student logits shape:", logits.shape) # batch_size × total_classes
print("Student hidden representation shape:", hidden_representation.shape) # batch_size × hidden_representation_size

# 让输入经过教师网络
logits, hidden_representation = modified_nn_deep(sample_input)

# 打印张量的形状
print("Teacher logits shape:", logits.shape) # batch_size × total_classes
print("Teacher hidden representation shape:", hidden_representation.shape) # batch_size × hidden_representation_size
```

**输出**

```text
Student logits shape: torch.Size([128, 10])
Student hidden representation shape: torch.Size([128, 1024])
Teacher logits shape: torch.Size([128, 10])
Teacher hidden representation shape: torch.Size([128, 1024])
```

在我们的例子里，`hidden_representation_size` 是 1024。这是学生网络最后一层卷积的特征图展平后的结果，也是其分类器的输入。教师网络同样也是 1024，这是我们用 `avg_pool1d` 把它从 2048 降下来的。这里使用的损失只影响学生在损失计算之前的那些权重；换句话说，它不影响学生的分类器。改造后的训练循环如下：

在余弦损失最小化中，我们希望通过向学生网络回传梯度，最大化两组表示的余弦相似度：

```python
def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # 教师网络切换到评估模式
    student.train() # 学生网络切换到训练模式

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 教师网络前向传播，只保留隐藏表示
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(inputs)

            # 学生网络前向传播
            student_logits, student_hidden_representation = student(inputs)

            # 计算余弦损失。目标为全 1 向量；由上面的损失公式可知，此时最小化损失等价于提高余弦相似度。
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))

            # 计算真实标签损失
            label_loss = ce_loss(student_logits, labels)

            # 两项损失的加权和
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
```

出于同样的原因，测试函数也需要修改。这里我们忽略模型返回的隐藏表示。

```python
def test_multiple_outputs(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs) # 忽略元组中的第二个张量
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
```

在这种情况下，我们其实很容易把知识蒸馏与余弦损失最小化放进同一个函数。在“教师—学生”范式中，组合使用多种方法以取得更好性能是很常见的。目前，我们先跑一个简单的训练—测试流程。

```python
# 用交叉熵 + 余弦损失训练并测试轻量网络
train_cosine_loss(teacher=modified_nn_deep, student=modified_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_cosine_loss = test_multiple_outputs(modified_nn_light, test_loader, device)
```

**输出**

```text
Epoch 1/10, Loss: 1.306798359317243
Epoch 2/10, Loss: 1.0764028016868454
Epoch 3/10, Loss: 0.9782528128770306
Epoch 4/10, Loss: 0.9029897430058941
Epoch 5/10, Loss: 0.8479770197892738
Epoch 6/10, Loss: 0.8029736768254234
Epoch 7/10, Loss: 0.7589672209356751
Epoch 8/10, Loss: 0.7231402253860708
Epoch 9/10, Loss: 0.6870624677604421
Epoch 10/10, Loss: 0.6581869998856273
Test Accuracy: 71.16%
```

## 中间回归器实验

我们的朴素最小化并不能保证更好的结果，原因有几点，其中之一是向量的维度。对于高维向量，余弦相似度通常比欧氏距离效果更好，但我们处理的是每个向量含 1024 个分量的情况，要提取有意义的相似性就更难了。此外，正如我们所说，“把师生隐藏表示推向一致”在理论上也站不住脚——我们并没有充分理由去追求这两个向量的 1:1 匹配。

下面给出最后一个训练干预的例子：引入一个名为“回归器”（regressor）的额外网络。目标是先取出教师在某一卷积层之后的特征图，再取出学生在某一卷积层之后的特征图，最后尝试匹配这两个特征图。不过这一次，我们在两个网络之间引入一个回归器来辅助匹配。该回归器是可训练的，理想情况下会比我们朴素的余弦损失最小化做得更好。它的主要作用是把这些特征图的维度对齐，以便我们在教师与学生之间正确定义损失函数。定义这样的损失函数就提供了一条“教学路径”——本质上是一条回传梯度的通路，用来改变学生的权重。

聚焦于原网络中每个分类器之前那层卷积的输出，形状如下：

```python
# 只把示例输入送进卷积特征提取器
convolutional_fe_output_student = nn_light.features(sample_input)
convolutional_fe_output_teacher = nn_deep.features(sample_input)

# 打印它们的形状
print("Student's feature extractor output shape: ", convolutional_fe_output_student.shape)
print("Teacher's feature extractor output shape: ", convolutional_fe_output_teacher.shape)
```

**输出**

```text
Student's feature extractor output shape:  torch.Size([128, 16, 8, 8])
Teacher's feature extractor output shape:  torch.Size([128, 32, 8, 8])
```

教师有 32 个滤波器，学生有 16 个。我们将加入一个可训练层，把学生的特征图转换成教师的特征图形状。实践中，我们改造轻量类，使其在经由一个中间回归器（其输出尺寸与卷积特征图一致）之后返回隐藏状态；并改造教师类，使其返回最后一层卷积的输出，不做池化或展平。

这个可训练层使两个中间张量的形状匹配，从而可以正确定义均方误差（MSE）：

```python
class ModifiedDeepNNRegressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        conv_feature_map = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, conv_feature_map

class ModifiedLightNNRegressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 加入一个额外的回归器（这里用线性层实现）
        self.regressor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        regressor_output = self.regressor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, regressor_output
```

之后还需要再次更新训练循环。这次我们取出学生的回归器输出与教师的特征图，在这两个张量上计算 MSE（它们形状完全相同，因此可以正确定义），并在常规分类交叉熵损失之外，基于该损失反向传播梯度。

```python
def train_mse_loss(teacher, student, train_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # 教师网络切换到评估模式
    student.train() # 学生网络切换到训练模式

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 同样忽略教师的 logits
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)

            # 学生网络前向传播
            student_logits, regressor_feature_map = student(inputs)

            # 计算损失
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)

            # 计算真实标签损失
            label_loss = ce_loss(student_logits, labels)

            # 两项损失的加权和
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# 注意这里的测试函数与上一节完全相同：我们只关心实际输出，因为我们要衡量的是准确率。

# 初始化一个 ModifiedLightNNRegressor
torch.manual_seed(42)
modified_nn_light_reg = ModifiedLightNNRegressor(num_classes=10).to(device)

# 当然不必从零训练改造后的深层网络，直接从已训练的实例加载权重即可
modified_nn_deep_reg = ModifiedDeepNNRegressor(num_classes=10).to(device)
modified_nn_deep_reg.load_state_dict(nn_deep.state_dict())

# 再次训练并测试
train_mse_loss(teacher=modified_nn_deep_reg, student=modified_nn_light_reg, train_loader=train_loader, epochs=10, learning_rate=0.001, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_mse_loss = test_multiple_outputs(modified_nn_light_reg, test_loader, device)
```

**输出**

```text
Epoch 1/10, Loss: 1.6890651789467659
Epoch 2/10, Loss: 1.31339025741343
Epoch 3/10, Loss: 1.1697968727792316
Epoch 4/10, Loss: 1.0727734683114853
Epoch 5/10, Loss: 0.9939984606050164
Epoch 6/10, Loss: 0.9305637527609725
Epoch 7/10, Loss: 0.8756104726010882
Epoch 8/10, Loss: 0.827191528152017
Epoch 9/10, Loss: 0.7833019546840502
Epoch 10/10, Loss: 0.7460303509326847
Test Accuracy: 71.32%
```

可以预期，最后这个方法会比余弦损失表现更好，因为现在我们在教师与学生之间加入了一个可训练层，给了学生一定的调整余地，而不是硬逼着它复制教师的表示。引入这个额外网络，正是“基于提示的蒸馏”（hint-based distillation）背后的思路。

```python
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
print(f"Student accuracy with CE + CosineLoss: {test_accuracy_light_ce_and_cosine_loss:.2f}%")
print(f"Student accuracy with CE + RegressorMSE: {test_accuracy_light_ce_and_mse_loss:.2f}%")
```

**输出**

```text
Teacher accuracy: 74.54%
Student accuracy without teacher: 69.96%
Student accuracy with CE + KD: 70.63%
Student accuracy with CE + CosineLoss: 71.16%
Student accuracy with CE + RegressorMSE: 71.32%
```

## 结论

上述方法都没有增加网络的参数量或推理时间，因此性能提升只以训练期间额外计算梯度为代价。在机器学习应用中，我们主要关心推理时间，因为训练发生在模型部署之前。

- 如果我们的轻量模型对部署来说仍然过重，还可以采用其他思路，例如训练后量化（post-training quantization）。
- 额外损失可以应用到许多任务中，而不只是分类；你也可以在系数、温度、神经元数量等量上做实验。
- 欢迎调整本教程中的任何数字，但请记住：一旦改动神经元/滤波器的数量，很可能出现形状不匹配。
