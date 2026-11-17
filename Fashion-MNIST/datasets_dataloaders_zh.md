# 数据集与数据加载器

**Datasets & DataLoaders** · PyTorch 官方教程 · 中文精读

> 原文：<https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html> ｜ 原文更新：2026-05-07 ｜ 中文整理：2026-11-17

原文逐段翻译，每一节后面跟一段「解读」：讲清这段代码在做什么、为什么这样设计、哪里容易踩坑。

---

## 00 这套流程，一句话讲清

> **把「一堆图片 + 一张标签表」，变成模型能一口一口吃下去的「数据便当」。**

先记住一个画面：**你在食堂打饭**。

- 后厨的**冰箱**里堆着食材，旁边贴着一张**清单**——这就是 `Dataset`。
- 你喊一声“来一份”，配菜师傅从冰箱里拿出**一份**食材，顺手洗好切好——这就是 `__getitem__` 加上 `transform`。
- **打饭窗口**决定这一队按什么顺序排、一次端几盘——这就是 `DataLoader`。
- 每盘里整整齐齐码着 64 份，端到窗口，**厨师**（模型）接过去开炒。

| 食堂里的角色 | PyTorch 里的名字 | 干的活 |
| --- | --- | --- |
| 冰箱 + 清单 | `Dataset` | 知道第 *i* 份在哪，能取出 `(图, 标签)` |
| 洗菜切菜 | `transform` | 统一尺寸、把 0–255 变成 0–1 |
| 排队顺序 | `Sampler` | 这一轮按什么顺序取（`shuffle=True` 就是随机） |
| 打饭窗口 | `DataLoader` | 按批取、可多窗口并行 |
| 摆盘 | `collate_fn` | 把 64 个样本码成两个大张量 |
| 厨师 | 模型 | 收到 `[64, 1, 28, 28]` 开炒 |

### 一张图走完全程

```text
磁盘：img_dir/xxx.jpg  +  labels.csv
   │
   │ ① __init__          只记“去哪找、有哪些”，不搬食材
   ▼
Dataset ──② __len__       一共 60000 份
   │
   │ ③ __getitem__(i)     读一张图 → 解码 → transform → 返回 (图, 标签)
   ▼
DataLoader ──④ 切批 64 个 / 洗牌 / 多进程 / 拼成张量
   │
   │ ⑤ for X, y in loader:
   ▼
X: [64, 1, 28, 28]        y: [64]
   │
   ▼
⑥ 喂给模型
```

记住节奏：**① 只做一次；② 只报个数；③ 每取一个样本就调用一次；④ 每轮重新洗牌；⑤⑥ 才是训练循环。**

### 六步，每步只做一件事

1. **备料**：图片放一个文件夹，标签写一张 CSV（`文件名, 类别`）。
2. **建仓**：写 `Dataset` 的三个函数——`__init__` 记路径，`__len__` 报数量，`__getitem__(i)` 交出一份。
3. **加工**：在 `__getitem__` 里挂 `transform`，把图变成统一规格的张量；标签用 `target_transform`。
4. **装盘**：`DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)`——一次一盘、每轮重排、四个窗口并行。
5. **取餐**：`for X, y in loader:`，每圈拿到 64 个样本。
6. **开炒**：`pred = model(X)`，后面就是 `loss`、`backward`、`step`。

### 为什么要这么“绕”

- **不撑爆内存**：60000 张图不会一次性读进内存，用一张读一张（惰性）。
- **不饿着 GPU**：多进程提前把下一批备好，GPU 不用停下来等磁盘。
- **不写死**：换数据集只换 `Dataset`，训练循环一行不用改。

一句话：**`Dataset` 管“一份长什么样”，`DataLoader` 管“一盘怎么来”。**

---

## 目录

- 00 这套流程，一句话讲清
- 01 写在前面：Dataset 与 DataLoader
- 02 加载一个数据集
- 03 遍历与可视化数据集
- 04 为你的文件创建自定义数据集（`__init__` / `__len__` / `__getitem__`）
- 05 用 DataLoader 为训练准备数据
- 06 遍历 DataLoader
- 07 延伸阅读
- 08 附：速查表与常见陷阱

---

## 01 写在前面：Dataset 与 DataLoader

*Why two primitives?*

处理数据样本的代码很容易变得杂乱、难以维护。理想情况下，我们希望把**数据集代码**与**模型训练代码**解耦，以获得更好的可读性和模块化。

PyTorch 提供了两个数据原语：`torch.utils.data.DataLoader` 与 `torch.utils.data.Dataset`，它们让你既能使用预加载好的数据集，也能使用自己的数据。`Dataset` 存储样本及其对应的标签，`DataLoader` 则在 `Dataset` 之上包装出一个可迭代对象，从而轻松访问样本。

PyTorch 的领域库提供了许多预加载数据集（例如 FashionMNIST），它们继承 `torch.utils.data.Dataset`，并针对特定数据实现了相应函数。你可以用它们来快速做原型设计和基准测试。可以在这里找到它们：[图像数据集](https://pytorch.org/vision/stable/datasets.html)、[文本数据集](https://pytorch.org/text/stable/datasets.html) 和 [音频数据集](https://pytorch.org/audio/stable/datasets.html)。

### 💡 解读 01 · 两个原语，各管一段

**Dataset 回答“数据在哪、第 i 个是什么”，DataLoader 回答“怎么成批、按什么顺序、用几个进程取”。**把这两件事拆开，是 PyTorch 数据管道设计的核心：训练循环只依赖 `DataLoader` 的迭代协议（`for X, y in loader`），于是换数据集不需要改动模型代码，换模型也不需要改动数据处理代码。

#### 一条完整的数据流水线

- **Dataset**：按索引取单个样本，返回 `(特征, 标签)`，并在这里挂上 transform。
- **Sampler**：决定这一轮按什么顺序取索引 —— `shuffle=True` 就是随机顺序。
- **DataLoader**：把 Sampler 给出的索引切成 batch，交给若干 worker 并行取数据，再用 `collate_fn` 把样本列表拼成 batch 张量。

一个容易记住的类比：`Dataset` 像一个列表（能 `len()`、能 `[i]`），`DataLoader` 像“for 循环 + 打包机”。此后 `Dataset` 只是 `DataLoader` 的内部零件，你的训练代码基本只跟 `DataLoader` 打交道。

顺带区分两个同样常被混淆的概念：`map-style` 数据集（实现 `__len__` 与 `__getitem__`，可随机索引，本文这类）与 `IterableDataset`（实现 `__iter__`，只能顺序读取，适合日志流、超大数据、无法随机访问的场景）。

## 02 加载一个数据集

*Loading a Dataset*

下面是从 TorchVision 加载 Fashion-MNIST 数据集的示例。Fashion-MNIST 是一个由 Zalando 的商品图片构成的数据集，包含 60,000 个训练样本和 10,000 个测试样本。每个样本由一张 28×28 的灰度图像，以及一个来自 10 个类别之一的标签组成。

我们用以下参数加载 FashionMNIST 数据集：`root` 是训练/测试数据存储的路径，`train` 指定加载训练集还是测试集，`download=True` 表示如果 `root` 下没有数据就从互联网下载，`transform` 与 `target_transform` 分别指定特征和标签的变换。

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
```

首次运行时，你会看到如下下载进度（来自 `tqdm`）：

```text
 13%|█▎        | 3.34M/26.4M [00:01<00:04, 5.56MB/s]
 55%|█████▍    | 14.5M/26.4M [00:01<00:00, 19.9MB/s]
100%|██████████| 26.4M/26.4M [00:02<00:00, 12.3MB/s]
...（测试集与元数据文件，略）
```

*原文完整输出里还包含 4.42M 的测试集与若干小文件（约 29.5k、5.15k）的下载进度，这里为可读性做了省略。*

### 💡 解读 02 · 四行参数背后的机制

#### 1）transform 是一条流水线，顺序不能反

`v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])` 做了两件事：

- `ToImage()`：把 PIL 图像或 NumPy 数组变成 **uint8 的 CHW 张量**（通道在前，取值 0–255）。
- `ToDtype(torch.float32, scale=True)`：转成 `float32`，并把 0–255 线性映射到 0–1。

必须先用 `ToImage()`：`ToDtype` 期望输入已经是张量。归一化到 [0,1] 并非理论上的必需，但能让激活值和梯度处在更稳定的量级，对后面训练更友好。旧教程里的 `transforms.ToTensor()` 大致等价于这两步的组合；`v2` 是 torchvision 0.15 之后推荐的新写法，好处是变换可以作用在张量上、且能保留更多元信息（如视频、mask）。

#### 2）download 的真实行为

数据会被下载到 `data/FashionMNIST/raw/`，并做校验；文件已存在且完整时会**跳过下载**。所以第二次运行时那一大段进度条不会再出现。26.4M 是训练集压缩包，4.42M 是测试集。

#### 3）两个容易忽略的坑

- `root="data"` 是**相对当前工作目录**，不是相对脚本文件。换个目录启动脚本就会重新下载一遍。稳妥写法：`root = Path(__file__).parent / "data"`。
- `train=True` 和 `train=False` 两个对象共享同一个 `root`，它们各自独立做文件检查；这不是重复下载，但会让你误以为下载了两份。

另外，`transform` 只作用于图像，`target_transform` 只作用于标签 —— 这两个参数在很多自定义数据集里都会被写错位置，后面第 04 节还会遇到。

## 03 遍历与可视化数据集

*Iterating and Visualizing the Dataset*

我们可以像操作列表一样手动索引数据集：`training_data[index]`。下面用 matplotlib 可视化训练数据中的一些样本。

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

![FashionMNIST 的 9 个随机样本，标注为 Sneaker、Coat、Sandal、Shirt、T-Shirt、Bag、Dress 等](images/fashionmnist-9-samples.png)

*随机抽取的 9 个训练样本：每格上方是类别名，图像是 28×28 灰度图*

### 💡 解读 03 · 能 `len()` 能 `[i]`，是因为实现了两个魔法方法

`len(training_data)` 调的是 `__len__`，`training_data[i]` 调的是 `__getitem__`。返回的永远是一个 **二元组** `(image, label)`，所以可以写 `img, label = training_data[sample_idx]`。

- `torch.randint(len(training_data), size=(1,))` 返回形状为 `(1,)` 的张量，`.item()` 取出其中的 Python 整数；直接 `random.randrange(len(ds))` 也一样。
- `figure.add_subplot(rows, cols, i)` 的参数顺序是**行、列、序号**，而序号 `i` 从 1 开始 —— 写成 `range(1, 10)` 正是为此。
- `img.squeeze()` 去掉长度为 1 的通道维：单张图是 `(1, 28, 28)`，压成 `(28, 28)` 才能直接 `imshow`。

#### 为什么值得先画图

训练前“看数据”是性价比最高的调试手段：能立刻发现标签错位（图和名字对不上）、变换把图像裁坏、类别严重不均衡等问题。这类错误不会让程序报错，只会让模型莫名其妙地学不好。

#### 一个性能提示

`Dataset` 是**惰性**的：每次 `training_data[i]` 都会重新从磁盘读取并解码图片。上面这个循环只画 9 张，所以无所谓；但如果在训练循环里逐张索引，磁盘 I/O 和解码会成为瓶颈 —— 这正是 `DataLoader` 要用多进程的原因（见第 05 节）。

`squeeze()` 的坑：它会压掉**所有**长度为 1 的维度。这里没问题，但如果你拿着一个 `batch_size=1` 的批去 `squeeze()`，N 维也会被压掉，后面就会莫名其妙地少一维。更稳妥的写法是显式取 `img[0]`。

## 04 为你的文件创建自定义数据集

*Creating a Custom Dataset for your files*

自定义的 `Dataset` 类必须实现三个函数：`__init__`、`__len__` 和 `__getitem__`。请看下面的实现：FashionMNIST 的图像存放在目录 `img_dir` 中，标签则单独存放在一个 CSV 文件 `annotations_file` 里。

接下来的几小节，我们会逐个拆解这些函数里发生了什么。

```python
import os
import pandas as pd
from torchvision.io import decode_image

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
```

### 💡 解读 04 · 三个函数，一个契约

这段代码看起来很短，但它定义了 PyTorch 数据集的全部契约。可以用一张表记住各自职责：

| 函数 | 谁在调用 | 职责 |
| --- | --- | --- |
| `__init__` | 你创建对象时，一次 | 记录路径、读元数据（CSV）、保存 transform |
| `__len__` | `len(ds)`、DataLoader 算批数、Sampler 取索引范围 | 返回样本总数 |
| `__getitem__` | 每次取一个样本（含每个 worker） | 读盘 → 解码 → 取标签 → 应用变换 → 返回元组 |

#### 为什么 `__init__` 里不读图片

因为那样会在数据集对象创建时把所有图片解进内存 —— 启动慢、内存爆、而且多进程时每个 worker 都要复制一份。**元数据（路径 + 标签）放内存，像素数据按需解码**，是自定义数据集最重要的一条设计原则。标签文件只是文本，读进 DataFrame 的开销可以忽略。

#### 返回值约定

`__getitem__` 返回 `(image, label)`，于是 `DataLoader` 的默认 `collate_fn` 会把 N 个样本堆成两个 batch 张量。返回值类型也决定了 batch 的形状：这里是 `(1, 28, 28)` 与标量，堆叠后就是 `(B, 1, 28, 28)` 与 `(B,)`。

### __init__

`__init__` 函数在实例化 `Dataset` 对象时运行一次。我们在其中初始化图像所在目录、标注文件，以及两个变换（下一节会详细展开）。

标签文件 `labels.csv` 的样子如下：

```csv
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

### 💡 解读 05 · 把“配置”和“数据”分开

四个参数其实就是四件配置：**标注在哪**、**图片在哪**、**图像怎么变换**、**标签怎么变换**。默认值写成 `None`，意味着“不额外变换”，这样同一个类既能用于训练（带数据增强），也能用于推理或可视化（原图直出）。

- `self.img_labels` 这一份 DataFrame 是整个数据集唯一的“全量状态”；之后无论多少 worker、取多少样本，都从它里面查。
- 如果你的数据没有 CSV，而是目录名即标签（如 `cat/xxx.jpg`），同样在 `__init__` 里 `os.walk` 一遍，把 `(路径, 标签)` 整理成列表即可 —— 原则不变：**只存元数据**。
- `__init__` 里放重活（解码、转张量）会让多进程失去意义：worker 是复制这个对象，不是重新执行 `__init__`。

### __len__

`__len__` 函数返回数据集中样本的数量。

```python
def __len__(self):
    return len(self.img_labels)
```

### 💡 解读 06 · 一个必须准确的数字

这个返回值不是“批数”，而是**样本数**。`DataLoader` 用它算出每轮有多少个 batch：`ceil(len / batch_size)`。5000 张图、`batch_size=64`，就是 79 个 batch（最后一个只有 8 张）。

常见的错误是返回 DataFrame 的“行数减一”（误以为不含表头）或返回文件数。长度**多报**会索引越界，**少报**会静默丢掉样本 —— 后者更隐蔽，训练集少了几百张图，你只会看到指标略微变差。所以定义完 `Dataset` 后，第一件事就是 `len(ds)` 和实际文件数对一遍。

### __getitem__

`__getitem__` 函数在给定索引 `idx` 处加载并返回数据集中的一个样本。它根据索引定位图像在磁盘上的位置，用 `decode_image` 将其转换为张量，从 `self.img_labels` 中的 CSV 数据取出对应的标签，对二者调用变换函数（如果适用），最后以元组形式返回张量图像和对应标签。

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = decode_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```

### 💡 解读 07 · 逐行拆解，以及三处值得改进的地方

**① 拼路径。**`os.path.join` 会自动处理分隔符，跨平台安全。注意 CSV 第一列是**文件名**，不是完整路径。

**② 解码。**`decode_image` 读文件并解码为 **uint8 的 CHW 张量**，通道在前（PyTorch 惯例），支持 JPEG/PNG/WebP。别把它和 OpenCV 的 BGR、或 PIL 的 HWC 混用，否则会出现“颜色看着对、形状却不对”的怪问题。这一步是 `__getitem__` 里最耗时的部分。

**③ 取标签。**`self.img_labels.iloc[idx, 1]` 取第 idx 行、第 2 列（标签列）。取出来是 `numpy.int64`，堆叠成 batch 后是 `int64` 张量 —— 正好是 `CrossEntropyLoss` 需要的类型。

**④ 条件变换。**`if self.transform:` 利用的是 `None` 为假值的特性，比 `is not None` 简洁。变换顺序是**先解码、再变换**：transform 收到的是张量，所以自定义 transform 里不要假设输入是 NumPy 数组。

#### 三处“按官方示例能跑，但换个数据就出事”的地方

- **列序依赖：**`iloc[idx, 0]` 按**列的位置**取值，不是列名。别人往 CSV 里插一列（比如 `id`），代码不会报错，只会把所有图片名和标签错位——这是最经典的静默 bug。更稳的写法是按列名取：`row = df.iloc[idx]; img_path = os.path.join(self.img_dir, row["filename"])`，或在 `__init__` 里用 `usecols` 固定列。
- **标签类型：**若 `target_transform` 返回 `float`（例如误用了 `ToDtype(float32)`），`CrossEntropyLoss` 会在训练时报类型错误，且报错信息与数据无关，很难定位。分类任务的标签保持 `int64`。
- **错误处理：**多进程下 `__getitem__` 抛出的异常堆栈会被截断得面目全非。建议在解码前显式检查文件是否存在，并抛出带路径的错误信息：`raise FileNotFoundError(img_path)`。排查时能省下很多时间。

**扩展方向：**多标签分类返回 `float` 向量即可；语义分割返回 `(image, mask)`，此时 `transform` 必须同时作用于两者（`v2` 的变换支持传 dict 参数，正好解决坐标对齐问题）；如果数据是流式、无法随机索引（日志、超大数据集），请改用 `IterableDataset` 并实现 `__iter__`。

## 05 用 DataLoader 为训练准备数据

*Preparing your data for training with DataLoaders*

`Dataset` 一次只取一个样本的特征和标签。而训练模型时，我们通常希望以“小批量”（minibatch）的方式送入样本、在每个 epoch 重新打乱数据以减少模型过拟合，并使用 Python 的多进程来加快数据读取。

`DataLoader` 是一个可迭代对象，它用一个简洁的 API 为我们抽象了上述复杂度。

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

### 💡 解读 08 · DataLoader 到底替你做了什么

三个参数只暴露了冰山一角，背后它做了四件实事：

- **取索引**：由 Sampler 决定顺序。`shuffle=True` 时用 `RandomSampler`，每个 epoch 重新洗牌；`shuffle=False` 时按 `0..N-1` 顺序取。
- **切批**：把索引序列按 `batch_size` 切开，得到 938 个 batch（60000 / 64 = 937.5，最后一批 32 个）。
- **并行取数**：`num_workers` 个 worker 进程各自调用 `__getitem__`，主进程只负责汇总。默认 `num_workers=0`，也就是不并行。
- **拼装**：默认 `collate_fn` 把一批 `(image, label)` 堆成两个张量 —— 这就是上一节返回元组的意义。

#### 常用参数速查

| 参数 | 默认值 | 说明与建议 |
| --- | --- | --- |
| `batch_size` | 1 | 太小梯度噪声大、吞吐低；太大显存吃紧、泛化可能变差。64/128/256 是常见起点 |
| `shuffle` | False | **训练集必须开**；验证/测试集保持 False，便于逐样本对比与复现 |
| `num_workers` | 0 | 0=主进程。数据解码是 CPU 密集的，GPU 训练时通常设 4–8（不超过 CPU 核数） |
| `pin_memory` | False | GPU 训练设为 True，配合 `batch.to(device, non_blocking=True)` 可加速拷贝 |
| `drop_last` | False | 丢掉最后一个不满的 batch。含 BatchNorm 且 batch 太小时建议开，避免“最后一个 batch 只有 1 个样本”报错 |
| `collate_fn` | default_collate | 样本长度不一致（如文本）时必须自定义 |
| `sampler` | — | 自定义采样策略（如加权采样处理类别不均衡、分布式训练的分片） |
| `generator` | — | 传入带种子的 `torch.Generator` 可控制 shuffle 的随机性，保证实验可复现 |

两个实践提醒：**①** `num_workers > 0` 时，Windows 与 macOS 需要把启动代码放在 `if __name__ == "__main__":` 下，否则会无限递归创建进程；Jupyter 里多进程也有坑，通常先设 0 调试。**②** 每个 worker 会 fork 一份数据集对象的**引用**（Linux 上写时复制），但如果你的 `__init__` 里塞了几百 MB 的数组，内存占用仍会显著上升。

最后，教程里测试集也写了 `shuffle=True` —— 对指标计算没有影响（只是顺序不同），但会破坏“第 i 个预测对应第 i 个样本”的直觉，也不好做错误案例分析。实际项目里测试集建议 `shuffle=False`。

## 06 遍历 DataLoader

*Iterate through the DataLoader*

我们已经把数据集加载进 `DataLoader`，现在可以按需遍历它。下面的每次迭代都会返回一批 `train_features` 和 `train_labels`（分别包含 `batch_size=64` 个特征和标签）。因为我们指定了 `shuffle=True`，所以在遍历完所有 batch 之后，数据会被重新打乱（若想更精细地控制数据加载顺序，可以了解 [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)）。

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

```text
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 2
```

![DataLoader 取出的第一个样本：一张 28×28 的灰度服装图片](images/fashionmnist-1-sample.png)

*这是 `train_dataloader` 第一个 batch 里的第 0 个样本，标签为 2（Pullover，套头衫）*

### 💡 解读 09 · 读懂这两个形状

`[64, 1, 28, 28]` 是 PyTorch 的 **NCHW** 布局：64 个样本、1 个通道（灰度）、高 28、宽 28。`[64]` 是 64 个标签，每个 batch 一个标量。记住这个形状对后面搭模型至关重要：

- 全连接网络（如 `nn.Linear`）需要展平：`x.view(x.size(0), -1)` → `[64, 784]`。
- 卷积网络（`nn.Conv2d`）直接接收 `[64, 1, 28, 28]`，无需展平。
- 若把 `batch_size` 改成 32，只有第一维变化；如果哪天看到第二维不是 1，那多半是 `transform` 把通道顺序弄反了。

#### `next(iter(loader))` 的正确用法

`iter(loader)` 每次都**新建**一个迭代器，并触发一次新的 shuffle。所以取“下一批”的正确写法是：

```python
it = iter(train_dataloader)   # 一次 epoch 的迭代器
first  = next(it)             # 第 1 批
second = next(it)             # 第 2 批 —— 同一个迭代器上继续
```

写成 `next(iter(loader))` 两次，拿到的是两个独立 epoch 的第一批，对“连续查看几批”这类需求是错的；教程里只用一次，所以没问题。

#### 训练循环的真正写法

教程为了演示只取了一批，真实训练是双层循环：

```python
for epoch in range(num_epochs):
    for X, y in train_dataloader:          # 遍历完一轮 = 一个 epoch
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

内层 `for` 每轮结束就自动结束这一 epoch，**不需要你手动重置迭代器** —— `for` 循环每次都会调用 `iter()`，于是又洗一次牌。这正是 `shuffle=True` 生效的位置。

#### 细节

- 最后一批可能不满 64 个（本例 32 个），所有 batch 相关的张量操作都要能处理可变的第一维。
- `train_labels[0]` 是 0 维张量，`print` 显示为 `2`；要当 Python 整数用，请 `label.item()`。
- `plt.show()` 会阻塞程序，放在训练循环里会让训练慢一个数量级，仅用于调试。

## 07 延伸阅读

*Further Reading*

- [torch.utils.data API](https://pytorch.org/docs/stable/data.html) —— `Dataset`、`DataLoader`、`Sampler`、`collate_fn` 的完整文档。

*原文脚本总运行时间约 6.876 秒（不含首次下载与可视化窗口交互）。*

## 08 附：速查表与常见陷阱

*Cheat sheet*

### 三句话总结

**1.** `Dataset` = 索引式数据源，实现 `__init__` / `__len__` / `__getitem__`，`__getitem__` 负责“读一个样本并返回 `(特征, 标签)`”。  
**2.** `DataLoader` = 批处理 + 打乱 + 多进程 + 拼装，训练循环只跟它打交道。  
**3.** 元数据进内存、像素按需解码 —— 这是自定义数据集性能的分水岭。

### 症状 → 病因对照表

| 你观察到的现象 | 大概率原因 |
| --- | --- |
| GPU 利用率忽高忽低、训练很慢 | `num_workers=0` 或 worker 太少，数据解码成为瓶颈；也可能是 `pin_memory` 未开 |
| 最后一个 batch 报 BatchNorm 相关错误 | 最后一批只有 1 个样本，设 `drop_last=True` |
| loss 不降、模型像在“瞎猜” | 图像与标签错位：检查 CSV `iloc` 列序，或 transform 是否把图弄坏 |
| 训练 loss 很好、验证很差 | `shuffle` 没开、或数据增强只作用在训练集（后者是正确做法） |
| `CrossEntropyLoss` 报 dtype 错误 | 标签被 `target_transform` 转成了 float，应保持 `int64` |
| 多进程下内存暴涨 / 卡死 | 数据集对象太大被复制；Windows 忘记 `if __name__ == "__main__"` |
| 换目录运行就重新下载数据 | `root` 是相对工作目录的；改用绝对路径或 `Path(__file__).parent` |
| 取出的图颜色/通道不对 | `decode_image` 给的是 CHW（RGB），OpenCV 是 BGR（HWC），PIL 是 HWC |

### 读完后可以直接上手的一件事

把 `CustomImageDataset` 复制出来，改三处：`__init__` 里换成你自己 CSV 的列名（别用 `iloc` 硬编码下标）、`__getitem__` 里按列名取文件名、标签保持 `int64`。然后先用 `len(ds)` 和文件数对账，再画 9 张图确认图文对应 —— 这两步能挡掉后面 80% 的“模型不收敛”问题。

### ✅ 一句话记住

**Dataset 决定“一个样本长什么样”，DataLoader 决定“一批样本怎么来”。**前者写一次、复用一生；后者调参数、按硬件和任务微调。


---

*本文译自 PyTorch 官方教程 [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)（Last Updated: 2026-05-07），代码与图像版权归 PyTorch 基金会所有；中文翻译与「解读」为附加内容，供学习参考。*

*术语约定：`Dataset`、`DataLoader`、`batch`、`epoch`、`transform` 保留英文（社区通用叫法），首次出现时给出中文说明。*
