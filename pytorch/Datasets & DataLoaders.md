# Datasets & DataLoaders

Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allow you to use pre-loaded datasets as well as your own data. D
- **Dataset** stores the samples and their corresponding labels
- **DataLoader** wraps an iterable around the Dataset to enable easy access to the samples.

PyTorch domain libraries provide a number of pre-loaded datasets (such as `FashionMNIST`) that subclass torch.utils.data.Dataset and implement functions specific to the particular data. They can be used to prototype and benchmark your model.

## Loading a Dataset

> Here is an example of how to load the `Fashion-MNIST` dataset from TorchVision. 

- Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

We load the FashionMNIST Dataset with the following parameters:

        `root` is the path where the train/test data is stored,

        `train` specifies training or test dataset,

        `download=True` downloads the data from the Internet if it’s not available at `root`.

        `transform` and `target_transform` specify the feature and label transformations

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

**FashionMNIST 的参数**

- root="data"：指定数据集下载和保存的路径。运行后，你的当前目录下会自动生成一个名为 data 的文件夹。

- train=True / train=False：分别代表加载训练集（6万张图）和测试集（1万张图）。

- download=True：如果 data 文件夹里没有数据集，它会自动从网上下载。（首次运行需要联网下载，可能会稍微卡一会儿）。

- transform 数据转换
```python
transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
```

等价于旧版本的 transforms.ToTensor()。它做了两件极其重要的事情：

    v2.ToImage()：把原本 PIL 格式的图片，转换成 PyTorch 能够处理的 Tensor（张量） 格式。

    v2.ToDtype(torch.float32, scale=True)：

        转类型：把数据类型转为 32 位浮点数（float32），这是神经网络训练的标准数据类型。

        归一化（scale=True）：FashionMNIST 原本的像素值是 0~255 的整数。这个参数会自动把像素值除以 255，压缩到 0.0 ~ 1.0 之间。这能极大加快模型收敛速度。

Out
```
  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 32.8k/26.4M [00:00<01:53, 232kB/s]
  0%|          | 65.5k/26.4M [00:00<01:54, 230kB/s]
  0%|          | 98.3k/26.4M [00:00<01:54, 230kB/s]
  1%|          | 164k/26.4M [00:00<01:21, 321kB/s]
  1%|          | 262k/26.4M [00:00<00:57, 453kB/s]
  2%|▏         | 459k/26.4M [00:00<00:33, 768kB/s]
  3%|▎         | 852k/26.4M [00:00<00:18, 1.42MB/s]
  6%|▋         | 1.70M/26.4M [00:01<00:08, 2.87MB/s]
 13%|█▎        | 3.34M/26.4M [00:01<00:04, 5.56MB/s]
 25%|██▌       | 6.65M/26.4M [00:01<00:01, 11.0MB/s]
 40%|███▉      | 10.6M/26.4M [00:01<00:00, 16.5MB/s]
 55%|█████▍    | 14.5M/26.4M [00:01<00:00, 19.9MB/s]
 69%|██████▉   | 18.4M/26.4M [00:01<00:00, 21.5MB/s]
 84%|████████▍ | 22.2M/26.4M [00:01<00:00, 24.0MB/s]
 99%|█████████▉| 26.1M/26.4M [00:02<00:00, 25.0MB/s]
100%|██████████| 26.4M/26.4M [00:02<00:00, 12.3MB/s]

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 209kB/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 208kB/s]

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|          | 32.8k/4.42M [00:00<00:19, 230kB/s]
  1%|▏         | 65.5k/4.42M [00:00<00:18, 230kB/s]
  3%|▎         | 131k/4.42M [00:00<00:12, 334kB/s]
  4%|▍         | 197k/4.42M [00:00<00:11, 384kB/s]
  9%|▉         | 393k/4.42M [00:00<00:05, 742kB/s]
 16%|█▋        | 721k/4.42M [00:00<00:02, 1.27MB/s]
 33%|███▎      | 1.44M/4.42M [00:00<00:01, 2.51MB/s]
 64%|██████▍   | 2.85M/4.42M [00:01<00:00, 4.85MB/s]
100%|██████████| 4.42M/4.42M [00:01<00:00, 3.86MB/s]

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 30.2MB/s]
```

## Iterating and Visualizing the Dataset

We can index `Datasets` manually like a list: `training_data[index]`. We use `matplotlib` to visualize some samples in our training data.

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

## Creating a Custom Dataset for your files

A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. Take a look at this implementation; the FashionMNIST images are stored in a directory img_dir, and their labels are stored separately in a CSV file annotations_file.

In the next sections, we’ll break down what’s happening in each of these functions.

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

__init__

The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the images, the annotations file, and both transforms (covered in more detail in the next section).

The labels.csv file looks like:

tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9

def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

__len__

The __len__ function returns the number of samples in our dataset.

Example:

def __len__(self):
    return len(self.img_labels)

__getitem__

The __getitem__ function loads and returns a sample from the dataset at the given index idx. Based on the index, it identifies the image’s location on disk, converts that to a tensor using decode_image, retrieves the corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicable), and returns the tensor image and corresponding label in a tuple.

def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = decode_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label

Preparing your data for training with DataLoaders

The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity for us in an easy API.

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

Iterate through the DataLoader

We have loaded that dataset into the DataLoader and can iterate through the dataset as needed. Each iteration below returns a batch of train_features and train_labels (containing batch_size=64 features and labels respectively). Because we specified shuffle=True, after we iterate over all batches the data is shuffled (for finer-grained control over the data loading order, take a look at Samplers).

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

data tutorial

Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 2
