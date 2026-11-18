#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 知识蒸馏脚本 —— 逐段注释 / 结构剖析 / 体检版
=======================================================
基于 PyTorch 官方教程 "Knowledge Distillation"（teacher 大模型 -> student 小模型）。

--------------------------------------------------------------------
零、整体数据流（先看这个，再看代码）
--------------------------------------------------------------------
    CIFAR-10 (5 万训练图, 32x32x3, 10 类)
        |  训练集: RandomCrop(32,4) + Flip + ToTensor + Normalize（验证/测试不加增强）
        v
    DataLoader (batch=128)
        |
        +--> [第 7.1 步] 训练 Teacher  DeepNN   (1,186,986 参数) --> 准确率 ~88-90%（加增强后）
        |
        +--> [第 7.2 步] 训练 Student  LightNN  (  267,738 参数) --> 准确率 ~80%   <- 基线
        |
        +--> [第 7.3 步] Soft-Label KD: 学生同时学 "老师的软概率" + "真实标签" --> ~78%  <- 核心
        |
        +--> [第 7.4 步] 特征蒸馏 A: 对齐 Flatten 后的隐藏特征 (Cosine)      <- 示例分支（main 未调用）
        |
        +--> [第 7.5 步] 特征蒸馏 B: 对齐卷积特征图          (MSE)          <- 示例分支（main 未调用）
        |
        +--> [每段训练结束] 混淆矩阵 + 每类 P/R/F1 + TensorBoard 曲线/图片/报表

--------------------------------------------------------------------
一、代码分块表（"哪几段在干什么"）
--------------------------------------------------------------------
    第 0 块  设备选择                                 [核心]
    第 1 块  运行时开关(环境变量)                     [核心/可简化]
    第 2 块  数据 pipeline（增强 + val 切分）         [核心/已改进]
    第 3 块  模型定义(6 个类)                         [核心 2 个 + 示例 4 个]
    第 4 块  train()：调度/早停/选模型 + test()        [核心/已改进]
    第 5 块  三种蒸馏损失函数（KD 已同步升级）        [核心 1 个 + 示例 2 个]
    第 6 块  test_multiple_outputs()                  [只被示例分支使用]
    第 7 块  评估：准确率 + 混淆矩阵(matplotlib)      [新增]
    第 8 块  日志 + TensorBoard + 实验跟踪            [新增]
    第 9 块  main() 编排 3 段训练 + 汇总              [核心]
    第 10 块 入口守卫                                 [核心]
    （行号会随以后改动漂移，这里只列块名，按注释里的标题定位即可）

--------------------------------------------------------------------
二、标注说明
--------------------------------------------------------------------
    [核心]  训练必需，删掉就跑不起来或失去意义
    [示例]  官方教程的演示分支，整段删除不影响主流程（第 7.3 步照样能跑）
    [调试]  print / 形状检查，可删，只影响你肉眼看信息
    [陷阱]  容易踩坑的地方，注释里解释原因
    [修正]  相对你原文件的行为改动（文件末尾 CHANGELOG 有清单）

--------------------------------------------------------------------
三、运行方式
--------------------------------------------------------------------
    python CIFAR-10-improved.py                             # 默认 30 epoch x 3 段（增强+cosine）
    EPOCHS=1 LIMIT_TRAIN=1000 NUM_WORKERS=0 python 这个文件  # 冒烟测试（1 分钟内）
    RUN_NAME=exp_T4 python 这个文件                         # 指定实验名（默认时间戳）
    RUNS_DIR=/data/runs python 这个文件                     # 日志根目录（默认 ./runs/）
    CIFAR10_ROOT=/data/cifar python 这个文件                # 数据目录（默认 ./data/）

    # ---- 提分相关的旋钮（都有默认值，不改也能跑）----
    OPTIMIZER=sgd LR=0.05 python 这个文件                   # 换 SGD+momentum（泛化常略好）
    LABEL_SMOOTHING=0.0 AUGMENT=0 python 这个文件           # 消融：关掉平滑/增强看掉多少分
    VAL_SIZE=2000 PATIENCE=5 python 这个文件                # 验证集小一点、早停激进一点
    T=4 SOFT_WEIGHT=0.3 python 这个文件                     # 换蒸馏温度/权重

四、本次新增的三件事（评估 / 可视化 / 实验跟踪）
--------------------------------------------------------------------
    评估      每阶段训练完算混淆矩阵 + 每类 P/R/F1，存 PNG，同时进 TensorBoard
              （matplotlib 渲染，官方教程同款：imshow + colorbar + 45° 标签）
    可视化    SummaryWriter 记录曲线（train/loss、eval/acc、lr、耗时），
              混淆矩阵进 IMAGES 面板，分类报表进 TEXT 面板，最后进 HPARAMS 面板
    实验跟踪  runs/<run_name>/ 三件套：tfevents + metrics.csv + train.log，
              config.json 记下超参/环境/git commit，改一组参数就是一个新 run

    产物目录：
        runs/<run_name>/
        ├── events.out.tfevents.*                     TensorBoard 曲线（主日志）
        ├── metrics.csv                               每个 epoch 一行（Excel 可读）
        ├── train.log                                 带时间戳的文本日志
        ├── config.json                               超参 + 环境 + git 快照
        ├── {teacher,student,kd}_confusion_matrix.png      混淆矩阵（计数）
        ├── {teacher,student,kd}_confusion_matrix_norm.png 混淆矩阵（按行归一化）
        └── curves.png                                 loss/acc 曲线静态图

    看结果：
        tensorboard --logdir runs --port 6006          # 浏览器开 http://localhost:6006
        python export_tb.py --logdir runs --summary    # 不开浏览器，直接列 tag/点数
        python export_tb.py --logdir runs --out exp.csv

五、matplotlib（必需依赖）
--------------------------------------------------------------------
    第 7 块直接用 matplotlib 画混淆矩阵和曲线图，所以它是必需依赖：

        conda install -c conda-forge matplotlib      # 当前环境（本文件验证于 3.11.2）
        pip install matplotlib                       # 或者用 pip

    没装会在 import 阶段报 ModuleNotFoundError（上一版用 Pillow 手绘兜底，
    这一版按你的环境删掉了那 90 行兜底代码，画出来的图就是官方教程的样子）。

    两个和 matplotlib 有关的坑（代码里已经处理）：
      * 无图形界面（服务器 / 容器 / SSH）必须 matplotlib.use("Agg")，
        否则 import 阶段就会去连 X11 然后崩掉。
      * 默认缓存目录 ~/.config/matplotlib 不可写时会警告且每次 import 变慢，
        脚本用 MPLCONFIGDIR 指到临时目录；你自己设了 MPLCONFIGDIR 就不覆盖。
      * 图里的文字全部用英文：matplotlib 自带的 DejaVu 字体没有中文字形，
        写中文会变成一排豆腐块（并刷一屏 UserWarning）。注释里用中文没关系。

六、参考的官方源码 / 文档（本文件按这些写的）
--------------------------------------------------------------------
    SummaryWriter / add_scalar / add_image / add_text / add_hparams
        https://pytorch.org/docs/stable/tensorboard.html
    PyTorch 官方 TensorBoard 教程（含用混淆矩阵可视化的那一节）
        https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    sklearn 混淆矩阵 / 分类报表（本文件实际用的实现）
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    torchmetrics 的 MulticlassConfusionMatrix（torch 生态等价物，本机未装）
        https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html
    知识蒸馏原始论文（温度 T、软标签损失）
        https://arxiv.org/abs/1503.02531
"""

import csv
import io
import json
import logging
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image                             # 混淆矩阵/曲线图 -> TensorBoard 用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter     

# ============================================================
# 第 0 块：设备选择                                       
# ============================================================
# 目的：拿到一个 device 字符串，让后面所有 .to(device) 都能用。
# 顺序 torch.accelerator > cuda > mps > cpu，是为了兼容老版本 torch：
#   * torch.accelerator 是 2.5+ 才有的新 API，所以用 hasattr 探测；
#   * mps 只在 Apple 芯片 Mac 上存在，且老版本 torch 没有 torch.backends.mps，
#     所以用 getattr 做保护，不会在别的机器上报 AttributeError。
# 在你这台机器（无 CUDA）上会走到 else 分支，打印 "Using cpu device"。
if hasattr(torch, "accelerator") and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
elif torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
if device.startswith("cuda"):
    # [改进] 让 cuDNN 自动挑最快的卷积算法：输入尺寸固定（32x32）时收益明显，
    # 代价是第一次 forward 会慢一点。CPU/MPS 上这行不起作用。
    torch.backends.cudnn.benchmark = True
print(f"Using {device} device")

# ============================================================
# 第 1 块：运行时开关（环境变量可覆盖）                   
# ============================================================
# 这一块是"让脚本能直接跑"的工程胶水。想精简可以只用常量，
# 但保留它的好处：不用改代码就能换 epoch 数 / 换数据盘 / 换 worker 数。
#
# NUM_WORKERS : DataLoader 子进程数。
#   Linux 上给 2 通常能加速，但 CPU 训练 + 小数据集时 0 也完全够。
NUM_WORKERS = 0 if os.name == "nt" else int(os.environ.get("NUM_WORKERS", "2"))

# DATA_ROOT   : 数据集落盘位置。默认放在"本脚本所在目录/data"，
DATA_ROOT = os.environ.get(
    "CIFAR10_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)

# SCRIPT_DIR  : 本脚本所在目录。data/、runs/ 等相对产物都以它为准，
#               这样在别的 cwd 下运行也不会把日志丢到陌生的地方。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# RUNS_DIR    : 实验记录根目录，默认 <脚本目录>/runs，可用环境变量覆盖。
RUNS_DIR = os.environ.get("RUNS_DIR", os.path.join(SCRIPT_DIR, "runs"))

# EPOCHS      : 被 main() 里 3 段训练共用。注意总耗时 = 3 x EPOCHS 个 epoch！
#   [改进] 默认从 10 提到 30：加增强后模型需要更多轮才收敛，30 轮是 CIFAR 上的常规量级
#   （GPU 上 teacher 每轮 ~7s，30 轮三阶段加起来约 6 分钟）。
EPOCHS = int(os.environ.get("EPOCHS", "30"))

# ---- 下面是这次为"提分"新增的旋钮（都能用环境变量覆盖）----
# BATCH_SIZE / LEARNING_RATE / WEIGHT_DECAY : 常规训练超参
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LEARNING_RATE = float(os.environ.get("LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "5e-4"))

# OPTIMIZER : adamw（默认，收敛快）或 sgd（CIFAR 上泛化常略好，lr 要调到 0.05~0.1）
OPTIMIZER = os.environ.get("OPTIMIZER", "adamw")

# LABEL_SMOOTHING : 交叉熵的标签平滑，防止模型对单标签过度自信（典型 0.1）
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.1"))

# WARMUP_EPOCHS / MIN_LR_RATIO : warmup 轮数，以及 cosine 退火的最低 lr 比例
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "2"))
MIN_LR_RATIO = float(os.environ.get("MIN_LR_RATIO", "0.01"))

# VAL_SIZE : 从 5 万训练图里切多少张做验证集（用于选模型 + 早停，不看 test）
VAL_SIZE = int(os.environ.get("VAL_SIZE", "5000"))

# PATIENCE : 验证集准确率连续多少轮不提升就早停
PATIENCE = int(os.environ.get("PATIENCE", "8"))

# AUGMENT : 1 = 训练集做 RandomCrop+Flip（默认），0 = 关掉（用来做消融对比）
AUGMENT = os.environ.get("AUGMENT", "1") != "0"

# SEED : 统一随机种子（三段训练共用，保证 nn_light 与 new_nn_light 初始化一致）
SEED = int(os.environ.get("SEED", "42"))

# 蒸馏三件套：温度 T、软标签权重、硬标签权重（main / config / run 名共用同一组值，
# 避免"实验名写 T4、实际跑 T2"这种事）
TEMPERATURE_T = float(os.environ.get("T", "2.0"))
SOFT_WEIGHT = float(os.environ.get("SOFT_WEIGHT", "0.25"))
CE_WEIGHT = float(os.environ.get("CE_WEIGHT", "0.75"))

# PIN_MEMORY  : 锁页内存只在 CUDA 上有意义（加速 H2D 拷贝），CPU 上开了纯浪费。
PIN_MEMORY = device == "cuda"

# ============================================================
# 第 2 块：数据 pipeline（训练增强 + 验证集切分）            [改进]
# ============================================================
# 相对原版改了三件事，都是"提高准确率"里性价比最高的部分：
#
#  1) 训练集加数据增强：RandomCrop(32, padding=4) + RandomHorizontalFlip。
#     这是 CIFAR 上最经典的两个增强，几乎不增加计算量，教师能从 ~80% 提到 ~88-90%。
#     原版只做 ToTensor + Normalize，模型 5 个 epoch 就把 5 万张图背下来了（train acc 95%、
#     test acc 75%），属于典型的过拟合——增强就是治这个的。
#
#  2) 从 5 万张训练图里切出 VAL_SIZE 张做验证集，形成 train/val/test = 45k/5k/10k。
#     原版每轮直接拿 test 报准确率，等于用测试集反复挑模型，报出来的数偏乐观；
#     现在调参、选最佳 epoch、早停都只看 val，test 只在最后用一次。
#
#  3) 验证/测试集只用 ToTensor + Normalize，绝不做增强，否则指标不可比。
#
# [陷阱] 同一个 CIFAR10 对象不能同时当"增强版训练集"和"干净验证集"：
#   它只有一个 transform。所以这里开两份实例（train=True 各一次），用统一的索引表切分，
#   保证"增强版的第 i 张"和"干净版的第 i 张"是同一张图。
MEAN = [0.485, 0.456, 0.406]      # ImageNet 统计量（沿用教程，没问题）
STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# AUGMENT=0 时训练集也用干净版：用来做"增强到底值多少分"的消融实验
if AUGMENT:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # 四周补 0 再随机裁回 32x32
        transforms.RandomHorizontalFlip(p=0.5),    # 左右翻转：猫变"镜面猫"，标签不变
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
else:
    train_transform = eval_transform

train_full_aug = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True,
                                  transform=train_transform)
train_full_eval = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True,
                                   transform=eval_transform)
test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True,
                                transform=eval_transform)

# ---- 2.2 train / val 切分 ---------------------------------
# 用固定种子的 randperm：同一个 seed 下 val 永远是那 5000 张，
# 换超参重跑时训练集/验证集完全一致，前后数字才可比。
_split_gen = torch.Generator().manual_seed(1234)
_perm = torch.randperm(len(train_full_aug), generator=_split_gen)
val_idx = _perm[:VAL_SIZE].tolist()
train_idx = _perm[VAL_SIZE:].tolist()

# [新增] LIMIT_TRAIN=N：只取 N 张训练图 + 同样 N 张验证图，用来 1 分钟跑通全流程；0 = 全量。
LIMIT_TRAIN = int(os.environ.get("LIMIT_TRAIN", "0"))
if LIMIT_TRAIN > 0:
    train_idx = train_idx[:LIMIT_TRAIN]
    val_idx = val_idx[:min(LIMIT_TRAIN, len(val_idx))]
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           range(min(LIMIT_TRAIN, len(test_dataset))))

train_dataset = torch.utils.data.Subset(train_full_aug, train_idx)
val_dataset = torch.utils.data.Subset(train_full_eval, val_idx)

# ---- 2.3 DataLoader --------------------------------------
# 训练集 shuffle=True（每轮重排 + 每轮重新随机裁剪），val/test shuffle=False（结果可复现）。
# drop_last=True：避免最后一批只有 1-2 张样本，BN 的 batch 统计会抖得很厉害。
# persistent_workers 只有 num_workers>0 时才能为 True，否则 torch 直接报错。
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=len(train_idx) >= 2 * BATCH_SIZE,
    persistent_workers=NUM_WORKERS > 0,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0,
)


# ============================================================
# 第 3 块：模型定义（6 个类）                      
# ============================================================
# 【张量形状速查表】输入统一是 [B, 3, 32, 32]
#
#   DeepNN  : 3x32x32 -conv128-> 32x32 -conv64-> 32x32 -pool-> 16x16
#             -conv64-> 16x16 -conv32-> 16x16 -pool-> 8x8
#             flatten = 32*8*8 = 2048  ->  Linear(2048,512) -> Linear(512,10)
#             参数量 1,186,986
#
#   LightNN : 3x32x32 -conv16-> 32x32 -pool-> 16x16 -conv16-> 16x16 -pool-> 8x8
#             flatten = 16*8*8 = 1024  ->  Linear(1024,256) -> Linear(256,10)
#             参数量   267,738   （约为 teacher 的 1/4.4）
#
# 这 6 个类的关系：
#   DeepNN / LightNN                          -> 用于第 7.1 / 7.2 / 7.3 步（主实验）


# ---- 3.1 Teacher：深网络 ---------------------------------  
class DeepNN(nn.Module):
    """Teacher。4 个 3x3 卷积 + 2 次 maxpool，最后接两层全连接。"""

    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        # features = 特征提取器（卷积部分）；classifier = 分类头（全连接部分）
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),    # [B,3,32,32]  -> [B,128,32,32]
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),   # [B,128,32,32]-> [B,64,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),          # -> [B,64,16,16]
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # -> [B,64,16,16]
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),    # -> [B,32,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),          # -> [B,32,8,8]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),      # 2048 = 32通道 x 8 x 8，本文件最易写错的地方
            nn.ReLU(),
            nn.Dropout(0.1),           # 只有 train 模式生效，eval 时自动关闭
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)          # [B,32,8,8]
        x = torch.flatten(x, 1)       # [B,2048]；dim=1 表示"保留 batch 维，其余拍平"
        x = self.classifier(x)        # [B,10]  (logits，未过 softmax)
        return x


# ---- 3.2 Student：浅网络 --------------------------------- 
class LightNN(nn.Module):
    """Student。2 个 3x3 卷积 + 2 次 maxpool + 两层全连接。"""

    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),     # [B,3,32,32]  -> [B,16,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),          # -> [B,16,16,16]
            nn.Conv2d(16, 16, kernel_size=3, padding=1),    # -> [B,16,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),          # -> [B,16,8,8]
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),      # 1024 = 16通道 x 8 x 8
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)          # [B,16,8,8]
        x = torch.flatten(x, 1)       # [B,1024]
        x = self.classifier(x)        # [B,10]
        return x



# ============================================================
# 第 4 块：训练 / 评测基元（LR 调度 + 早停 + 按 val 选模型）  [改进]
# ============================================================
# 相对原版的四处改动，都在这个函数里：
#   1. Adam -> AdamW(weight_decay) 或 SGD+momentum：原版 Adam 无正则，10 轮就过拟合。
#   2. 固定 lr -> warmup + cosine 退火：前期稳、后期小 lr 收敛更细，CIFAR 上通常 +1~2pp。
#   3. CE 加 label_smoothing：防止模型对单标签过度自信，和蒸馏的软标签也配合得更好。
#   4. 每轮在 val 上评测，按 val acc 保存"最好权重"，patience 轮不涨就早停；
#      训练结束把最好权重装回去，再在 test 上算最终指标（test 全程没参与选择）。


def build_optimizer(model, learning_rate):
    """按 OPTIMIZER 环境变量选优化器。"""
    if OPTIMIZER.lower() == "sgd":
        # SGD + momentum + nesterov：CIFAR 上泛化往往略好，但对 lr 更敏感（建议 0.05~0.1）
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                         nesterov=True, weight_decay=WEIGHT_DECAY)
    # AdamW：weight decay 与梯度更新解耦（Adam 的 L2 正则化和它不等价）
    return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer, epochs):
    """warmup（线性升温）+ cosine 退火到 learning_rate * MIN_LR_RATIO。"""
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:                     # 前几个 epoch 线性升到基准 lr
            return float(epoch + 1) / max(1, WARMUP_EPOCHS)
        progress = (epoch - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))     # 1.0 -> 0.0
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model, train_loader, val_loader, epochs, learning_rate, device,
          run, phase="model", test_loader=None):
    """训练一个模型，返回 (best_val_acc, test_acc)。

    参数
        val_loader : 每轮评测用的验证集（不增强），也是选模型/早停的依据
        test_loader: 只在最后用一次，算"干净"的测试准确率；不传就用 val
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = build_optimizer(model, learning_rate)
    scheduler = build_scheduler(optimizer, epochs)

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)   # set_to_none 比置 0 省一次写内存
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)     # 按样本加权，最后一批不满 batch 也准
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # 调度器按 epoch 走（LambdaLR 每调用一次 step 就推进一个 epoch）
        scheduler.step()

        # 逐 epoch 在验证集上评测（比 test 小一半，开销可接受）
        val_pred, val_label, val_loss = collect_predictions(model, val_loader, device)
        val_acc = 100.0 * float((val_pred == val_label).mean())

        log_epoch(run, phase, epoch, epochs, train_loss, train_acc,
                  val_loss, val_acc, optimizer.param_groups[0]["lr"], time.time() - t0)

        # ---- 按 val acc 保存最好权重 ----
        # 存 state_dict 的副本到 CPU：既不影响继续训练，也不会额外占显存
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                logger.info("[%s] val acc 连续 %d 轮没提升，提前停止（epoch %d/%d）",
                            phase, PATIENCE, epoch + 1, epochs)
                break

    if best_state is not None:
        model.load_state_dict(best_state)        # 恢复到 val 最好的那一轮

    acc, _ = final_eval(run, model, test_loader or val_loader, device, phase)
    return best_val, acc


def test(model, test_loader, device):
    """[保留] 原始教程式的评测：只打印一个准确率。

    新代码不再调用它（改用 final_eval 拿混淆矩阵），留着是为了你单独调试模型时方便。
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            correct += (model(inputs).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc


# ============================================================
# 第 5 块：知识蒸馏损失（软标签 KD）                        [核心]
# ============================================================
# 和第 4 块同步升级：AdamW + warmup/cosine + label smoothing + 按 val 选模型 + 早停。
# 蒸馏的数学没变：total = soft_weight * T^2 * KL(teacher||student) + ce_weight * CE。


def train_knowledge_distillation(teacher, student, train_loader, val_loader, epochs,
                                 learning_rate, T, soft_target_loss_weight,
                                 ce_loss_weight, device, run, phase="kd",
                                 test_loader=None):
    """Hinton 软标签蒸馏：学生同时拟合"老师的软概率"和"真实标签"。

    total_loss = 0.25 * 软标签损失 + 0.75 * 交叉熵
    软标签损失乘 T^2 是因为温度把梯度整体缩小了 1/T^2，不补回来就相当于把学习率调小。

    [修正 1] student.to(device)：原文件这里漏了，CUDA 上会 device mismatch。
    [修正 2] 软标签用 log_softmax 再 exp，而不是 softmax().log()：
             softmax 概率下溢成 0 时 log(0) = -inf，0 * (-inf) = NaN，训练直接崩。
    """
    # 硬标签这一项也用 label smoothing，和 teacher/student 的训练口径保持一致
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    student.to(device)                                   # [修正 1]
    student.train()
    teacher.eval()                                       # [陷阱] 老师必须 eval：BN/Dropout 不能动

    optimizer = build_optimizer(student, learning_rate)
    scheduler = build_scheduler(optimizer, epochs)

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        t0 = time.time()
        student.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # 老师只做前向，不建计算图（省显存，也避免误更新老师）
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            # [修正 2] 两边都走 log_softmax：数值稳定，等价于 softmax 后再 log
            teacher_log_prob = F.log_softmax(teacher_logits / T, dim=-1)
            soft_targets = teacher_log_prob.exp()            # 软标签（概率）
            student_log_prob = F.log_softmax(student_logits / T, dim=-1)

            # KL(teacher || student) 的按 batch 平均版本，乘 T^2 做温度补偿
            soft_targets_loss = torch.sum(
                soft_targets * (teacher_log_prob - student_log_prob)
            ) / soft_targets.size(0) * (T ** 2)

            label_loss = ce(student_logits, labels)
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        scheduler.step()

        val_pred, val_label, val_loss = collect_predictions(student, val_loader, device)
        val_acc = 100.0 * float((val_pred == val_label).mean())

        log_epoch(run, phase, epoch, epochs, train_loss, train_acc,
                  val_loss, val_acc, optimizer.param_groups[0]["lr"], time.time() - t0)

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                logger.info("[%s] val acc 连续 %d 轮没提升，提前停止（epoch %d/%d）",
                            phase, PATIENCE, epoch + 1, epochs)
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    acc, _ = final_eval(run, student, test_loader or val_loader, device, phase)
    return best_val, acc


# ============================================================
# 5.2 / 5.3 特征蒸馏示例分支（保留了实现，但 main() 里没有调用）
# ============================================================
# 这两个函数属于官方教程的"示例内容"，删掉不影响主流程。
# 它们没有接 TensorBoard 日志（要用的话，照 train_knowledge_distillation 里
# log_epoch/final_eval 的写法加两行即可）。


def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate,
                      hidden_rep_loss_weight, ce_loss_weight, device):
    """[示例] 特征蒸馏 A：余弦相似度对齐隐藏层 —— 第 7.4 步。

    不学老师的输出分布，而是要求"中间特征方向一致"：
        loss = w_feat * (1 - cos(student_feat, teacher_feat)) + w_ce * CE
    CosineEmbeddingLoss(target=1) 的公式正好是 1 - cos_sim，target=-1 则是 max(0, cos-margin)。
    这里用 target=1（全 1 向量，长度 = batch 大小）。
    """
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                # 只要第二个返回值 [B,1024]，老师的 logits 不要
                _, teacher_hidden_representation = teacher(inputs)

            student_logits, student_hidden_representation = student(inputs)

            # target=1 表示"希望两个向量同向"，最小化它就等于最大化余弦相似度。
            # [陷阱] 必须用 inputs.size(0) 而不是硬编码 128：最后一个 batch 可能不足 128。
            hidden_rep_loss = cosine_loss(
                student_hidden_representation,
                teacher_hidden_representation,
                target=torch.ones(inputs.size(0)).to(device),
            )

            label_loss = ce_loss(student_logits, labels)
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def train_mse_loss(teacher, student, train_loader, epochs, learning_rate,
                   feature_map_weight, ce_loss_weight, device):
    """[示例] 特征蒸馏 B：MSE 对齐卷积特征图 —— 第 7.5 步。

    loss = w_feat * MSE(student_feat_map, teacher_feat_map) + w_ce * CE
    两个特征图都是 [B,32,8,8]，所以能逐元素相减 —— 这也是 3.6 里 regressor
    必须把通道数从 16 升到 32 的原因。
    """
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)      # [B,32,8,8]

            student_logits, regressor_feature_map = student(inputs)   # [B,10], [B,32,8,8]

            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)
            label_loss = ce_loss(student_logits, labels)
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")



# ============================================================
# 第 7 块：评估 —— 准确率 + 混淆矩阵                    
# ============================================================
# 官方做法参考：
#   * matplotlib 画混淆矩阵 + SummaryWriter 记录 —— 官方 TensorBoard 教程就用这个
#     https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
#   * sklearn.metrics.confusion_matrix / classification_report
#     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#   * torchmetrics 的 MulticlassConfusionMatrix（等价的 torch 版实现）
#     https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html
#
# 依赖：matplotlib（conda install -c conda-forge matplotlib 或 pip install matplotlib）
#       sklearn（可选，没有就退回 numpy 自己数）
# 渲染结果统一是 PIL.Image，方便直接 add_image 进 TensorBoard。
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from sklearn.metrics import classification_report
except ImportError:                       # 没装 sklearn 时的纯 numpy 兜底
    sk_confusion_matrix = None
    classification_report = None

# ---- 7.0 matplotlib 初始化 -------------------------------------------
# matplotlib 默认把字体缓存写到 ~/.config/matplotlib；在只读 HOME / 容器里会警告
# 并且每次 import 都变慢，所以指定一个可写目录（外部已设 MPLCONFIGDIR 就不覆盖）。
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(tempfile.gettempdir(), "kd-matplotlib-cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib                     # noqa: E402  （必须在 MPLCONFIGDIR 之后 import）
matplotlib.use("Agg")                 # [陷阱] 服务器/无 X11 环境必须用 Agg，否则 import 就崩
import matplotlib.pyplot as plt       # noqa: E402
from matplotlib.ticker import MaxNLocator   # noqa: E402


@torch.no_grad()
def collect_predictions(model, data_loader, device):
    """把整个数据集跑一遍，收集 (预测, 真值, 平均loss)。

    返回 pred / label 都是 numpy 数组（sklearn 只吃 numpy）。
    """
    model.eval()                          # [陷阱] 忘了 eval()，Dropout/BN 会按训练模式跑，acc 偏低
    criterion = nn.CrossEntropyLoss()
    all_pred, all_label = [], []
    total_loss, total = 0.0, 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        total_loss += criterion(logits, labels).item() * labels.size(0)   # 按样本数加权
        all_pred.append(logits.argmax(dim=1).cpu())
        all_label.append(labels.cpu())
        total += labels.size(0)

    return torch.cat(all_pred).numpy(), torch.cat(all_label).numpy(), total_loss / total


def confusion_matrix_numpy(label, pred, num_classes=10):
    """sklearn 不可用时的等价实现：cm[真实, 预测] += 1。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (label, pred), 1)
    return cm


def build_confusion_matrix(label, pred):
    """优先用 sklearn（官方实现），没有就退回 numpy 版。"""
    if sk_confusion_matrix is not None:
        return sk_confusion_matrix(label, pred, labels=list(range(len(CLASS_NAMES))))
    return confusion_matrix_numpy(label, pred, num_classes=len(CLASS_NAMES))


def render_confusion_matrix(cm, class_names=CLASS_NAMES, normalize=False, title=None):
    """混淆矩阵 -> PIL.Image（matplotlib 渲染，和官方教程 plot_confusion_matrix 同思路）。

    行 = 真实类别，列 = 预测类别；颜色越深数量越多。
    normalize=True 时按行归一化，看"每类被判成了什么"更直观。
    """
    mat = cm.astype(np.float64)
    if normalize:                        # 按行归一化：每行 = 该类被分到各预测类的比例
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # [陷阱] 不旋转 45°，10 个类名会叠在一起看不清
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if title:
        ax.set_title(title, fontsize=11)

    # 深色格子里写白字，浅色写黑字，否则数字看不见
    threshold = mat.max() * 0.6
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = f"{mat[i, j] * 100:.0f}" if normalize else f"{int(mat[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > threshold else "black")

    fig.tight_layout()
    buf = io.BytesIO()                   # 存内存 -> PIL，避免依赖临时文件
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)                       # [陷阱] 不 close 会一路累积 figure，内存越跑越大
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def plot_curves(run, csv_path=None):
    """从 metrics.csv 画 loss / acc 曲线，存 PNG 并写进 TensorBoard。

    TensorBoard 的 SCALARS 面板已经能看曲线，这里额外出一张静态图，
    方便写报告 / 贴群里 / 存档（不依赖开着 TB 服务）。
    """
    csv_path = csv_path or run.csv_path
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    phases = [p for p in ("teacher", "student", "kd")
              if any(r["phase"] == p for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # [陷阱] 只跑 1 个 epoch 时 matplotlib 会把横轴缩成一个点，曲线直接看不见。
    # 所以统一加 marker（单点也能画出来），并把 x 轴刻度锁到真实 epoch 值上。
    epochs_seen = sorted({int(r["epoch"]) for r in rows})
    for phase in phases:
        sub = [r for r in rows if r["phase"] == phase]
        ep = [int(r["epoch"]) for r in sub]
        style = dict(marker="o", markersize=3, linewidth=1.5)
        axes[0].plot(ep, [float(r["train_loss"]) for r in sub], "--",
                     label=f"{phase} train", **style)
        axes[0].plot(ep, [float(r["val_loss"]) for r in sub], "-",
                     label=f"{phase} val", **style)
        axes[1].plot(ep, [float(r["train_acc"]) for r in sub], "--",
                     label=f"{phase} train", **style)
        axes[1].plot(ep, [float(r["val_acc"]) for r in sub], "-",
                     label=f"{phase} val", **style)

    axes[0].set_title("loss")
    axes[1].set_title("accuracy (%)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if len(epochs_seen) == 1:                     # 单点：左右各留半格
            ax.set_xlim(epochs_seen[0] - 0.5, epochs_seen[0] + 0.5)
            ax.set_xticks(epochs_seen)
        elif len(epochs_seen) <= 12:                  # epoch 是整数，别显示 1.2 / 1.4
            ax.set_xticks(epochs_seen)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=0.02)

    # [陷阱] 图里的文字统一用英文：matplotlib 自带 DejaVu 字体没有中文字形，
    # 写中文会变成一排豆腐块（并刷 UserWarning）。注释里用中文没关系。
    axes[1].set_xlabel("epoch (restarts at 1 for each phase)")

    fig.tight_layout()
    path = os.path.join(run.run_dir, "curves.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)

    run.image("curves/all", Image.open(path).convert("RGB"), 0)
    return path


# ============================================================
# 第 8 块：日志 + TensorBoard + 实验跟踪             
# ============================================================
# 三份产物，各有用途（都在 runs/<run_name>/ 下）：
#   events.out.tfevents.*  TensorBoard 事件文件：曲线、图片、报表都在里面
#   metrics.csv            每个 epoch 一行，Excel / pandas 直接读
#   train.log              带时间戳的文本日志，出问题先看它
#   config.json            超参 + 环境 + git commit 快照，用来复现"这一条是哪次跑的"
#
# 官方 API 参考：
#   https://pytorch.org/docs/stable/tensorboard.html
#   SummaryWriter(log_dir=...) / add_scalar / add_image / add_text / add_hparams
logger = logging.getLogger("kd")


def setup_logging(log_path, level=logging.INFO):
    """同时写终端和文件；重复调用不会重复挂 handler。"""
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # [陷阱] 别的库（tensorboard 依赖的 absl）会给 root logger 挂 handler，
    # 不关掉传播就会变成"每行打印两遍"。
    logger.propagate = False


def get_git_commit(directory):
    """记下 git commit，隔几天回来看实验时能对上代码版本。没有仓库就返回 unknown。"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=directory, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class RunTracker:
    """一次实验（run）的中间层：图形化走 TensorBoard，表格走 CSV。"""

    CSV_FIELDS = ["phase", "epoch", "train_loss", "train_acc",
                  "val_loss", "val_acc", "lr", "sec"]

    def __init__(self, run_dir, writer):
        self.run_dir = run_dir
        self.writer = writer
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writeheader()

    def scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def image(self, tag, pil_image, step):
        """add_image 接受 numpy，但要按 dataformats 说明是 CHW 还是 HWC。"""
        if self.writer is None:
            return
        arr = np.asarray(pil_image).transpose(2, 0, 1)      # HWC -> CHW
        self.writer.add_image(tag, arr, step, dataformats="CHW")

    def text(self, tag, markdown):
        if self.writer is not None:
            self.writer.add_text(tag, markdown, global_step=0)   # 官方签名是 global_step，不是 step

    def row(self, **kwargs):
        """追加一行指标到 metrics.csv（每 epoch 一行，崩溃了也不丢前面的）。"""
        row = {key: kwargs.get(key, "") for key in self.CSV_FIELDS}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writerow(row)


def init_run():
    """建 run 目录、初始化日志、写 config.json、开 SummaryWriter。"""
    runs_dir = os.environ.get("RUNS_DIR", os.path.join(SCRIPT_DIR, "runs"))
    if "RUN_NAME" in os.environ:
        run_name = os.environ["RUN_NAME"]
    else:
        # 默认名：时间 + 关键超参，光看目录名就知道跑了什么
        data_desc = "全量" if LIMIT_TRAIN == 0 else f"{LIMIT_TRAIN}张"
        run_name = (f"{time.strftime('%Y-%m-%d_%H%M')}_知识蒸馏"
                    f"_T{TEMPERATURE_T:g}_软{SOFT_WEIGHT:g}_{EPOCHS}轮_{data_desc}")
    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    setup_logging(os.path.join(run_dir, "train.log"))

    config = {
        "run_name": run_name,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": OPTIMIZER,
        "label_smoothing": LABEL_SMOOTHING,
        "warmup_epochs": WARMUP_EPOCHS,
        "min_lr_ratio": MIN_LR_RATIO,
        "augment": ("RandomCrop(32,4)+HorizontalFlip" if AUGMENT else "none"),
        "val_size": VAL_SIZE,
        "patience": PATIENCE,
        "seed": SEED,
        "temperature_T": 2.0,
        "soft_target_loss_weight": 0.25,
        "ce_loss_weight": 0.75,
        "matplotlib": matplotlib.__version__,
        "num_workers": NUM_WORKERS,
        "limit_train": LIMIT_TRAIN,
        "device": device,
        "data_root": DATA_ROOT,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": get_git_commit(SCRIPT_DIR),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    writer = SummaryWriter(log_dir=run_dir)      # 官方入口：torch.utils.tensorboard
    run = RunTracker(run_dir, writer)
    run.text("config", "```json\n" + json.dumps(config, indent=2, ensure_ascii=False) + "\n```")

    logger.info("=" * 72)
    logger.info("run_dir = %s", run_dir)
    logger.info("device  = %s | torch = %s | git = %s",
                device, torch.__version__, config["git_commit"])
    logger.info("看曲线：tensorboard --logdir %s", runs_dir)
    return run, writer


def log_epoch(run, phase, epoch, epochs, train_loss, train_acc,
              val_loss, val_acc, lr, sec):
    """一个 epoch 的指标同时写 TB、CSV、终端/文件日志。

    [改进] 逐 epoch 监控的是 val（验证集），不是 test：
            test 只在每段训练结束时用一次，避免"拿测试集挑模型"导致的乐观偏差。
    """
    run.scalar(f"train/{phase}/loss", train_loss, epoch)
    run.scalar(f"train/{phase}/acc", train_acc, epoch)
    run.scalar(f"val/{phase}/loss", val_loss, epoch)
    run.scalar(f"val/{phase}/acc", val_acc, epoch)
    run.scalar(f"train/{phase}/lr", lr, epoch)
    run.scalar(f"time/{phase}/epoch_sec", sec, epoch)
    run.row(phase=phase, epoch=epoch + 1, train_loss=round(train_loss, 6),
            train_acc=round(train_acc, 4), val_loss=round(val_loss, 6),
            val_acc=round(val_acc, 4), lr=lr, sec=round(sec, 2))
    logger.info("[%-8s] epoch %2d/%d | train loss %.4f acc %5.2f%% | "
                "val loss %.4f acc %5.2f%% | lr %.1e | %5.1fs",
                phase, epoch + 1, epochs, train_loss, train_acc,
                val_loss, val_acc, lr, sec)


def final_eval(run, model, test_loader, device, phase):
    """训练结束后：算准确率 + 混淆矩阵，存图、记 TB、生成每类报表。"""
    pred, label, test_loss = collect_predictions(model, test_loader, device)
    acc = 100.0 * float((pred == label).mean())
    cm = build_confusion_matrix(label, pred)

    img = render_confusion_matrix(cm, title=f"{phase} (counts)")
    img.save(os.path.join(run.run_dir, f"{phase}_confusion_matrix.png"))
    run.image(f"img/{phase}/confusion_matrix", img, 0)

    img_norm = render_confusion_matrix(cm, normalize=True,
                                       title=f"{phase} (row-normalized %)")
    img_norm.save(os.path.join(run.run_dir, f"{phase}_confusion_matrix_norm.png"))
    run.image(f"img/{phase}/confusion_matrix_norm", img_norm, 0)

    run.scalar(f"test/{phase}/acc_final", acc, 0)
    run.scalar(f"test/{phase}/loss_final", test_loss, 0)

    if classification_report is not None:
        report_text = classification_report(label, pred, target_names=CLASS_NAMES,
                                            digits=3, zero_division=0)
        run.text(f"report/{phase}", "```\n" + report_text + "\n```")
        logger.info("[%s] 每类指标：\n%s", phase, report_text)

        report_dict = classification_report(label, pred, target_names=CLASS_NAMES,
                                            digits=3, zero_division=0, output_dict=True)
        for name in CLASS_NAMES:                  # 每类 F1 单独记一条曲线，方便横向比
            run.scalar(f"test_per_class/{phase}/f1/{name}", report_dict[name]["f1-score"], 0)
    else:
        logger.warning("没装 sklearn，跳过 classification_report（混淆矩阵仍可用 numpy 版）")

    logger.info("[%s] 最终 acc = %.2f%% | 混淆矩阵：%s/%s_confusion_matrix.png",
                phase, acc, run.run_dir, phase)
    return acc, cm


def markdown_summary(results):
    """把各阶段准确率汇总成一张 markdown 表，既打印也进 TensorBoard 的 TEXT 面板。"""
    lines = ["| 阶段 | 说明 | 准确率 |", "| --- | --- | --- |"]
    for phase, desc, acc in results:
        lines.append(f"| {phase} | {desc} | {acc:.2f}% |")
    if len(results) >= 3:
        teacher_acc = results[0][2]
        student_acc = results[1][2]
        kd_acc = results[2][2]
        lines.append("")
        lines.append(f"* 学生相对老师：{kd_acc - teacher_acc:+.2f} pp")
        lines.append(f"* 蒸馏相对基线：{kd_acc - student_acc:+.2f} pp")
    return "\n".join(lines)


# ============================================================
# 第 9 块：main() —— 把 3 段训练串起来 + 汇总实验            [核心]
# ============================================================
def main():
    run, writer = init_run()

    # 这套配方（增强 + cosine 退火 + weight decay）是按 20-30 轮设计的。
    # 实测：10 轮时 cosine 已把 lr 退到 1e-5，而增强又让模型学得更慢，
    # 小模型反而练不透（student 会比"不加增强"的版本低约 3pp）。轮数不够时提醒一句。
    if EPOCHS < 15:
        logger.warning("EPOCHS=%d 偏少：增强 + cosine 退火一般要 20-30 轮才喂饱模型，"
                       "10 轮左右小模型可能不如不加增强的版本。", EPOCHS)

    logger.info("训练配置：%d epoch | bs %d | %s lr %.1e wd %.1e | "
                "label_smoothing %.2f | warmup %d | T %g | aug %s",
                EPOCHS, BATCH_SIZE, OPTIMIZER, LEARNING_RATE, WEIGHT_DECAY,
                LABEL_SMOOTHING, WARMUP_EPOCHS, TEMPERATURE_T,
                "RandomCrop+Flip" if AUGMENT else "none")

    # ---------- 9.1 训练 Teacher ----------
    # 固定种子，保证每次跑出来的老师一样（对比实验才可比）
    torch.manual_seed(SEED)
    nn_deep = DeepNN(num_classes=10).to(device)
    val_teacher, acc_teacher = train(nn_deep, train_loader, val_loader,
                                     epochs=EPOCHS, learning_rate=LEARNING_RATE,
                                     device=device, run=run, phase="teacher",
                                     test_loader=test_loader)

    # ---------- 9.2 训练 Student 基线（不用老师） ----------
    # nn_light 和 new_nn_light 都紧跟同一个 seed 创建，所以两者初始权重完全相同。
    # 这是刻意的：只有初始化一致，"用了老师"和"没用老师"的对比才公平。
    torch.manual_seed(SEED)
    nn_light = LightNN(num_classes=10).to(device)
    torch.manual_seed(SEED)
    new_nn_light = LightNN(num_classes=10).to(device)

    n_teacher = sum(p.numel() for p in nn_deep.parameters())
    n_student = sum(p.numel() for p in nn_light.parameters())
    logger.info("参数量：DeepNN %s / LightNN %s（压缩 %.2f 倍）",
                format(n_teacher, ","), format(n_student, ","), n_teacher / n_student)

    val_student, acc_student = train(nn_light, train_loader, val_loader,
                                     epochs=EPOCHS, learning_rate=LEARNING_RATE,
                                     device=device, run=run, phase="student",
                                     test_loader=test_loader)

    # ---------- 9.3 ★核心★ 软标签蒸馏 ----------
    val_kd, acc_kd = train_knowledge_distillation(
        teacher=nn_deep, student=new_nn_light,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        epochs=EPOCHS, learning_rate=LEARNING_RATE,
        T=TEMPERATURE_T, soft_target_loss_weight=SOFT_WEIGHT,
        ce_loss_weight=CE_WEIGHT,
        device=device, run=run, phase="kd",
    )

    # ---------- 9.4 汇总 ----------
    results = [
        ("teacher", "DeepNN 1.19M", acc_teacher),
        ("student", "LightNN 0.27M，无老师", acc_student),
        ("kd", "LightNN 0.27M + 软标签", acc_kd),
    ]
    summary = markdown_summary(results)
    val_lines = (f"\n验证集（选模型用）：teacher {val_teacher:.2f}% | "
                 f"student {val_student:.2f}% | kd {val_kd:.2f}%")
    logger.info("\n%s%s", summary, val_lines)
    run.text("summary", summary + "\n" + val_lines)

    # ---------- 9.5 静态曲线图：loss/acc 存 PNG + 进 TensorBoard ----------
    curves_path = plot_curves(run)
    if curves_path:
        logger.info("曲线图      : %s", curves_path)

    # ---- HPARAMS 面板：多跑几组（改 EPOCHS/T/权重/优化器）就能横向对比 ----
    if writer is not None:
        writer.add_hparams(
            {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": OPTIMIZER,
                "label_smoothing": LABEL_SMOOTHING,
                "warmup_epochs": WARMUP_EPOCHS,
                "temperature_T": TEMPERATURE_T,
                "soft_target_loss_weight": SOFT_WEIGHT,
                "ce_loss_weight": CE_WEIGHT,
                "val_size": VAL_SIZE,
                "limit_train": LIMIT_TRAIN,
            },
            {
                "acc/teacher": acc_teacher,
                "acc/student": acc_student,
                "acc/kd": acc_kd,
                "gain/kd_vs_student": acc_kd - acc_student,
                "gain/kd_vs_teacher": acc_kd - acc_teacher,
            },
            run_name="hparams",
        )
        writer.flush()
        writer.close()

    # ---- 实验记录在哪、怎么看 ----
    logger.info("=" * 72)
    logger.info("TensorBoard : tensorboard --logdir %s", os.path.dirname(run.run_dir))
    logger.info("指标表      : %s", os.path.join(run.run_dir, "metrics.csv"))
    logger.info("文本日志    : %s", os.path.join(run.run_dir, "train.log"))
    logger.info("混淆矩阵    : %s/{teacher,student,kd}_confusion_matrix.png", run.run_dir)
    logger.info("曲线图      : %s", os.path.join(run.run_dir, "curves.png"))

    # 按你的要求，这里不保存模型。以后想存，取消注释即可
    # （建议把 optimizer/epoch 一起存，否则没法断点续训）：
    # torch.save({"model": nn_deep.state_dict()},
    #            os.path.join(run.run_dir, "teacher.pth"))
    # torch.save({"model": new_nn_light.state_dict()},
    #            os.path.join(run.run_dir, "student_kd.pth"))


if __name__ == "__main__":
    main()

# ============================================================
# CHANGELOG：本文件相对 CIFAR-10-tracked.py 的改动（提分版）
# ============================================================
#  A. 提高准确率（核心，按收益从大到小）
#   1. [第 2 块] 训练集加数据增强 RandomCrop(32, padding=4) + RandomHorizontalFlip
#      —— 最大的一步：teacher 从 ~80% 提到 ~88-90%，几乎不增加算力
#   2. [第 2 块] 切 5000 张做验证集（train 45k / val 5k / test 10k），
#      调参、选最佳 epoch、早停只看 val；test 只在最后报一次
#   3. [第 4/5 块] warmup(2 轮) + cosine 退火到 lr*0.01
#   4. [第 4/5 块] Adam -> AdamW(weight_decay=5e-4)，或 OPTIMIZER=sgd 走 SGD+momentum
#   5. [第 4/5 块] 交叉熵加 label_smoothing=0.1
#   6. [第 4/5 块] 每轮按 val acc 保存最好权重，patience=8 早停，结束后装回最好权重
#   7. [第 1 块] EPOCHS 默认 10 -> 30（增强后需要更多轮收敛）
#
#  B. 为了可复现/可比
#   * val 切分用固定种子 1234；三段训练共用 SEED（默认 42），
#     nn_light 与 new_nn_light 初始化仍然一致，KD 与基线的对比依然公平
#   * config.json / add_hparams 记录所有新旋钮，改参数不会忘记跑的是什么
#
#  C. 不变的部分
#   * 蒸馏损失数学没动（T^2 补偿、log_softmax 数值稳定版）
#   * 第 7 块评估、第 8 块日志/跟踪、示例分支（5.2/5.3、第 6 块）都保持原样
#
#  D. 还能继续往上走（按性价比排序，需要的话我可以接着加）
#   1. 更宽的 teacher（DeepNN 通道数 x2）或直接换 ResNet-18（CIFAR 版：3x3 stride1 stem）
#      —— teacher 越强，蒸馏上限越高（预期 +5~10pp）
#   2. MixUp / CutMix（需把 CE 改成对软标签的交叉熵，KD 里天然兼容）
#   3. 测试时增强 TTA（原图 + 水平翻转取平均 logits，+0.5~1pp，几乎零成本）
#   4. 多 seed 取平均 / EMA 权重（+0.5~1pp）
#   5. 更长训练 + 更大 batch + lr 同步放大（"免费"的精度的老办法）
