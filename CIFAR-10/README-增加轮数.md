# 提分版说明：CIFAR-10-improved.py

在原 `CIFAR-10-tracked.py` 基础上改的，**只动"训练策略"，不动蒸馏的数学**。目标是让同一个小模型（LightNN，0.27M）在自己的数据、自己的机器上把准确率顶上去。

## 实测结果（本机 CPU 跑出来的，可比同配置）

**实验 A：全量 5 万张训练图 + 10 轮**（同样的数据、同样的轮数，只差训练策略）

| 阶段 | 原版 | 改进版 | 变化 |
| --- | --- | --- | --- |
| teacher (1.19M) | 73.75% | **81.11%** | **+7.4 pp** |
| student 基线 (0.27M) | 70.03% | 67.23% | -2.8 pp |
| KD (0.27M) | 69.98% | 68.92% | -1.1 pp |

**实验 B：只用 1 万张图 + 30 轮**（数据不够的场景，最能看出一套训练配方稳不稳）

| 阶段 | 原版 | 改进版 | 变化 |
| --- | --- | --- | --- |
| teacher | 60.77%（train 98.2% → 严重过拟合） | **74.99%** | **+14.2 pp** |
| student 基线 | 59.29%（train 97.1%） | **61.89%** | +2.6 pp |
| KD | 58.61%（蒸馏还不如基线） | **64.20%**（比基线 +2.3 pp） | **+5.6 pp** |

两条结论：

1. **数据增强是大头**。1 万张图时原版 train 98% / test 60%，模型纯粹在背题；加了增强之后同一个网络 test 75%。
2. **这套配方要 20-30 轮才喂得饱**。实验 A 里 student 反而掉了 2.8pp，因为 10 轮就把 cosine 退到 lr=1e-5，小模型还没学完 lr 就没了。所以默认 `EPOCHS=30`；轮数 <15 时脚本会打印一条 WARNING 提醒你。

## 一句话结论

原版最大的瓶颈不是模型结构，而是**训练太"素"**：没有数据增强、没有 LR 调度、没有正则、每轮直接拿 test 报数还用它挑模型。补上这四样：

* 全量数据 10 轮：teacher 73.75% → 81.11%（+7.4pp）
* 数据偏少（1 万张）30 轮：teacher 60.77% → 74.99%（+14.2pp），KD 58.61% → 64.20%（+5.6pp）
* 小模型的收益要等轮数够（≥20 轮）才显现，10 轮反而会被 cosine 退火"饿着"

在你这块 GPU 上（teacher 每轮 ~7s），30 轮的完整三段训练约 6 分钟，值得直接跑默认配置看全量 30 轮的结果。

## 改了什么（按收益排序）

| # | 改动 | 在哪 | 预期的效果 |
| --- | --- | --- | --- |
| 1 | 训练集加 `RandomCrop(32,padding=4)` + `RandomHorizontalFlip` | 第 2 块 | **最大的一项**，teacher +8~10pp，几乎零算力 |
| 2 | 切 5000 张作验证集（45k/5k/10k），按 val 选模型 + 早停，test 只报一次 | 第 2、4、5 块 | 数字不再虚高；能选到真正最好的那一轮 |
| 3 | warmup(2 轮) + cosine 退火到 `lr*0.01` | 第 4、5 块 | +1~2pp，后期收敛更细 |
| 4 | Adam → AdamW（`weight_decay=5e-4`），可选 SGD+momentum | 第 4、5 块 | +0.5~1.5pp，抑制过拟合 |
| 5 | CE 加 `label_smoothing=0.1` | 第 4、5 块 | +0.3~0.8pp，和软标签蒸馏更配 |
| 6 | 每轮存 val 最好权重、`patience=8` 早停、最后装回最好权重 | 第 4、5 块 | 不用手工数 epoch |
| 7 | `EPOCHS` 默认 10 → 30 | 第 1 块 | 加了增强后需要更多轮才收敛 |
| 8 | `cudnn.benchmark=True` | 第 0 块 | GPU 提速（不影响精度） |

## 新增的旋钮（全部有默认值，不设也能跑）

```bash
python CIFAR-10-improved.py                  # 默认：30 epoch / bs128 / AdamW 1e-3 / wd 5e-4 / ls 0.1

EPOCHS=10 python CIFAR-10-improved.py        # 少跑点，先看趋势
OPTIMIZER=sgd LR=0.05 python CIFAR-10-improved.py     # SGD+momentum，泛化常略好
LABEL_SMOOTHING=0.0 AUGMENT=0 python CIFAR-10-improved.py   # 消融：关掉两项看掉多少分
VAL_SIZE=2000 PATIENCE=5 python CIFAR-10-improved.py  # 验证集小一点、早停激进一点
T=4 SOFT_WEIGHT=0.3 python CIFAR-10-improved.py       # 换蒸馏温度/权重
SEED=1 python CIFAR-10-improved.py                    # 换种子（多跑几次看方差）
```

每个旋钮的含义写在文件第 1 块的注释里；跑出来的 `config.json` 会把它们全部记下来。

## 两个容易忽略但很关键的点

1. **验证集是"同一批图的两个视图"**。`train_full_aug`（带增强）和 `train_full_eval`（干净）是同一个 CIFAR-10 训练集实例化两次，用同一个 `randperm(seed=1234)` 切分。所以"增强版的第 i 张"和"干净版的第 i 张"严格是同一张图，val 指标才可信。
2. **选模型的"最好"是 val 最好，不是 test 最好**。训练结束后 `model.load_state_dict(best_state)` 把 val 最优那轮的权重重装回去，**再**在 test 上算最终指标——test 全程没参与任何选择，报出来的数才是干净的。

## 怎么验证改动的收益

```bash
# 一次跑三阶段：teacher / student 基线 / KD
EPOCHS=10 python CIFAR-10-improved.py

# 想只跑 KD 对比不同温度，改 T 后重复跑几次，然后在 TensorBoard 的 HPARAMS 面板横向看
T=2 python CIFAR-10-improved.py; T=4 python CIFAR-10-improved.py; T=6 python CIFAR-10-improved.py
```

产物和原版完全一样（`runs/<中文名>/{tfevents, metrics.csv, train.log, config.json, *.png}`），只是 `metrics.csv` 的 `test_*` 列改成了 `val_*`，TensorBoard 的 tag 从 `eval/*` 改成 `val/*`（逐轮）+ `test/*`（最终一次）。

## 还能继续往上走（按性价比排序）

1. **换更强的 teacher**：`DeepNN` 通道数翻倍，或直接上 CIFAR 版 ResNet-18（`conv1` 改 3x3 stride 1、去掉 `maxpool`）。teacher 越强，蒸馏上限越高，预计 +5~10pp。
2. **MixUp / CutMix**：需要把 CE 换成对软标签的交叉熵，KD 里天然兼容（软标签 + MixUp 是绝配）。
3. **TTA**：推理时原图 + 水平翻转各跑一次，logits 取平均，+0.5~1pp，几行代码。
4. **EMA 权重 / 多 seed 平均**：+0.5~1pp，稳定但收益有限。
5. **更长训练 + 更大 batch + 同步放大 lr**：最朴素的提分办法，有卡就上。

## 参考

- PyTorch 官方蒸馏教程：<https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html>
- Hinton et al., *Distilling the Knowledge in a Neural Network*：<https://arxiv.org/abs/1503.02531>
- CIFAR 上的经典训练配方（增强 + cosine + wd）：He et al. *Bag of Tricks for Image Classification*，<https://arxiv.org/abs/1812.01187>
