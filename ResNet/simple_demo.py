#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小可跑 demo：BasicBlock / Bottleneck / DeformConv（纯 CPU 即可）。

只依赖 torch 和本目录三个模块，不需要数据集、不需要 torchvision、不需要 CUDA：
    python simple_demo.py

依次演示：
  1. BasicBlock 单独使用（ResNet-18/34 的积木）；
  2. Bottleneck 单独使用（ResNet-50+ 的积木）；
  3. DeformConv2d 单独使用（普通卷积 + 可学习 offset），并验证零 offset 等价普通卷积；
  4. DeformConv2dPack 嵌入 ResNet（DeformResNet-18）端到端前向。
"""

import torch
import torch.nn as nn

from deform_conv import DeformConv2d, DeformConv2dPack
from deform_resnet import deform_resnet18
from resnet import BasicBlock, Bottleneck, conv1x1

torch.manual_seed(0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | device={device}")

    # ------------------------------------------------------------------ #
    # 1) BasicBlock：输入输出同通道数，identity shortcut
    # ------------------------------------------------------------------ #
    x = torch.randn(2, 64, 32, 32, device=device, requires_grad=True)
    block = BasicBlock(64, 64).to(device)
    y = block(x)
    assert y.shape == (2, 64, 32, 32)
    y.square().mean().backward()
    print(f"[1] BasicBlock      {tuple(x.shape)} -> {tuple(y.shape)}, "
          f"params={sum(p.numel() for p in block.parameters()):,}, "
          f"|grad x|={x.grad.norm():.3f}")

    # stride=2 + 1x1 投影 shortcut：空间减半、通道翻倍
    # 改变通道数/空间尺寸时，shortcut 需要显式 1x1 投影（ResNet._make_layer 会自动建）
    downsample = nn.Sequential(conv1x1(64, 128, stride=2), nn.BatchNorm2d(128))
    block_s2 = BasicBlock(64, 128, stride=2, downsample=downsample).to(device)
    y2 = block_s2(x.detach())
    assert y2.shape == (2, 128, 16, 16)
    print(f"[1] BasicBlock(s2)  {tuple(x.shape)} -> {tuple(y2.shape)}, "
          f"params={sum(p.numel() for p in block_s2.parameters()):,}")

    # ------------------------------------------------------------------ #
    # 2) Bottleneck：1x1 降维 -> 3x3 -> 1x1 升维，expansion=4
    # ------------------------------------------------------------------ #
    xb = torch.randn(2, 64, 32, 32, device=device)
    downsample_b = nn.Sequential(conv1x1(64, 256, stride=2), nn.BatchNorm2d(256))
    neck = Bottleneck(64, 64, stride=2, downsample=downsample_b).to(device)  # 64 -> 256
    yb = neck(xb)
    assert yb.shape == (2, 256, 16, 16)
    print(f"[2] Bottleneck      {tuple(xb.shape)} -> {tuple(yb.shape)}, "
          f"expansion={neck.expansion}, "
          f"params={sum(p.numel() for p in neck.parameters()):,}")

    # ------------------------------------------------------------------ #
    # 3) DeformConv2d：offset 是显式输入，形状 (N, 2*kH*kW, H_out, W_out)
    #    offset 全 0 时严格退化为普通卷积（可替换任意 3x3 conv）
    # ------------------------------------------------------------------ #
    xd = torch.randn(2, 8, 16, 16, device=device)
    offset = torch.zeros(2, 2 * 3 * 3, 16, 16, device=device)  # (dy,dx) 逐点交错
    dcn = DeformConv2d(8, 8, kernel_size=3, padding=1).to(device)
    conv = nn.Conv2d(8, 8, kernel_size=3, padding=1).to(device)
    conv.weight.data.copy_(dcn.weight.data)
    conv.bias.data.copy_(dcn.bias.data)

    yd = dcn(xd, offset)
    yc = conv(xd)
    diff = (yd - yc).abs().max().item()
    assert diff < 1e-5, diff
    print(f"[3] DeformConv2d    {tuple(xd.shape)} -> {tuple(yd.shape)}, "
          f"零 offset 与 nn.Conv2d 最大误差={diff:.2e}")

    # 让 offset 参与反向传播（模拟 offset 由网络另一分支预测）
    offset.requires_grad_(True)
    dcn(xd, offset).square().mean().backward()
    print(f"[3] offset 可学习   |grad offset|={offset.grad.norm():.3f}")

    # ------------------------------------------------------------------ #
    # 4) DeformConv2dPack：内部自己预测 offset，直接当卷积层用
    #    初始 offset 分支为 0 -> 与同权重普通卷积等价
    # ------------------------------------------------------------------ #
    pack = DeformConv2dPack(8, 8, kernel_size=3, padding=1).to(device)
    conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1).to(device)
    conv2.weight.data.copy_(pack.conv.weight.data)
    conv2.bias.data.copy_(pack.conv.bias.data)
    pack_diff = (pack(xd) - conv2(xd)).abs().max().item()
    assert pack_diff < 1e-5, pack_diff
    print(f"[4] Pack 初始等价普通卷积，最大误差={pack_diff:.2e}")

    # 嵌入 ResNet：BasicBlock 里的 3x3 换成 DCN，CIFAR stem 直接吃 32x32
    model = deform_resnet18(num_classes=10, stem="cifar").to(device)
    img = torch.randn(2, 3, 32, 32, device=device)
    logits = model(img)
    assert logits.shape == (2, 10)
    print(f"[4] DeformResNet-18 {tuple(img.shape)} -> {tuple(logits.shape)}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    print("\n全部通过：三个组件都能独立实例化、前向、反向，且 DeformConv 在 CPU 上可运行。")


if __name__ == "__main__":
    main()
