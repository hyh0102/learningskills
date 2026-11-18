
## training-env

记录机器上 `training-env` 深度学习环境的完整配置与复现步骤，用于重建同名 conda 环境。

当前环境定位：

| 项 | 值 |
|---|---|
| conda 版本 | 26.9.4 (`/home/hanyh/miniconda3`) |
| 环境路径 | `/home/hanyh/miniconda3/envs/training-env` |
| Python | 3.15.2 |
| pip | 26.2.2 |
| PyTorch | 2.15.0+cu132 (CUDA 13.2 构建) |
| torchvision | 0.29.2+cu132 |
| 加速库 | cuda-toolkit 13.2.2、nvidia-cudnn-cu13 9.26.0.51、triton 3.9.0 |
| WandB | 0.31.0 |
| TensorBoard | 2.21.0 |
| scikit-learn | 1.9.1（conda-forge 安装；含 numpy 2.5.5、scipy 1.18.1） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，7.6 GB，算力 8.9 (sm_89) |
| 驱动 | NVIDIA 595.99.02 (Open Kernel Module) |

创建一个新环境如下:

```bash
conda create --name training-env python=3.15 -y
```

现在我们可以激活 training-env 环境:

```bash
conda activate training-env
```

## pip 镜像

本机已配置清华源与 60 秒超时（`~/.config/pip/pip.conf`），无 GPU 源需求时可直连:

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
```

## 深度学习框架 PyTorch

安装带 CUDA 13.2 支持的 PyTorch（含 torchvision 与 triton），支持 GPU 训练:

```bash
pip install torch==2.15.0 torchvision==0.29.2
```

> 说明：从 PyPI 默认索引安装会拉到 `+cu132` 变体；若需强制指定 CUDA 版本，追加 `--index-url https://download.pytorch.org/whl/cu132`。

验证 CUDA 是否可用:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出形如 `2.15.0+cu132 13.2 True NVIDIA GeForce RTX 4060 Laptop GPU`。

## 训练记录与可视化组件

```bash
pip install wandb==0.31.0 tensorboard==2.21.0
```

安装说明：`wandb` 曾装在 `base` 环境中，已从 `base` 卸载后统一装到 `training-env`；`tensorboard` 版本与 `tensorboard` 环境保持一致（2.21.0），保证两个环境行为相同。

## 常用工具包

```bash
pip install pillow==13.0.0
```

> `numpy` 现由 conda 提供（2.5.5），不要再单独 pip 安装，否则会与 conda 版本混装。

## scikit-learn

用 conda 安装最新版（`conda-forge` 渠道，与本机 Python 3.15 匹配的 build 只有 `conda-forge` 提供；`defaults` 无 py315 版本）:

```bash
conda install -y -c conda-forge scikit-learn=1.9.1
```

检查版本:

```bash
conda list -n training-env scikit-learn
```

安装前先卸载 pip 版，避免同一包被 pip 与 conda 同时管理:

```bash
pip uninstall -y scikit-learn scipy numpy joblib threadpoolctl cloudpickle narwhals
```

本次安装连带发生的变更:

| 包 | 变更 |
|---|---|
| scikit-learn | 1.9.1 由 pip 改为 conda-forge `np2py315h09497ca_1` |
| numpy | 2.5.4 (pip) → 2.5.5 (conda-forge) |
| scipy | 1.18.1，改由 conda 提供（`pkgs/main`） |
| openssl | 3.5.9 (pkgs/main) → 3.6.5 (conda-forge) |
| 新增 | blas/openblas、libblas/libcblas/liblapack、libgfortran5 等 BLAS 运行库 |

检查安装情况:

```bash
python3 -m pip show scikit-learn  # show scikit-learn version and location
python3 -c "import sklearn; sklearn.show_versions()"
conda list -n training-env | grep -E "scikit-learn|numpy|scipy"
```

## 检查版本

```bash
python --version
pip --version
pip show torch
pip show torchvision
pip show wandb
pip show tensorboard
conda list -n training-env scikit-learn   # scikit-learn 由 conda 管理
```

一次性打印关键版本:

```bash
python -c "
import torch, torchvision, wandb, tensorboard, numpy, sklearn, scipy
print('python      :', __import__('sys').version.split()[0])
print('torch       :', torch.__version__)
print('torchvision :', torchvision.__version__)
print('cuda build  :', torch.version.cuda)
print('cuda avail  :', torch.cuda.is_available())
print('wandb       :', wandb.__version__)
print('tensorboard :', tensorboard.__version__)
print('numpy       :', numpy.__version__)
print('scikit-learn:', sklearn.__version__)
print('scipy       :', scipy.__version__)
"
```

查看全部已安装包:

```bash
pip list --format=freeze    # 干净的 name==version 形式（推荐）
pip list                    # 带版本与位置概览
```

> 注意：环境里现混有 conda 包与 pip 包。对 conda 安装的包，`pip freeze` 会输出 `name @ file:///home/conda/...` 这类本地构建路径，无法直接用于重装；需要清单时请用 `pip list --format=freeze` 或 `conda env export`。

## 导出与复现

优先用 conda 导出（能完整表达 conda 包的 build 号）:

```bash
conda env export -n training-env > training-env.yaml
conda env create -f training-env.yaml
```

仅导出显式请求的包（依赖由解析器重算，跨机器更稳）:

```bash
conda env export -n training-env --from-history
```

补充导出 pip 层依赖（conda 未覆盖的包，如 torch、wandb）:

```bash
pip list --format=freeze | grep -vE "^(scikit-learn|scipy|numpy|joblib|threadpoolctl|cloudpickle|narwhals)==" > requirements-pip.txt
```

## 完整依赖清单

`pip list --format=freeze` 全量输出（2026-11，Python 3.15.2，共 75 个包）:

```text
absl-py==2.5.1
annotated-types==0.8.0
anyio==4.16.0
certifi==2026.10.23
charset-normalizer==3.5.2
click==8.5.1
cloudpickle==3.1.2
cuda-toolkit==13.2.2
cwevals==0.0.1b1
distro==1.9.0
filelock==4.2.4
fsspec==2026.9.0
googleapis-common-protos==1.76.0
grpcio==1.85.1
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.20
Jinja2==3.1.6
joblib==1.6.0
Markdown==3.11.3
MarkupSafe==2.1.5
mpmath==1.3.0
narwhals==2.29.0
networkx==3.7
numpy==2.5.5
nvidia-cublas==13.4.1.3
nvidia-cuda-cccl==13.3.4.3.1
nvidia-cuda-cupti==13.2.86
nvidia-cuda-nvrtc==13.2.86
nvidia-cuda-runtime==13.2.86
nvidia-cudnn-cu13==9.26.0.51
nvidia-cufft==12.2.0.57
nvidia-cufile==1.17.1.22
nvidia-cusolver==12.2.0.11
nvidia-cusparse==12.7.10.12
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.7.2
nvidia-nvtx==13.2.86
opentelemetry-api==1.45.2
opentelemetry-exporter-http-transport==0.66b2
opentelemetry-exporter-otlp-common==0.66b2
opentelemetry-exporter-otlp-proto-common==1.45.2
opentelemetry-exporter-otlp-proto-http==1.45.2
opentelemetry-proto==1.45.2
opentelemetry-sdk==1.45.2
opentelemetry-semantic-conventions==0.66b2
packaging==26.3
pillow==13.0.0
pip==26.2.2
platformdirs==4.14.7
protobuf==7.36.2
pydantic==2.14.3
pydantic_core==2.50.4
PyYAML==6.0.3
requests==2.34.2
scikit-learn==1.9.1
scipy==1.18.1
setuptools==78.1.0
sniffio==1.3.1
sympy==1.14.0
tensorboard==2.21.0
tensorboard-data-server==0.7.2
threadpoolctl==3.8.0
torch==2.15.0+cu132
torchvision==0.29.2+cu132
triton==3.9.0
typing_extensions==4.16.0
typing-inspection==0.4.5
urllib3==2.8.0
wandb==0.31.0
Werkzeug==3.1.9
xxhash==4.1.0
```

## 已知问题

1. `nvidia-smi` 报 `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`，但 `torch.cuda.is_available()` 返回 `True` 且能识别 RTX 4060 Laptop。说明驱动内核模块本身工作正常（`/proc/driver/nvidia/version` 显示 595.99.02），疑似 `nvidia-smi` 用户态工具与内核模块版本不一致（或 595 系列 `nvidia-smi` 的已知问题）。查看显存占用可先用 `torch.cuda.mem_get_info()` 替代；要修 `nvidia-smi`，可尝试重装与内核模块一致的 `nvidia-utils`（需 sudo，未验证）。
2. `base` 环境已不再包含 `wandb`，脚本若依赖 `base` 中的 wandb 会报 `ModuleNotFoundError`，请改用 `training-env`。
3. 环境为 Python 3.15，部分只发布 `cp312`/`cp313` wheel 的包需源码编译，装包时可能失败；遇到时优先找有 `cp315` wheel 的版本。已验证有 `cp315` wheel 的包：`numpy 2.5.5`、`scipy 1.18.1`、`scikit-learn 1.9.1`、`pydantic 2.14.3`、`grpcio 1.85.1`、`pillow 13.0.0`；`pyyaml` 需本地编译（本机已成功构建 `pyyaml-6.0.3-cp315`）。
4. 本环境是 conda 与 pip 混装：`torch`/`torchvision`/`triton`/`wandb`/`tensorboard` 来自 pip，`python`/`numpy`/`scipy`/`scikit-learn`/`openssl` 来自 conda。装 `scikit-learn` 时 conda 把 `openssl` 从 `3.5.9`(pkgs/main) 换成 `3.6.5`(conda-forge)，`numpy` 升到 `2.5.5`；如后续某 pip 包对 numpy ABI 敏感，注意此处基线是 2.5.5 而非 2.5.4。
5. `conda-forge` 与 `defaults` 混用会导致 `openssl` 这类基础库在渠道间来回切换。若要长期稳定，建议执行一次 `conda config --add channels conda-forge` 并设置 `channel_priority strict` 统一渠道（会触发较大范围重解，需谨慎）。
