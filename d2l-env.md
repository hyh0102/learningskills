
## d2l

创建一个新环境如下:

```bash
conda create --name d2l python=3.9 -y
```

现在我们可以激活d2l环境:

```bash
conda activate d2l
```

安装深度学习框架 d2l 包装

可以安装 PyTorch(指定版本会在时间上进行测试书面支持)支持CPU或GPU如下:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```

下一步是安装d2l我们开发的软件包命令封装常用函数和类本书全文:

```bash
pip install d2l==1.0.3
```

检查版本

```bash
pip show torch
pip show d2l
```


## scikit-learn

检查版本

```bash
python3 --version
pip3 --version
```
安装(pip)

```bash
pip3 install -U scikit-learn
```

检查安装情况

```bash
python3 -m pip show scikit-learn  # show scikit-learn version and location
python3 -m pip freeze             # show all installed packages in the environment
python3 -c "import sklearn; sklearn.show_versions()"
```

## tensorboard

```bash
conda install -y -c conda-forge tensorboard
```
