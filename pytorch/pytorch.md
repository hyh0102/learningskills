# Tensors

 Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

* Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see Bridge with NumPy). Tensors are also optimized for automatic differentiation (we’ll see more about that later in the Autograd section). If you’re familiar with ndarrays, you’ll be right at home with the Tensor API. If not, follow along!

```python
import torch
import numpy as np
```

## Initializing a Tensor

- Tensors can be initialized in various ways. Take a look at the following examples:

### Directly from data

Tensors can be created directly **from data**. The data type is automatically inferred.

```python
>>> data1 = [[1,2],[3,4]]
>>> x1 = torch.tensor(data1)
>>> x1
tensor([[1, 2],
        [3, 4]])
>>> data2 = [1,2,3,4]
>>> x2 = torch.tensor(data2)
>>> x2
tensor([1, 2, 3, 4])
>>> data3 = (1,2,3,4)
>>> x3 = torch.tensor(data3)
>>> x3
tensor([1, 2, 3, 4])
>>> data4 = {1,2,3,4}
>>> x4 = torch.tensor(data4)
Traceback (most recent call last):
  File "<python-input-22>", line 1, in <module>
    x4 = torch.tensor(data4)
RuntimeError: Could not infer dtype of set
```

### From a NumPy array

Tensors can be created from **NumPy arrays** (and vice versa - see Bridge with NumPy).

```python
>>> np_a = np.array(data1)
>>> print(np_a)
[[1 2]
 [3 4]]
>>> x1_np = torch.from_numpy(np_a)
>>> x1_np
tensor([[1, 2],
        [3, 4]])
>>> np_b = np.array(data2)
>>> print(np_b)
[1 2 3 4]
>>> x2_np = torch.from_numpy(np_b)
>>> x2_np
tensor([1, 2, 3, 4])
```

### From another tensor:

The new tensor retains the properties (**shape, datatype**) of the argument tensor, unless explicitly overridden.

```python
x_ones = torch.ones_like(x2) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
```
> 作用：创建一个与 x1 形状相同（2x2），且数据类型相同（整数）的新张量，但里面的元素全部填充为 1。

```python
x_rand = torch.rand_like(x1, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```
> 作用：创建一个与 x1 形状相同的新张量，里面的元素是 0 到 1 之间的随机数。

> 关键点（**dtype=torch.float**）：因为 x1 原本是整数（int），如果直接使用 rand_like，PyTorch 会默认生成整数随机数（全是 0）。为了得到带有小数的随机数，这里通过 dtype=torch.float 强制覆盖了数据类型为浮点数。

Out
```python
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.8574, 0.9516],
        [0.9877, 0.0178]])
```

### With random or constant values:

`shape` is a **tuple** of tensor dimensions. In the functions below, it determines the **dimensionality** of the output tensor.

```python
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

Out
```python
Random Tensor:
 tensor([[0.1677, 0.9952, 0.6849],
        [0.0639, 0.3657, 0.1553]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

## Attributes of a Tensor

Tensor attributes describe their **shape**, **datatype**, and the device on which they are **stored**.

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

Out

```python
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

## Operations on Tensors

Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling and more are comprehensively described.

