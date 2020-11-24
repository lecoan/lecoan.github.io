# Pytorch 模型量化上手

## 前因

论文中看到树莓派的VGG16推理速度应该在2s左右，但实际使用起来(pytorch arm)大概在1min上下，很不可思议。推测应该是树莓派的浮点运算能力较弱，于是尝试对模型进行量化，之后推理速度降到了1.7s上下。

## Background

### 支持平台类型

> Today, PyTorch supports the following backends for running quantized operators efficiently:
>
> - x86 CPUs with AVX2 support or higher (without AVX2 some operations have inefficient implementations) - **fbgemm**
> - ARM CPUs (typically found in mobile/embedded devices) - **qnnpack**

### 量化方式

> 1. **Post Training Dynamic Quantization**：这是最简单的一种量化方法，Post Training指的是在浮点模型训练收敛之后进行量化操作，其中weight被提前量化，而activation在前向推理过程中被动态量化，即每次都要根据实际运算的浮点数据范围每一层计算一次scale和zero_point，然后进行量化；
> 2. **Post Training Static Quantization**：第一种不是很常见，一般说的Post Training Quantization指的其实是这种静态的方法，而且这种方法是最常用的，其中weight跟上述一样也是被提前量化好的，然后activation也会基于之前校准过程中记录下的固定的scale和zero_point进行量化，整个过程不存在量化参数*(*scale和zero_point)的再计算；
> 3. **Quantization Aware Training**：对于一些模型在浮点训练+量化过程中精度损失比较严重的情况，就需要进行量化感知训练，即在训练过程中模拟量化过程，数据虽然都是表示为float32，但实际的值的间隔却会受到量化参数的限制。
>

## 上手

因为我只想测试推理时间，不关注精度，所以采用第二种方法

### 设置量化 backend

Q: 这里的qconfig可以应该自定义，但是还不知道怎么弄

```python
torch.backends.quantized.engine = 'qnnpack'  # arm cpu
qconfig = torch.quantization.get_default_qconfig('qnnpack')  # 获取默认的量化配置
```

```python
QConfig(
  activation=functools.partial(
    <class 'torch.quantization.observer.HistogramObserver'>, reduce_range=False), 
  weight=functools.partial(
    <class 'torch.quantization.observer.MinMaxObserver'>, 
    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
)
```



### 设置模型

```python
import torch
from torchvision import models

model = models.vgg16()
model = torch.nn.Sequential(
  torch.quantization.QuantStub(),   # 量化前不起作用(Observer)，之后负责将fp32数据量化
  model, 
  torch.quantization.DeQuantStub()) # 量化前不起作用(Observer)，之后负责将量化tensor转为fp32
model.eval()                        # 必须设置
model.qconfig = qconfig             # 设置模型的量化配置
```

### 开始量化

之后调用 prepare在模型后部插入一个`HistogramObserver`，用来观测推理时的tensor数据变化，决定量化后的tensor的scale和zero_point

```python
model_prepared = torch.quantization.prepare(model) 
```

`HistogramObserver` 在VGG中的位置如下

```text
  (6): Linear(
  in_features=4096, out_features=1000, bias=True
  (activation_post_process): HistogramObserver()
  )
```

做一次推理，使`HistogramObserver`得到优化的目标，再进行量化

```python
fpdata = torch.randn(1,3,244,244)
model_prepared(fpdata)
model_int8 = torch.quantization.convert(model_prepared)
```

## Additional Material

- `QuantStub` & `QuantStub` 不是必须的，如果不加可以手动对tensor进行量化和解量化

- tensor的量化与解量化

  ```python
  torch.quantize_per_tensor(
    fpdata, scale, zero_point, dtype=torch.quint8)  # 量化
  qdata.dequantize()
```
  

## References

> [Pytorch quantization]: https://pytorch.org/docs/stable/quantization.html/
> [PyTorch模型量化工具学习]: https://zhuanlan.zhihu.com/p/144025236/
>
