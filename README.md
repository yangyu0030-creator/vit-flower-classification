# 项目介绍

## 1. 下载数据集

代码中默认使用的是花分类数据集，下载地址：

https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

下载后解压到 `data/` 目录。

## 2. 切分数据集

运行 `split_data.py`，将数据集切分为训练集和验证集：

```bash
python split_data.py
```

切分后目录结构如下：

```
data/
  flower_photos/     # 原始数据
  train/             # 训练集
    daisy/
    dandelion/
    roses/
    sunflowers/
    tulips/
  val/               # 验证集
    daisy/
    dandelion/
    roses/
    sunflowers/
    tulips/
```

## 3. 下载预训练权重

本项目使用 ImageNet-21k 预训练的 ViT-Base/16：

https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth

下载后放到 `weights/` 目录下。

## 4. 配置训练参数

在 `train.py` 的 `__main__` 中设置好数据路径和预训练权重路径：

```python
data_path    = "./data"
weights_path = "weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
num_classes  = 5
num_epochs   = 10
lr           = 1e-4
batch_size   = 8
```

## 5. 预测

在 `predict.py` 中导入和训练脚本相同的模型结构，并将 `weights_path` 设置为训练好的模型权重路径，`img_path` 设置成自己需要预测的图片绝对路径：

```python
weights_path = "weights/model.pth"
img_path     = "./test_images/daisy.jpg"
```

然后运行：

```bash
python predict.py
```