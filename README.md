<div align="center">

# 🧠 DeepLearningForClassification

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-13.2-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-FashionMNIST-FF6F00?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p>基于 PyTorch 的深度学习图像分类实战系列</p>
<p>以 <b>FashionMNIST</b> 为主线，循序渐进演示从浅层网络到自归一化深层网络的完整实践路径</p>
<p>每个 Notebook 均配有详细中文注释，适合深度学习入门与进阶学习</p>

</div>

---

## 📋 目录

- [项目概览](#-项目概览)
- [目录结构](#-目录结构)
- [Notebook 介绍](#-notebook-介绍)
- [数据集](#-数据集)
- [学习路径](#-学习路径)
- [环境依赖](#-环境依赖)
- [快速开始](#-快速开始)

---

## 🗺 项目概览

| # | Notebook | 核心技术 | 亮点 |
|---|----------|----------|------|
| 1 | 浅层全连接网络 | `nn.Sequential` · SGD | 完整分类流程入门 |
| 2 | 训练精细化控制 | 回调机制 · 早停 · TensorBoard | 工程化训练范式 |
| 3 | 深层网络 + BN | `BatchNorm1d` · ReLU · He 初始化 | 梯度稳定 · 加速收敛 |
| 4 | 深层网络 + SELU | `SELU` · `AlphaDropout` · LeCun 初始化 | 自归一化 · 无需 BN |

---

## 📁 目录结构

```
DeepLearningForClassification/
├── 📓 1.image_classification_s(shallow)nn.ipynb
├── 📓 2.image_classification_s(shallow)nn_train_more_control.ipynb
├── 📓 3.image_classification_dnn_bn.ipynb
├── 📓 4.image_classification_dnn_selu_AlphaDropout.ipynb
│
├── 📂 data/
│   └── FashionMNIST/              # 自动下载的数据集缓存
│
├── 📂 model_checkpoints/          # 各模型最佳检查点
│   ├── snn/
│   ├── relu+he+bn/
│   └── selu+lecun+AlphaDropout/
│
├── 📂 tensorboard_logs/           # TensorBoard 训练日志
│   ├── snn/
│   ├── relu+he+bn/
│   └── selu+lecun+AlphaDropout/
│
├── 📄 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```

---

## 📚 Notebook 介绍

### 1 · 浅层全连接网络（Shallow Neural Network）

> `1.image_classification_s(shallow)nn.ipynb`

从零搭建基础全连接神经网络，系统演示 FashionMNIST 图像分类的**完整端到端流程**。

| 章节 | 内容 |
|------|------|
| 环境配置与库导入 | PyTorch、torchvision 等依赖导入 |
| 数据可视化 | 灰度图展示，理解数据分布 |
| 数据预处理 | 标准化 + DataLoader 构建 |
| 定义模型 | 全连接神经网络（`nn.Sequential`） |
| 训练循环 | 损失函数 + 优化器 + 准确率评估 |
| 学习曲线 | 训练 / 验证 loss 与 accuracy 可视化 |
| 最终评估 | 测试集性能报告 |

---

### 2 · 训练过程精细化控制

> `2.image_classification_s(shallow)nn_train_more_control.ipynb`

在基础篇之上，专注于**工程化训练控制机制**，不重复基础内容。

| 对比维度 | 基础篇（1） | 本篇（2） |
|----------|:-----------:|:---------:|
| 数据可视化 | ✅ | — |
| 训练循环 | 手写基础版 | 封装为带回调的函数 |
| 模型保存 | — | ✅ 自动保存最佳检查点 |
| 早停机制 | — | ✅ `EarlyStopCallback` |
| 训练可视化 | 事后绘制学习曲线 | ✅ TensorBoard 实时监控 |

```shell
# 启动 TensorBoard
tensorboard --logdir=tensorboard_logs/snn
```

---

### 3 · 深层网络 + 批归一化（Batch Normalization）

> `3.image_classification_dnn_bn.ipynb`

在 DNN 基础上引入 **批归一化（BN）** 技术，集成回调机制与 TensorBoard 监控。

> **BN 原理** · 对每个 mini-batch 的激活值进行归一化，缓解梯度消失 / 爆炸，加速收敛，兼具正则化效果。

| 配置项 | 详情 |
|--------|------|
| 激活函数 | ReLU |
| 权重初始化 | He 初始化（`kaiming_normal_`） |
| 检查点路径 | `model_checkpoints/relu+he+bn/` |
| 日志路径 | `tensorboard_logs/relu+he+bn/` |
| 参考论文 | *Batch Normalization*（Ioffe & Szegedy, 2015） |

```shell
# 启动 TensorBoard
tensorboard --logdir=tensorboard_logs/relu+he+bn
```

---

### 4 · 深层网络 + SELU + AlphaDropout

> `4.image_classification_dnn_selu_AlphaDropout.ipynb`

在 DNN 基础上引入 **SELU 激活函数** 与 **AlphaDropout**，构建自归一化神经网络（SNN）。

> **SELU 原理** · 具有自归一化特性，各层激活值自动保持均值 ≈ 0、方差 ≈ 1，从根本上解决梯度消失 / 爆炸，**无需 BN**。
>
> **AlphaDropout 原理** · 置零时执行线性变换（缩放 + 平移），使输出仍保持自归一化特性，与 SELU 完美兼容；`model.eval()` 后自动关闭。

| 配置项 | 详情 |
|--------|------|
| 激活函数 | SELU |
| 权重初始化 | LeCun 初始化（`lecun_normal_`） |
| 正则化 | `AlphaDropout` |
| 检查点路径 | `model_checkpoints/selu+lecun+AlphaDropout/` |
| 日志路径 | `tensorboard_logs/selu+lecun+AlphaDropout/` |
| 参考论文 | *Self-Normalizing Neural Networks*（Klambauer et al., 2017） |

```shell
# 启动 TensorBoard
tensorboard --logdir=tensorboard_logs/selu+lecun+AlphaDropout
```

---

## 🗃 数据集

**FashionMNIST** · 由 Zalando 发布的服装图像数据集

| 属性 | 详情 |
|------|------|
| 类别数 | 10（T-shirt · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot） |
| 训练集 | 60,000 张 |
| 测试集 | 10,000 张 |
| 图像尺寸 | 28 × 28 灰度图 |

数据集通过 `torchvision.datasets.FashionMNIST` 首次运行时**自动下载**至 `data/FashionMNIST/`，无需手动准备。

---

## 🛤 学习路径

```
Notebook 1          Notebook 2          Notebook 3          Notebook 4
浅层网络基础    →   训练工程化      →   DNN + BN        →   DNN + SELU
                                                            + AlphaDropout
```

每个 Notebook 均为前一篇的进阶扩展，建议按编号顺序学习。

---

## ⚙️ 环境依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.12.0+cu132 | 深度学习框架核心（CUDA 13.2） |
| `torchvision` | 0.27.0+cu132 | 数据集加载与图像变换 |
| `numpy` | 2.4.6 | 数值计算 |
| `matplotlib` | 3.10.9 | 数据可视化 |
| `pandas` | 3.0.3 | 训练记录管理 |
| `scikit-learn` | 1.9.0 | 评估指标计算 |
| `Pillow` | 12.2.0 | 图像读取与处理 |
| `tqdm` | 4.68.2 | 训练进度条显示 |
| `tensorboard` | 2.20.0 | 训练过程实时可视化（02 / 03 / 04） |

---

## 🚀 快速开始

**1. 克隆仓库**

```bash
git clone https://github.com/your-username/DeepLearningForClassification.git
cd DeepLearningForClassification
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

> ⚠️ `torch` 与 `torchvision` 的 CUDA 版本需与本机驱动匹配。如需 CPU 版本或其他 CUDA 版本，请参考 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 替换。

**3. 启动 Jupyter**

```bash
jupyter notebook
```

按编号顺序打开 Notebook 开始学习即可。

---

## 📄 License

[MIT](LICENSE) © 2026
