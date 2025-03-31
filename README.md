# 基于SE-ResNet的CIFAR10/100图像分类

本项目实现了基于ResNet18/34架构的改进神经网络，通过引入Squeeze-and-Excitation注意力机制提升模型在CIFAR10和CIFAR100数据集上的分类性能。

## 目录

- [项目概述](#项目概述)
- [安装与环境](#安装与环境)
- [模型架构](#模型架构)
- [数据集说明](#数据集说明)
- [使用指南](#使用指南)
  - [训练模型](#训练模型)
  - [断点续训](#断点续训)
  - [最终测试与评估](#最终测试与评估)
  - [推理应用](#推理应用)
- [性能优化](#性能优化)
- [示例命令](#示例命令)
- [消融实验](#消融实验)
- [常见问题](#常见问题)

## 项目概述

### 功能特点

- 基于ResNet18/34架构，集成Squeeze-and-Excitation注意力机制
- 支持CIFAR10和CIFAR100数据集的图像分类任务
- 实现了高效的训练、评估和推理流程
- 提供了多种正则化和优化策略，提高模型泛化能力
- 完整的训练过程可视化和模型评估报告

### 项目结构

```
cifar_resnet_project/
├── models/
│   └── se_resnet.py     # SE-ResNet模型实现
├── train.py             # 训练代码
├── inference.py         # 推理代码
├── test_model.py        # 模型测试与分析代码
├── checkpoints/         # 模型检查点保存目录
└── README.md            # 项目说明文档
```

## 安装与环境

### 依赖要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy
- scikit-learn
- seaborn

### PyTorch兼容性说明

- **PyTorch 2.6+ 注意事项**：
  1. 自 PyTorch 2.6 起，`torch.load` 的 `weights_only` 参数默认值从 `False` 改为 `True`
  2. 在 PyTorch 2.6+ 中，可能出现 `torch` 变量作用域问题
  3. 代码已添加多层错误处理机制，应能自动处理大多数兼容性问题

## 模型架构

本项目基于ResNet18/34架构，引入了Squeeze-and-Excitation (SE) 注意力机制进行改进。SE模块通过显式建模通道间的相互依赖关系，自适应地重新校准通道特征响应，从而提升模型性能。

### SE模块工作原理

1. **Squeeze操作**：通过全局平均池化将每个通道的空间信息压缩为单个数值，生成通道描述符
2. **Excitation操作**：使用两个全连接层和非线性激活函数，学习通道间的非线性关系，生成每个通道的重要性权重
3. **重新校准**：将学习到的权重应用于原始特征图，增强重要通道的特征，抑制不重要通道的特征

### 网络结构优化

- 针对CIFAR10/100数据集的小尺寸图像(32x32)，调整了网络初始层的卷积核大小和步长
- 实现了可配置的SE模块，可通过参数控制是否使用SE模块以及缩减率
- 使用Cosine学习率调度策略，帮助模型更好地收敛
- 添加Dropout层缓解过拟合

## 数据集说明

### 数据集划分

项目采用标准的数据集划分方法：

1. **训练集 (Training Set)**：CIFAR10/100 官方训练集的 90% (默认)，用于模型训练
2. **验证集 (Validation Set)**：CIFAR10/100 官方训练集的 10% (默认)，用于超参数调优和早停
3. **测试集 (Test Set)**：CIFAR10/100 官方测试集，仅用于最终评估

训练过程使用验证集而不是测试集来监控模型性能，测试集仅在完成所有训练和超参数调整后用于最终评估。这种方法符合机器学习的最佳实践，避免信息泄露和过拟合测试集。

### 数据增强策略

- **训练集**：应用随机裁剪、水平翻转和颜色抖动等数据增强技术
- **验证集和测试集**：仅应用标准化变换，不使用数据增强

## 使用指南

### 训练模型

训练模型的基本命令：

```bash
python train.py --dataset cifar10 --model se_resnet18 --batch_size 128 --epochs 100 --save_dir checkpoints/cifar10_resnet18
```

#### 主要参数说明

- `--dataset`: 数据集选择，可选 'cifar10' 或 'cifar100'
- `--model`: 模型选择，可选 'se_resnet18' 或 'se_resnet34'
- `--use_se`: 是否使用SE模块，默认为True
- `--se_reduction`: SE模块的缩减率，默认为16
- `--dropout_rate`: Dropout丢弃率，默认为0.5
- `--batch_size`: 批量大小，默认为50
- `--lr`: 初始学习率，默认为0.001
- `--weight_decay`: 权重衰减系数，默认为1e-4
- `--epochs`: 训练轮数，默认为100
- `--val_ratio`: 用于验证的训练数据比例，默认为0.1
- `--no_augmentation`: 禁用数据增强，仅使用基础变换（ToTensor和Normalize）
- `--save_dir`: 检查点保存目录，默认为'checkpoints'
- `--resume`: 从检查点恢复训练，默认为空
- `--final_test`: 训练完成后在测试集上进行最终评估

### 断点续训

如果训练过程中断，可以使用以下命令从检查点恢复训练：

```bash
# 从最新检查点恢复训练
python train.py --dataset cifar10 --model se_resnet18 --resume checkpoints/cifar10_resnet18/cifar10_se_resnet18_latest.pth.tar

# 从特定epoch的检查点恢复训练
python train.py --dataset cifar10 --model se_resnet18 --resume checkpoints/cifar10_resnet18/cifar10_se_resnet18_epoch_50.pth.tar

# 恢复CIFAR-100训练
python train.py --dataset cifar100 --model se_resnet18 --resume checkpoints/cifar100_se_resnet18_latest.pth.tar
```

### 最终测试与评估

在完成所有训练和调参后，应当使用官方测试集进行最终一次性评估。这可以通过两种方式进行：

#### 1. 训练结束后自动测试

使用 `--final_test` 参数在训练结束后自动进行测试：

```bash
# 训练并在结束后进行最终测试
python train.py --dataset cifar10 --model se_resnet18 --epochs 100 --final_test --save_dir checkpoints/cifar10_resnet18
```

#### 2. 使用专门的测试工具

对已训练的模型进行详细测试和分析：

```bash
# 对最佳模型进行详细测试
python test_model.py --dataset cifar10 --model se_resnet18 --checkpoint checkpoints/cifar10_resnet18/cifar10_se_resnet18_best.pth.tar --save_dir results/cifar10_resnet18

# 对CIFAR-100模型进行测试
python test_model.py --dataset cifar100 --model se_resnet18 --checkpoint checkpoints/cifar10_resnet18/cifar100_se_resnet18_best.pth.tar
```

`test_model.py` 工具会生成详细的测试报告，包括：

- 混淆矩阵可视化：显示类别间的混淆情况
- 类别准确率分析：显示每个类别的准确率，识别表现最好和最差的类别
- 详细的分类报告：包含精确率、召回率和F1分数
- 整体模型性能摘要：包括整体准确率和测试结果概要

测试结果会保存在 `results` 目录中，作为模型性能的最终评估依据。

### 推理应用

模型训练完成后，可以使用 `inference.py` 脚本进行推理。与测试不同，推理主要用于模型的应用和可视化预测结果：

```bash
# 单张图像推理
python inference.py --dataset cifar10 --model se_resnet18 --checkpoint checkpoints/cifar10_resnet18/cifar10_se_resnet18_best.pth.tar --image path/to/image.jpg

# 批量推理整个文件夹内的图像
python inference.py --dataset cifar10 --model se_resnet18 --checkpoint checkpoints/cifar10_resnet18/cifar10_se_resnet18_best.pth.tar --image_dir path/to/image/folder
```

> **注意**：推理和测试的区别
> 
> - **测试 (test_model.py)**：在完整官方测试集上评估模型性能，生成详细分析报告，用于最终评估模型质量
> - **推理 (inference.py)**：对新的、未见过的图像进行预测并可视化结果，用于实际应用和展示

## 性能优化

为了在保证模型性能的同时适应计算资源限制（RTX 3060 Laptop 6G），本项目采取了以下优化措施：

### 模型优化

1. **正则化技术**：
   - **权重衰减 (Weight Decay)**：使用 L2 正则化 (权重衰减=1e-4) 防止权重过大，缓解过拟合
   - **Dropout**：在全局平均池化层后添加 Dropout (默认比率=0.5)，随机丢弃一部分神经元，提高模型泛化能力

2. **SE模块的缩减率**：通过调整SE模块的缩减率参数，可以平衡模型性能和计算复杂度

### 训练优化

1. **数据增强**：
   - 基础数据增强：`RandomCrop` 和 `RandomHorizontalFlip`
   - 高级色彩增强：`ColorJitter` 随机调整亮度、对比度、饱和度和色调，增加数据多样性

2. **验证集划分**：
   - 从训练数据中划分出验证集用于模型选择和超参数调优
   - 保留官方测试集仅用于最终评估，确保评估的无偏性

3. **批量大小调整**：根据GPU内存限制调整批量大小

4. **学习率策略**：
   - 使用 AdamW 优化器替代传统的 Adam，提供更有效的权重衰减实现
   - 可调节的权重衰减参数 (默认值 1e-4)，更灵活地控制正则化强度
   - 使用适合 AdamW 优化器的初始学习率 (0.001)
   - 使用 Cosine 学习率调度策略，帮助模型在较少的训练轮数内达到较好的性能

5. **检查点保存策略**：
   - 每个 epoch 保存最新模型状态
   - 每 10 个 epoch 保存一次中间检查点
   - 始终保存验证集上表现最佳的模型
   - 检查点文件名包含数据集和模型类型，便于区分

## 示例命令

根据不同需求，以下是一些推荐的运行命令：

### 基础训练示例

```bash
# CIFAR-10 基础训练 (SE-ResNet18)
python train.py --dataset cifar10 --model se_resnet18 --batch_size 128 --epochs 100 --val_ratio 0.1 --save_dir checkpoints/cifar10_resnet18

# CIFAR-100 基础训练 (SE-ResNet18)
python train.py --dataset cifar100 --model se_resnet18 --batch_size 128 --epochs 100 --val_ratio 0.1 --save_dir checkpoints/cifar100_resnet18

# 使用 SE-ResNet34 并在训练完成后进行测试
python train.py --dataset cifar10 --model se_resnet34 --batch_size 64 --epochs 100 --final_test --save_dir checkpoints/cifar10_resnet34
```

### 处理过拟合问题示例

对于严重过拟合的情况，可以尝试以下设置：

```bash
# 增加权重衰减
python train.py --dataset cifar10 --model se_resnet18 --weight_decay 5e-4 --val_ratio 0.1 --save_dir checkpoints/cifar10_wd5e4

# 增加 Dropout 率和验证集比例
python train.py --dataset cifar10 --model se_resnet18 --dropout_rate 0.7 --val_ratio 0.2 --save_dir checkpoints/cifar10_dropout7

# 综合优化 (适用于严重过拟合)
python train.py --dataset cifar10 --model se_resnet18 --dropout_rate 0.6 --weight_decay 5e-4 --se_reduction 32 --val_ratio 0.2 --save_dir checkpoints/cifar10_anti_overfit
```

### CIFAR-100 优化示例

由于 CIFAR-100 具有100个类别，过拟合问题通常更为严重：

```bash
# CIFAR-100 优化训练
python train.py --dataset cifar100 --model se_resnet18 --dropout_rate 0.6 --weight_decay 5e-4 --epochs 150 --val_ratio 0.15 --final_test --save_dir checkpoints/cifar100_optimized
```

## 消融实验

消融实验（Ablation Study）是通过移除或禁用模型的某些组件来验证这些组件对模型性能的贡献。通过对比完整模型与移除特定组件后的模型性能差异，可以量化各个组件的重要性，从而评估当前网络架构设计的有效性。

### 实验类型

本项目支持以下几种消融实验：

1. **SE模块消融实验**：禁用SE注意力机制，验证其对分类性能的贡献
2. **Dropout消融实验**：移除Dropout层，评估其防止过拟合的效果
3. **数据增强消融实验**：禁用高级数据增强，验证数据增强对模型泛化能力的影响
4. **SE缩减率实验**：调整SE模块的缩减率，测试不同配置对模型性能的影响

### 运行消融实验

#### SE模块消融实验

通过设置 `--use_se False` 参数禁用SE模块：

```bash
# 基准模型（带SE模块）
python train.py --dataset cifar10 --model se_resnet18 --batch_size 128 --epochs 100 --save_dir checkpoints/cifar10_with_se

# 消融模型（无SE模块）
python train.py --dataset cifar10 --model se_resnet18 --use_se False --batch_size 128 --epochs 100 --save_dir checkpoints/cifar10_without_se

# 在CIFAR-100上进行相同实验
python train.py --dataset cifar100 --model se_resnet18 --use_se False --batch_size 128 --epochs 100 --save_dir checkpoints/cifar100_without_se
```

#### Dropout消融实验

通过设置 `--dropout_rate 0.0` 禁用Dropout层：

```bash
# 基准模型（Dropout率 0.5）
python train.py --dataset cifar10 --model se_resnet18 --dropout_rate 0.5 --epochs 100 --save_dir checkpoints/cifar10_dropout_0.5

# 消融模型（无Dropout）
python train.py --dataset cifar10 --model se_resnet18 --dropout_rate 0.0 --epochs 100 --save_dir checkpoints/cifar10_no_dropout

# 在CIFAR-100上进行相同实验
python train.py --dataset cifar100 --model se_resnet18 --dropout_rate 0.0 --epochs 100 --save_dir checkpoints/cifar100_no_dropout
```

#### 数据增强消融实验

通过设置 `--no_augmentation` 参数禁用高级数据增强：

```bash
# 基准模型（完整数据增强）
python train.py --dataset cifar10 --model se_resnet18 --epochs 100 --save_dir checkpoints/cifar10_full_aug

# 消融模型（仅基础数据增强）
python train.py --dataset cifar10 --model se_resnet18 --no_augmentation --epochs 100 --save_dir checkpoints/cifar10_basic_aug

# 在CIFAR-100上进行相同实验
python train.py --dataset cifar100 --model se_resnet18 --no_augmentation --epochs 100 --save_dir checkpoints/cifar100_basic_aug
```

#### SE缩减率实验

测试不同SE模块缩减率对模型性能的影响：

```bash
# 缩减率 8（更多参数，可能更强的表达能力）
python train.py --dataset cifar10 --model se_resnet18 --se_reduction 8 --epochs 100 --save_dir checkpoints/cifar10_se_r8

# 缩减率 16（默认值）
python train.py --dataset cifar10 --model se_resnet18 --se_reduction 16 --epochs 100 --save_dir checkpoints/cifar10_se_r16

# 缩减率 32（更少参数，可能更低的过拟合风险）
python train.py --dataset cifar10 --model se_resnet18 --se_reduction 32 --epochs 100 --save_dir checkpoints/cifar10_se_r32
```

### 结果比较分析

完成消融实验后，可以使用以下命令对不同模型进行最终测试并比较结果：

```bash
# 测试带SE模块的模型
python test_model.py --dataset cifar10 --model se_resnet18 --checkpoint checkpoints/cifar10_with_se/best.pth.tar

# 测试无SE模块的模型
python test_model.py --dataset cifar10 --model se_resnet18 --use_se False --checkpoint checkpoints/cifar10_without_se/best.pth.tar
```

### 结果分析和可视化

使用以下命令可以生成消融实验结果的比较图表：

```bash
# 比较不同模型的准确率和损失
python compare_models.py --checkpoints checkpoints/cifar10_with_se/results.json checkpoints/cifar10_without_se/results.json --labels "With SE" "Without SE" --metric accuracy

# 比较不同缩减率配置的性能
python compare_models.py --checkpoints checkpoints/cifar10_se_r8/results.json checkpoints/cifar10_se_r16/results.json checkpoints/cifar10_se_r32/results.json --labels "SE-R8" "SE-R16" "SE-R32" --metric accuracy
```

消融实验结果将帮助您深入理解模型各组件的作用，并为改进网络架构提供科学依据。

## 常见问题

### PyTorch 2.6+ 兼容性问题

如果使用 PyTorch 2.6+ 版本，可能遇到检查点加载问题。代码已添加多层错误处理机制，应能自动处理大多数兼容性问题。如果仍然遇到问题，可以尝试：

1. 降级到 PyTorch 2.5 或更早版本
2. 重新训练并保存模型（新保存的检查点会兼容当前版本）
3. 使用 `--weights_only=False` 参数手动加载检查点

### 内存不足问题

如果遇到显存不足问题，可以尝试：

1. 减小批量大小 (`--batch_size`)
2. 使用更小的模型 (SE-ResNet18 而非 SE-ResNet34)
3. 增大 SE 模块的缩减率 (`--se_reduction` 设为 32 或更高)
4. 降低图像分辨率或减少数据增强的复杂度
