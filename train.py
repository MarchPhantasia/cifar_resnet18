import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.se_resnet import se_resnet18, se_resnet34

# 参数设置
def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR10/100 Training with PyTorch')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 or cifar100)')
    parser.add_argument('--model', default='se_resnet18', type=str, help='model name (se_resnet18 or se_resnet34)')
    parser.add_argument('--use_se', default=True, type=bool, help='use squeeze and excitation module')
    parser.add_argument('--se_reduction', default=16, type=int, help='squeeze and excitation reduction ratio')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--batch_size', default=50, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='ratio of training data used for validation')
    parser.add_argument('--no_augmentation', action='store_true', help='禁用数据增强，仅使用基础变换')
    parser.add_argument('--save_dir', default='checkpoints', type=str, help='directory to save checkpoints')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--final_test', action='store_true', help='run final test on test set after training')
    return parser.parse_args()

# 数据加载和预处理
def get_data_loaders(dataset, batch_size, val_ratio=0.1, seed=42, no_augmentation=False):
    """
    加载数据集并划分训练集、验证集和测试集
    
    Args:
        dataset: 数据集名称 ('cifar10' 或 'cifar100')
        batch_size: 批处理大小
        val_ratio: 用于验证的训练数据比例
        seed: 随机种子，确保可复现性
        no_augmentation: 是否禁用数据增强
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        num_classes: 类别数量
    """
    # 设置随机种子以确保划分的可复现性
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 数据增强和标准化
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        num_classes = 10
        
        # 训练集数据增强
        if no_augmentation:
            # 如果禁用数据增强，只使用基础变换
            print("禁用数据增强 - 仅使用基础变换")
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            # 使用完整数据增强
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        # 验证集和测试集不使用数据增强
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # 加载完整训练集
        full_train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform)
        
        # 加载测试集
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=val_test_transform)
        
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        num_classes = 100
        
        # 训练集数据增强
        if no_augmentation:
            # 如果禁用数据增强，只使用基础变换
            print("禁用数据增强 - 仅使用基础变换")
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            # 使用完整数据增强
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        # 验证集和测试集不使用数据增强
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # 加载完整训练集
        full_train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform)
        
        # 加载测试集
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=val_test_transform)
    
    # 从训练集中划分验证集
    dataset_size = len(full_train_dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    # 使用 random_split 划分训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 为验证集设置正确的变换（不使用数据增强）
    # 创建一个新的验证数据集，使用相同的图像但应用验证集的变换
    # 注意：这需要重新加载数据集
    if dataset == 'cifar10':
        val_dataset_no_aug = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=val_test_transform)
    else:
        val_dataset_no_aug = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=val_test_transform)
    
    # 获取验证集的索引
    val_indices = val_dataset.indices
    
    # 创建验证集的子集
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"数据集大小 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, num_classes

# 学习率调整策略
def adjust_learning_rate(optimizer, epoch, args):
    """Cosine学习率调整策略"""
    lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 训练一个epoch
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')
    
    return train_loss/len(train_loader), 100.*correct/total

# 测试
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/len(test_loader), 100.*correct/total

# 保存检查点
def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        # 从文件名中提取前缀（例如：cifar10_se_resnet18）
        prefix = filename.split('_latest')[0] if '_latest' in filename else 'model'
        best_filename = f"{prefix}_best.pth.tar"
        torch.save(state, os.path.join(save_dir, best_filename))

# 绘制训练过程
def plot_training(train_losses, train_accs, val_losses, val_accs, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据加载
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, args.val_ratio, args.seed, args.no_augmentation)
    
    # 模型选择
    if args.model == 'se_resnet18':
        model = se_resnet18(num_classes=num_classes, use_se=args.use_se, 
                          se_reduction=args.se_reduction, dropout_rate=args.dropout_rate)
    else:  # se_resnet34
        model = se_resnet34(num_classes=num_classes, use_se=args.use_se, 
                          se_reduction=args.se_reduction, dropout_rate=args.dropout_rate)
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 从检查点恢复
    start_epoch = 0
    best_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint {args.resume}')
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_losses = checkpoint.get('train_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_losses = checkpoint.get('val_losses', [])
            val_accs = checkpoint.get('val_accs', [])
            print(f'Loaded checkpoint {args.resume} (epoch {checkpoint["epoch"]})')
        else:
            print(f'No checkpoint found at {args.resume}')
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 调整学习率
        adjust_learning_rate(optimizer, epoch, args)
        
        # 训练
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        
        # 在验证集上评估
        val_loss, val_acc = test(model, val_loader, criterion, device)
        
        # 记录损失和准确率
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印结果
        print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}%')
        
        # 保存检查点
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        # 创建包含模型类型的文件名
        model_prefix = f"{args.dataset}_{args.model}"
        
        # 每个 epoch 保存最新的 checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
        }, is_best, args.save_dir, filename=f'{model_prefix}_latest.pth.tar')
        
        # 每 10 个 epoch 保存一次 checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
            }, False, args.save_dir, filename=f'{model_prefix}_epoch_{epoch+1}.pth.tar')
            
            # 绘制训练过程
            plot_training(train_losses, train_accs, val_losses, val_accs, args.save_dir)
    
    # 训练结束后在测试集上进行最终评估
    if args.final_test:
        print("\n在测试集上进行最终评估...")
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f'最终测试集结果 - Loss: {test_loss:.3f} | Accuracy: {test_acc:.3f}%')
        
        # 保存测试结果
        with open(os.path.join(args.save_dir, f'{model_prefix}_test_results.txt'), 'w') as f:
            f.write(f'Test Loss: {test_loss:.4f}\n')
            f.write(f'Test Accuracy: {test_acc:.2f}%\n')

if __name__ == '__main__':
    main()
