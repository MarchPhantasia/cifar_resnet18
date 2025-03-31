import torch
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from models.se_resnet import se_resnet18, se_resnet34
import matplotlib as mpl
import sys
import locale

# 配置系统编码
def configure_encoding():
    """配置系统编码为UTF-8"""
    try:
        # 检测当前系统编码
        current_encoding = locale.getpreferredencoding()
        print(f"当前系统默认编码: {current_encoding}")
        
        # 尝试设置默认编码（Python 3中这通常不起作用，但保留为兼容性考虑）
        if hasattr(sys, 'setdefaultencoding'):
            sys.setdefaultencoding('utf-8')
            
        # 确保标准输出使用UTF-8
        if sys.stdout.encoding != 'utf-8':
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                print("已将标准输出重新配置为UTF-8")
                
        # 提示信息
        if current_encoding.lower() != 'utf-8' and current_encoding.lower() != 'utf8':
            print("警告: 系统默认编码不是UTF-8，可能导致中文显示问题")
            print("建议设置环境变量 PYTHONIOENCODING=utf-8 再运行程序")
            
        return True
    except Exception as e:
        print(f"配置编码时出现错误: {e}")
        return False

# 尝试配置系统编码
configure_encoding()

# 解决中文显示问题
def configure_matplotlib_fonts():
    """配置matplotlib以支持中文显示"""
    # 尝试设置中文字体
    try:
        # 检查操作系统类型
        import platform
        system = platform.system()
        
        if system == "Windows":
            # Windows系统尝试使用微软雅黑
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        elif system == "Darwin":  # MacOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Heiti TC', 'Arial Unicode MS']
        else:  # Linux等其他系统
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'AR PL UMing CN', 'Arial Unicode MS']
        
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        return True
    except Exception as e:
        print(f"配置中文字体时出现错误: {e}")
        return False

# 尝试配置中文字体
use_chinese = configure_matplotlib_fonts()

"""
CIFAR-10/100 模型最终测试与评估工具

本工具用于在训练完成后对模型在官方测试集上进行最终评估，并生成详细的性能分析报告。
符合机器学习最佳实践，将测试集严格用于最终评估，而非模型选择或超参数调优。

主要功能：
1. 在完整测试集上评估模型性能
2. 生成混淆矩阵可视化
3. 分析每个类别的准确率
4. 输出详细的分类报告，包括精确率、召回率和F1分数
5. 识别模型表现最好和最差的类别

使用方法：
python test_model.py --dataset cifar10 --model se_resnet18 --checkpoint checkpoints/cifar10_se_resnet18_best.pth.tar
"""

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR10/100模型最终测试和性能分析工具')
    parser.add_argument('--dataset', default='cifar10', type=str, help='数据集 (cifar10 或 cifar100)')
    parser.add_argument('--model', default='se_resnet18', type=str, help='模型类型 (se_resnet18 或 se_resnet34)')
    parser.add_argument('--checkpoint', required=True, type=str, help='模型检查点路径')
    parser.add_argument('--batch_size', default=100, type=int, help='批量大小')
    parser.add_argument('--use_se', default=True, type=bool, help='是否使用SE模块')
    parser.add_argument('--se_reduction', default=16, type=int, help='SE模块缩减率')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='Dropout丢弃率')
    parser.add_argument('--save_dir', default='results', type=str, help='结果保存目录')
    return parser.parse_args()

def get_data_loader(dataset, batch_size):
    """
    加载官方测试数据集用于最终评估
    
    注意：这里只使用官方的测试集（test_batch），不涉及训练集或验证集
    """
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=test_transform)
        classes = test_dataset.classes
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader, classes

def load_model(args, num_classes):
    """加载模型"""
    # 确保在此作用域中有 torch
    import torch as torch_module
    
    if args.model == 'se_resnet18':
        model = se_resnet18(num_classes=num_classes, use_se=args.use_se,
                          se_reduction=args.se_reduction, dropout_rate=args.dropout_rate)
    else:  # se_resnet34
        model = se_resnet34(num_classes=num_classes, use_se=args.use_se,
                          se_reduction=args.se_reduction, dropout_rate=args.dropout_rate)
    
    device = torch_module.device('cuda' if torch_module.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 加载检查点
    if os.path.isfile(args.checkpoint):
        print(f'加载检查点: {args.checkpoint}')
        
        # 为 PyTorch 2.6+ 添加安全的全局变量
        try:
            import torch.serialization
            # 尝试添加必要的安全全局变量
            torch_module.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
        except (ImportError, AttributeError):
            print("无法添加安全全局变量，将尝试直接加载...")
        
        try:
            # 尝试直接加载
            checkpoint = torch_module.load(args.checkpoint, map_location=device)
        except Exception as e:
            print(f"遇到加载错误: {str(e)[:150]}...")
            print("尝试使用 weights_only=False 加载...")
            try:
                # 尝试使用 weights_only=False
                checkpoint = torch_module.load(args.checkpoint, map_location=device, weights_only=False)
            except Exception as e2:
                print(f"使用 weights_only=False 仍然失败: {str(e2)[:150]}...")
                print("尝试加载简单状态字典...")
                try:
                    # 尝试加载为简单字典
                    state_dict = torch_module.load(args.checkpoint, map_location=device, pickle_module=None)
                    if 'state_dict' in state_dict:
                        checkpoint = state_dict
                    else:
                        # 如果直接是状态字典，构造标准格式
                        checkpoint = {
                            'state_dict': state_dict,
                            'epoch': 0,
                            'best_acc': 0.0
                        }
                except Exception as e3:
                    print(f"所有加载方法都失败。最后错误: {e3}")
                    print("请考虑使用较早版本的PyTorch (1.7-2.5)，或重新保存模型检查点")
                    raise FileNotFoundError(f'无法加载检查点文件: {args.checkpoint}')
        
        # 加载模型权重
        if 'state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['state_dict'])
                print(f'成功加载检查点 (轮次: {checkpoint.get("epoch", "unknown")}, 最佳准确率: {checkpoint.get("best_acc", 0.0):.2f}%)')
            except Exception as e:
                print(f"加载状态字典失败: {e}")
                print("尝试加载不严格匹配的状态字典...")
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f'使用非严格模式加载了检查点')
        else:
            print("检查点文件格式异常，找不到 state_dict 键")
            raise ValueError("检查点文件格式异常，找不到 state_dict 键")
    else:
        raise FileNotFoundError(f'找不到检查点文件: {args.checkpoint}')
    
    return model, device

def test_model(model, test_loader, device):
    """测试模型性能并返回预测结果"""
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 收集预测和真实标签用于后续分析
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'测试损失: {avg_loss:.4f} | 测试准确率: {accuracy:.2f}%')
    
    return accuracy, avg_loss, np.array(all_preds), np.array(all_targets)

def plot_confusion_matrix(all_preds, all_targets, classes, save_path):
    """绘制混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 归一化混淆矩阵（按行）
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 转换类别名称为英文（如果不使用中文）
    if not use_chinese:
        xlabel = 'Predicted Class'
        ylabel = 'True Class'
        title = 'Normalized Confusion Matrix'
    else:
        xlabel = '预测类别'
        ylabel = '真实类别'
        title = '归一化混淆矩阵'
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True if len(classes) <= 20 else False, 
                fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # 保存图像前确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'混淆矩阵已保存至: {save_path}')

def analyze_class_accuracy(all_preds, all_targets, classes, save_path):
    """分析每个类别的准确率"""
    # 计算每个类别的准确率
    class_correct = np.zeros(len(classes))
    class_total = np.zeros(len(classes))
    
    for i in range(len(all_targets)):
        label = all_targets[i]
        class_total[label] += 1
        if all_preds[i] == label:
            class_correct[label] += 1
    
    # 计算每个类别的准确率
    class_accuracy = 100 * class_correct / class_total
    
    # 对类别按准确率排序
    sorted_idx = np.argsort(class_accuracy)
    
    # 转换标签为英文（如果不使用中文）
    if not use_chinese:
        xlabel = 'Accuracy (%)'
        title_all = 'Accuracy by Class'
        title_low = 'Lowest Accuracy Classes'
        title_high = 'Highest Accuracy Classes'
    else:
        xlabel = '准确率 (%)'
        title_all = '各类别准确率'
        title_low = '准确率最低的10个类别'
        title_high = '准确率最高的10个类别'
    
    # 选择要显示的类别数量（CIFAR-10显示全部，CIFAR-100仅显示最好和最差的10个）
    if len(classes) > 20:
        # 对于CIFAR-100，显示最好和最差的10个类别
        worst_idx = sorted_idx[:10]
        best_idx = sorted_idx[-10:]
        plot_idx = np.concatenate([worst_idx, best_idx])
        plot_classes = [classes[i] for i in plot_idx]
        plot_accuracy = class_accuracy[plot_idx]
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.barh(range(10), plot_accuracy[:10], color='salmon')
        plt.yticks(range(10), plot_classes[:10])
        plt.xlabel(xlabel)
        plt.title(title_low)
        plt.xlim(0, 100)
        
        plt.subplot(2, 1, 2)
        plt.barh(range(10), plot_accuracy[10:], color='skyblue')
        plt.yticks(range(10), plot_classes[10:])
        plt.xlabel(xlabel)
        plt.title(title_high)
        plt.xlim(0, 100)
    else:
        # 对于CIFAR-10，显示所有类别
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(classes)), class_accuracy[sorted_idx], color='skyblue')
        plt.yticks(range(len(classes)), [classes[i] for i in sorted_idx])
        plt.xlabel(xlabel)
        plt.title(title_all)
        plt.xlim(0, 100)
    
    # 保存图像前确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'类别准确率分析已保存至: {save_path}')
    
    # 打印总体分类报告
    report = classification_report(all_targets, all_preds, target_names=classes)
    print("\n分类报告:")
    print(report)
    
    # 打印最高和最低准确率的类别
    print("\n最高准确率的类别:")
    top5_idx = np.argsort(class_accuracy)[-5:][::-1]
    for i in top5_idx:
        print(f"{classes[i]}: {class_accuracy[i]:.2f}%")
    
    print("\n最低准确率的类别:")
    bottom5_idx = np.argsort(class_accuracy)[:5]
    for i in bottom5_idx:
        print(f"{classes[i]}: {class_accuracy[i]:.2f}%")
    
    return class_accuracy

def main():
    # 解析参数
    args = parse_args()
    
    # 明确导入torch，避免局部变量作用域问题
    import torch
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 加载数据
    test_loader, classes = get_data_loader(args.dataset, args.batch_size)
    
    # 加载模型
    model, device = load_model(args, len(classes))
    
    # 测试模型
    print(f"\n===== 开始在 {args.dataset} 官方测试集上进行最终评估 =====")
    print(f"模型: {args.model}, 使用SE模块: {args.use_se}, SE缩减率: {args.se_reduction}, Dropout率: {args.dropout_rate}")
    print("------------------------------------------------------")
    
    start_time = time.time()
    accuracy, loss, all_preds, all_targets = test_model(model, test_loader, device)
    test_time = time.time() - start_time
    
    # 创建模型特定的保存目录名
    model_name = os.path.basename(args.checkpoint).split('.')[0]
    result_dir = os.path.join(args.save_dir, f"{args.dataset}_{model_name}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(result_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_preds, all_targets, classes, cm_path)
    
    # 分析类别准确率
    acc_path = os.path.join(result_dir, 'class_accuracy.png')
    class_accuracy = analyze_class_accuracy(all_preds, all_targets, classes, acc_path)
    
    # 创建详细的分类报告
    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    
    # 将结果写入文本文件
    result_file = os.path.join(result_dir, 'test_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"============= {args.dataset} 官方测试集最终评估报告 =============\n\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"使用SE模块: {args.use_se}\n")
        f.write(f"SE缩减率: {args.se_reduction}\n")
        f.write(f"Dropout率: {args.dropout_rate}\n\n")
        
        f.write(f"最终测试准确率: {accuracy:.2f}%\n")
        f.write(f"最终测试损失: {loss:.4f}\n")
        f.write(f"测试时间: {test_time:.2f} 秒\n")
        f.write(f"测试样本数量: {len(all_targets)}\n\n")
        
        f.write("各类别准确率:\n")
        f.write("=============\n")
        # 按准确率从高到低排序
        sorted_indices = np.argsort(class_accuracy)[::-1]
        for i, idx in enumerate(sorted_indices):
            f.write(f"{i+1}. {classes[idx]}: {class_accuracy[idx]:.2f}%\n")
        
        f.write("\n整体性能指标:\n")
        f.write("=============\n")
        f.write(f"总体精确率 (macro avg precision): {report['macro avg']['precision']:.4f}\n")
        f.write(f"总体召回率 (macro avg recall): {report['macro avg']['recall']:.4f}\n")
        f.write(f"总体F1分数 (macro avg f1-score): {report['macro avg']['f1-score']:.4f}\n")
        
        # 添加结论
        f.write("\n测试总结:\n")
        f.write("=============\n")
        f.write(f"该模型在 {args.dataset} 官方测试集上最终准确率为 {accuracy:.2f}%。\n")
        
        # 表现最好和最差的类别
        best_class_idx = np.argmax(class_accuracy)
        worst_class_idx = np.argmin(class_accuracy)
        f.write(f"表现最好的类别: {classes[best_class_idx]} ({class_accuracy[best_class_idx]:.2f}%)\n")
        f.write(f"表现最差的类别: {classes[worst_class_idx]} ({class_accuracy[worst_class_idx]:.2f}%)\n")
        
        f.write("\n注: 此评估在官方测试集上进行，结果代表模型的真实泛化能力。\n")
    
    print("\n===== 测试完成 =====")
    print(f"最终测试准确率: {accuracy:.2f}%")
    print(f"详细结果已保存至: {result_dir}")
    print(f"- 混淆矩阵: {cm_path}")
    print(f"- 类别准确率分析: {acc_path}")
    print(f"- 详细测试报告: {result_file}")

if __name__ == "__main__":
    main() 