import torch
import argparse
import os
from models.se_resnet import se_resnet18, se_resnet34
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR10/100 Inference with PyTorch')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 or cifar100)')
    parser.add_argument('--model', default='se_resnet18', type=str, help='model name (se_resnet18 or se_resnet34)')
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint file path')
    parser.add_argument('--image', default='', type=str, help='image file path for single inference')
    parser.add_argument('--image_dir', default='', type=str, help='directory containing images for batch inference')
    parser.add_argument('--output_dir', default='inference_results', type=str, help='directory to save inference results')
    parser.add_argument('--use_se', default=True, type=bool, help='use squeeze and excitation module')
    parser.add_argument('--se_reduction', default=16, type=int, help='squeeze and excitation reduction ratio')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer (not used in inference, but added for consistency)')
    return parser.parse_args()

def get_class_names(dataset):
    if dataset == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    else:  # cifar100
        return torchvision.datasets.CIFAR100(root='./data', train=False, download=True).classes

def inference_single_image(model, image_path, transform, class_names, device):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # 获取前5个预测结果
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # 显示结果
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    y_pos = np.arange(5)
    plt.barh(y_pos, top5_prob.cpu().numpy())
    plt.yticks(y_pos, [class_names[idx] for idx in top5_idx.cpu().numpy()])
    plt.xlabel('Probability')
    plt.title('Top-5 Predictions')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    # 打印结果
    print('Top-5 predictions:')
    for i in range(5):
        print(f'{class_names[top5_idx[i]]}: {top5_prob[i]:.4f}')

def inference_test_set(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def inference_image_directory(model, directory_path, transform, class_names, device, output_dir):
    """
    对目录中的所有图片进行批量推理
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_file = os.path.join(output_dir, 'inference_results.txt')
    
    # 获取目录中的所有图片文件
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(directory_path) 
                  if os.path.isfile(os.path.join(directory_path, f)) 
                  and os.path.splitext(f.lower())[1] in valid_extensions]
    
    if not image_files:
        print(f"在目录 {directory_path} 中没有找到有效的图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始推理...")
    
    # 设置模型为评估模式
    model.eval()
    results = []
    
    # 打开结果文件
    with open(results_file, 'w') as f:
        f.write(f"文件名,预测类别,置信度\n")
        
        # 为每个图片进行推理
        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            try:
                # 加载图像
                image = Image.open(image_path).convert('RGB')
                
                # 预处理
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # 推理
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                # 获取预测结果
                top_prob, top_class = probabilities.max(0)
                predicted_class = class_names[top_class.item()]
                confidence = top_prob.item() * 100
                
                # 记录结果
                results.append((image_file, predicted_class, confidence))
                f.write(f"{image_file},{predicted_class},{confidence:.2f}%\n")
                
                # 打印进度
                print(f"处理: {image_file} - 预测: {predicted_class} (置信度: {confidence:.2f}%)")
                
                # 创建可视化结果图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # 显示原图
                ax1.imshow(image)
                ax1.set_title('输入图像')
                ax1.axis('off')
                
                # 显示Top-3预测
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                top3_classes = [class_names[idx] for idx in top3_idx.cpu().numpy()]
                
                y_pos = np.arange(3)
                ax2.barh(y_pos, top3_prob.cpu().numpy() * 100)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(top3_classes)
                ax2.set_xlabel('置信度 (%)')
                ax2.set_title('Top-3 预测结果')
                
                # 保存可视化结果
                fig_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_result.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close(fig)
                
            except Exception as e:
                print(f"处理 {image_file} 时出错: {e}")
                f.write(f"{image_file},处理失败,0%\n")
    
    # 创建汇总报告
    print(f"\n推理完成! 共处理 {len(results)} 个图片")
    print(f"详细结果已保存至: {results_file}")
    print(f"可视化结果已保存至: {output_dir}")
    
    # 返回结果
    return results

def main():
    args = parse_args()
    
    # 明确导入torch，避免局部变量作用域问题
    import torch
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 获取类别名称
    class_names = get_class_names(args.dataset)
    num_classes = len(class_names)
    
    # 加载模型
    if args.model == 'se_resnet18':
        model = se_resnet18(num_classes=num_classes, use_se=args.use_se, 
                           se_reduction=args.se_reduction, dropout_rate=args.dropout_rate)
    else:  # se_resnet34
        model = se_resnet34(num_classes=num_classes, use_se=args.use_se, 
                           se_reduction=args.se_reduction, dropout_rate=args.dropout_rate)
    
    # 加载检查点
    if os.path.isfile(args.checkpoint):
        print(f'Loading checkpoint {args.checkpoint}')
        
        # 确保在此作用域中有 torch
        import torch as torch_module
        
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
                    return
        
        # 加载模型权重
        if 'state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['state_dict'])
                print(f'Loaded checkpoint (epoch {checkpoint.get("epoch", "unknown")}, accuracy {checkpoint.get("best_acc", 0.0):.2f}%)')
            except Exception as e:
                print(f"加载状态字典失败: {e}")
                print("尝试加载不严格匹配的状态字典...")
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f'使用非严格模式加载了检查点')
        else:
            print("检查点文件格式异常，找不到 state_dict 键")
            return
    else:
        print(f'No checkpoint found at {args.checkpoint}')
        return
    
    model = model.to(device)
    
    # 数据预处理
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 选择推理模式
    if args.image:
        # 单张图像推理
        inference_single_image(model, args.image, transform, class_names, device)
    elif args.image_dir:
        # 目录批量推理
        inference_image_directory(model, args.image_dir, transform, class_names, device, args.output_dir)
    else:
        # 测试集推理
        if args.dataset == 'cifar10':
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform)
        else:  # cifar100
            test_dataset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=4)
        
        inference_test_set(model, test_loader, device)

if __name__ == '__main__':
    main()
