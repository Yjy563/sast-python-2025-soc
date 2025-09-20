import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image


# 自定义数据集类用于从文件夹加载图片并生成索引
class CustomImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(root_dir))
        self.file_to_id = {name: int(os.path.splitext(name)[0]) for name in self.image_files}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 返回图像和其对应的 ID
        return image, self.file_to_id[img_name]


def main():
    print("--- Step 1: 环境和数据准备 ---")
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载数据集 (训练集和标准测试集一次性加载)
    print("正在加载CIFAR-10训练和标准测试数据集...")
    trainset = torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    print("CIFAR-10数据集加载完成。")

    # 加载自定义的测试数据集
    print("正在加载自定义测试数据集...")
    test_data_path = 'dataset1/test'
    if not os.path.exists(test_data_path):
        print(f"错误: 找不到文件夹 '{test_data_path}'。请确保你的图片文件在这个路径下。")
        return

    predict_dataset = CustomImageFolderDataset(root_dir=test_data_path, transform=transform_test)
    predict_loader = DataLoader(predict_dataset, batch_size=100, shuffle=False, num_workers=2)
    print(f"已从 '{test_data_path}' 加载 {len(predict_dataset)} 张图片。")

    print("\n--- Step 2: 模型定义 ---")

    # ResNet基本块
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.relu(out)
            return out

    # ResNet18模型
    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = torch.nn.functional.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])

    model = ResNet18().to(device)
    print("模型初始化完成。")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 训练和测试函数
    def train(epoch):
        print(f'--- 训练 Epoch: {epoch} ---')
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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
        print(
            f'Epoch {epoch} 训练结束 | 训练准确率: {100. * correct / total:.3f}% | 训练损失: {train_loss / len(trainloader):.3f}')

    # 现在 test 函数不再需要重新加载数据集
    def test():
        print('--- 正在进行测试评估 ---')
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        print(f'测试评估完成 | 测试准确率: {acc:.3f}% | 测试损失: {test_loss / len(testloader):.3f}')
        return acc

    print("\n--- Step 3: 开始模型训练 ---")
    best_acc = 0
    # 训练模型（这里使用CIFAR-10测试集进行评估）
    for epoch in range(1, 21):
        train(epoch)
        acc = test()
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'resnet18_cifar10_best.pth')
            print(f'*** 最佳准确率更新: {best_acc:.3f}%，模型已保存 ***')

    print(f"\n训练完成！最终最佳准确率: {best_acc:.3f}%")

    print("\n--- Step 4: 加载最佳模型并生成提交文件 ---")
    model.load_state_dict(torch.load('resnet18_cifar10_best.pth'))
    model.eval()

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predictions = []
    ids = []

    print("正在对自定义测试集进行预测...")
    with torch.no_grad():
        for batch_idx, (inputs, indices) in enumerate(predict_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i, pred in enumerate(predicted):
                ids.append(indices[i].item())
                predictions.append(classes[pred.item()])
    print("预测完成，正在生成提交文件...")

    submission = pd.DataFrame({
        'ID': ids,
        'label': predictions
    })

    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv('submission.csv', index=False)
    print("提交文件已保存为 submission.csv")

    print("\n--- Step 5: 结果展示和参数统计 ---")
    print(f"提交文件包含 {len(submission)} 个预测结果。")
    print("前10个预测结果:")
    print(submission.head(10))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    print('\n前5个标准CIFAR-10测试样本的预测结果:')
    testloader_for_display = DataLoader(testset, batch_size=5, shuffle=False, num_workers=0)
    dataiter = iter(testloader_for_display)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(5):
        print(f'真实标签: {classes_cifar10[labels[i]]}, 预测标签: {classes_cifar10[predicted[i]]}')


if __name__ == '__main__':
    main()