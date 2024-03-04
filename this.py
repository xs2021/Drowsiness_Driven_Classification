import os
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split  # 自定义数据集，数据加载器
from my_utils import plot_acc_loss, model_fine_tuning, warmup_cosine_schedule  # 导入自定义的工具包

# 定义超参数
LR = 1e-2
EPOCHS = 50
BATCH_SIZE = 32
INPUT_IMG_SIZE = 224  # 输入尺寸
NUM_CLASSES = 4       # 几分类
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MDOEL_PATH = "./weights" # 模型保存路径
# BestVGG16.pth       BestRes65.pth       BestGoogle.pth
# BestVGG16_Pre.pth   BestRes50_Pre.pth   BestGoogle_Pre.pth
MODEL = "0304BestGoogle_Pre.pth"  # 保存的模型名称，后缀 pth/pkl/h5

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_IMG_SIZE),
        # transforms.RandomResizedCrop(INPUT_IMG_SIZE, scale=(0.8, 1.0)),  # 调整scale以减少过度裁剪
        transforms.RandomHorizontalFlip(), # 水平翻转，因为不改变闭眼、打哈欠等状态
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02), # 色彩抖动应当谨慎使用，以保持面部特征的真实性
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.151, 0.136, 0.129], std=[0.058, 0.051, 0.048]) # 需自己计算
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.151, 0.136, 0.129], [0.058, 0.051, 0.048])
    ]),
}

def dataInit():
    train_set = 0.0
    test_set = 0.0
    # 加载自定义数据集
    ROOT_TRAIN = "./data/Drowsiness_Driven_Dataset/train/"
    ROOT_TEST = "./data/Drowsiness_Driven_Dataset/test/"
    print(ROOT_TRAIN)
    # ImageFolder 会根据文件夹自动生成不同的标签
    try:
        train_set = datasets.ImageFolder(root=ROOT_TRAIN, transform=data_transforms['train'])
        test_set = datasets.ImageFolder(root=ROOT_TEST, transform=data_transforms['test'])
        print(train_set.class_to_idx)
        print(test_set.class_to_idx)
    except Exception as e:
        print(f"发生异常: {e}")

    # 划分训练集和验证集
    train_set, val_set = random_split(train_set, [int(round(0.8 * len(train_set))), int(round(0.2 * len(train_set)))])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train(model, train_loader, criterion, optimizer, epoch, val_interval=50):
    print(f'---------- Train Start 第 {epoch} 轮 ----------')
    train_loss_all = []
    train_acc_all = []
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    # 使用进度条
    from tqdm import tqdm
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (batch_x, batch_y) in loop:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        model.train()
        output = model(batch_x)  # 前向传播：获取一个批次的预测结果
        output_class = output.argmax(1)  # 获取一个批次预测结果所对应的最大概率的下标-即类别（每一个结果都是softmax概率）

        optimizer.zero_grad()  # 梯度置0,防止累加上一批次的结果
        loss = criterion(output, batch_y)
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 梯度下降法更新参数
        train_loss += loss.item() * batch_x.size(0)  # Accumulate loss for the epoch
        train_total += batch_y.size(0)  # 所有的样本
        train_correct += output_class.eq(batch_y).sum().item()
        train_acc = train_correct / train_total  # 计算准确率: 正确数量/总数量
        # if (step + 1) % val_interval == 0:
        #     print(f'[Epoch,Step]: [{epoch},{step + 1}] | train loss: {loss.item():.4f} | train accuracy: {train_acc:.4f}')
        # 进度条显示信息
        loop.set_description(f'Epoch [{epoch}]')
        loop.set_postfix(loss=loss.item(), acc=train_acc)

        scheduler.step()  # 学习率衰减
    
    print('下一轮 Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
    train_loss_all.append(train_loss / train_total)
    train_acc_all.append(train_correct / train_total)
    print(f'[Epoch]: [{epoch}] | train loss: {train_loss_all[-1]:.4f} | train accuracy: {train_acc_all[-1]:.4f}')
    print(f'---------- Train Finished 第 {epoch} 轮 ----------')
    return train_loss_all, train_acc_all

def val(model, val_loader, criterion, optimizer, scheduler, epoch):
    print('------ Val Start -----')
    val_loss_all = []
    val_acc_all = []
    val_loss = 0.0
    val_correct = 0.0
    val_total = 0.0
    p_n = []  # 预测类别(标签)
    r_n = []  # 原始类别(标签)
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(val_loader):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            model.eval()
            output = model(batch_x)
            output_class = output.argmax(1)  # 获取一个批次预测结果所对应的最大概率的下标-即类别（每一个结果都是softmax概率）
            loss = criterion(output, batch_y)
            val_loss += loss.item() * batch_x.size(0)  # Accumulate loss for the epoch
            val_total += batch_y.size(0)
            val_correct += output_class.eq(batch_y).sum().item()
            p_n.extend(output_class.tolist())  # 将预测的类别追加到列表
            r_n.extend(batch_y.tolist())  # 将原始类别追加到列表

    val_loss_all.append(val_loss / val_total)  # compute average loss over all batches
    val_acc_all.append(val_correct / val_total)
    print(f'[Epoch]: [{epoch}] | val loss: {val_loss_all[-1]:.4f} | val accuracy: {val_acc_all[-1]:.4f}')
    print('------ Val Finished -----')
    return val_loss_all, val_acc_all

def test(model, test_loader, criterion, epoch):
    from sklearn.metrics import confusion_matrix
    print('------ Test Start -----')
    test_loss_all = []
    test_acc_all = []
    test_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    p_n = []  # 预测类别(标签)
    r_n = []  # 原始类别(标签)
    classes = ('closed', 'no_yawn', 'open', 'yawn')
    confusion_matrix_result = None

    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            model.eval()
            output = model(batch_x)
            output_class = output.argmax(1)  # 获取一个批次预测结果所对应的最大概率的下标-即类别（每一个结果都是softmax概率）
            loss = criterion(output, batch_y)
            test_loss += loss.item() * batch_x.size(0)  # Accumulate loss for the epoch
            test_total += batch_y.size(0)
            test_correct += output_class.eq(batch_y).sum().item()
            p_n.extend(output_class.tolist())  # 将预测的类别追加到列表
            r_n.extend(batch_y.tolist())  # 将原始类别追加到列表

    test_loss_all.append(test_loss / test_total)  # compute average loss over all batches
    test_acc_all.append(test_correct / test_total)
    print(f'[Epoch]: [{epoch}] | test loss: {test_loss_all[-1]:.4f} | test accuracy: {test_acc_all[-1]:.4f}')

    # 打印一部分预测的类别与原始类别
    print('Prediction labels: ', p_n[:30])
    print('       Raw labels: ', r_n[:30])
    print('Prediction labels: ', [classes[pred] for pred in p_n[:30]])
    print('       Raw labels: ', [classes[raw] for raw in r_n[:30]])
    print('------ Test Finished -----')

    # 计算混淆矩阵
    confusion_matrix_result = confusion_matrix(r_n, p_n)
    print("混淆矩阵[行代表真实类别,列代表预测类别,即对角线为正确的预测数量]:")
    print(confusion_matrix_result)
    return test_loss_all, test_acc_all, confusion_matrix_result

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, val_interval=50):
    best_acc = 0.0  # 获取在验证集最高的精确度
    best_model_wts = 0.0  # 获取在验证集最佳权重
    since_time = time.time()  # 当前时间
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(1, epochs + 1):  # [1 - EPOCHS+1)
        train_loss_epoch, train_acc_epoch = train(model, train_loader, criterion, optimizer, epoch, val_interval)
        val_loss_epoch, val_acc_epoch = val(model, val_loader, criterion, optimizer, scheduler, epoch)
        # 将每一个epoch列表的值整合到一个列表中
        train_loss_all.append(train_loss_epoch[-1]) # [-1] 每次取最后一个
        train_acc_all.append(train_acc_epoch[-1])
        val_loss_all.append(val_loss_epoch[-1])
        val_acc_all.append(val_acc_epoch[-1])
        # 寻找最高准确度和权重
        if val_acc_epoch[-1] > best_acc:
            best_acc = val_acc_epoch[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            os.makedirs(MDOEL_PATH, exist_ok=True)
            torch.save(best_model_wts, os.path.join(MDOEL_PATH, MODEL))  # 保存最佳模型
        # 计算耗时
        time_use = time.time() - since_time
        print(f"训练和验证耗时： {time_use // 60:.2f}min {time_use % 60:.4f}s")

    # 训练和验证的数据（使用DataFrame格式）
    train_val_data = pd.DataFrame(data={"epoch": range(1, epochs + 1),
                                         "train_loss_all": train_loss_all,
                                         "train_acc_all": train_acc_all,
                                         "val_loss_all": val_loss_all,
                                         "val_acc_all": val_acc_all})
    return train_val_data, best_model_wts

if __name__ == "__main__":
    # 1. 数据加载
    train_loader, val_loader, test_loader = dataInit()  # 自定义数据集（input:3*224*224, 记得改模型最后的输出）

    # 2. 加载模型
    # 2.1 加载自定义模型
    # VGG16
    # from model.model_vgg import vggNet
    # model = vggNet('vgg19', in_channels=3 ,num_classes=NUM_CLASSES, init_weights=True).to(DEVICE)

    # ResNet65
    from model.model_res import resnet65
    model = resnet65(NUM_CLASSES).to(DEVICE)

    # GoogLeNet
    # from model.model_google import GoogLeNet
    # model = GoogLeNet(in_channels=3, num_classes=NUM_CLASSES, init_weights=True, aux_softmax=False).to(DEVICE)
    
    # print(model)
    # print(summary(model,(3, 224, 224)))
    
    # 2.2 加载预训练模型
    # VGG16
    # model = torchvision.models.vgg16(pretrained=True).to(DEVICE)  # 模型实例化
    # print("修改前: ",model)
    # print(summary(model,(3, 224, 224)))
    # # 使用迁移学习
    # model = model_fine_tuning(model)

    # ResNet50
    # model = torchvision.models.resnet50(weights=torchvision.models.VGG16_Weights.DEFAULT).to(DEVICE)  # 最新版写法
    # model = torchvision.models.resnet50(pretrained=True).to(DEVICE)
    # print("修改前: ",model)
    # print(summary(model,(3, 224, 224)))
    # # 使用迁移学习
    # model = model_fine_tuning(model)

    # GoogLeNet
    # model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(DEVICE)
    # print("修改前: ",model)
    # print(summary(model,(3, 224, 224)))
    # # 使用迁移学习
    # model = model_fine_tuning(model)

    # 3. 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()  # CrossEntropyLoss附带softmax    
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=5e-4)
    # 学习率衰减通常在训练集使用
    # 法1：学习率每8个epoch衰减成原来的1/10
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1) 
    # 法2：预热和余弦退火策略
    warmup_epochs = 5  # 预热阶段的epoch数
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
                                            lr_lambda=lambda epoch: warmup_cosine_schedule(epoch, warmup_epochs, EPOCHS))

    # 4. 开始训练和验证,返回训练和验证过程中的信息
    train_val_DataFrame, best_model_wts = train_and_evaluate(model,train_loader,val_loader,loss_function,
                                                             optimizer,scheduler,EPOCHS,100)
    
    # 5. 绘制准确率和损失
    plot_acc_loss(train_val_DataFrame)

    # 6. 加载模型并测试
    # model.load_state_dict(torch.load(MODEL))
    model.load_state_dict(best_model_wts)   # 给模型加载最佳权重
    test(model,test_loader,loss_function,EPOCHS)