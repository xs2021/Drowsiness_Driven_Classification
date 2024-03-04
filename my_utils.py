import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from torch.optim import lr_scheduler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4

# draw_picture
def plot_acc_loss(train_val_DataFrame):
    plt.figure(figsize=(12, 4))
    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_val_DataFrame["epoch"], train_val_DataFrame.train_loss_all, 'ro-', label='Train Loss')
    plt.plot(train_val_DataFrame["epoch"], train_val_DataFrame.val_loss_all, 'bs-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Plot Accuracies
    plt.subplot(1, 2, 2)
    # ro-: 绘制散点图-红色-圆圈-实线相连， bs-: 绘制散点图-蓝色-方块-实线相连
    plt.plot(train_val_DataFrame["epoch"], train_val_DataFrame.train_acc_all, 'ro-', label='Train Accuracy')
    plt.plot(train_val_DataFrame["epoch"], train_val_DataFrame.val_acc_all, 'bs-', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()  # 自动调整子图, 使子图适应整个绘图区
    plt.show()

# model_fine_tuning
def model_fine_tuning(model):
    # VGGNet
    # 获取最后一层全连接层的输入特征数量
    # last_layer_in_features = model.classifier[-1].in_features
    # # 修改全连接层，使得全连接层的输出与当前数据集类别数对应
    # model.classifier[-1] = nn.Linear(last_layer_in_features, 1024)
    # model.classifier.add_module("last_Linear", module=nn.Linear(1024, NUM_CLASSES))
    # print("=========================================")
    # print("修改后: ", model)
    # # 冻结所有层的参数
    # for param in model.parameters():
    #     param.requires_grad = False
    # # 解冻最后一层全连接层的参数
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    # model = model.to(DEVICE)  # 将模型移动到设备上（如 GPU）
    
    # ResNet & GoogLeNet
    # 法1：只微调训练模型最后一层（全连接分类层）
    # 获取最后一层全连接层的输入特征数量
    last_layer_in_features = model.fc.in_features
    # 修改全连接层，使得全连接层的输出与当前数据集类别数对应
    model.fc = nn.Linear(last_layer_in_features, NUM_CLASSES).to(DEVICE)
    print("=========================================")
    print("修改后: ", model)
    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = False
    # 解冻最后一层全连接层的参数
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(DEVICE)  # 将模型移动到设备上（如 GPU）

    return model

# 学习率调整: 预热（warmup）阶段和余弦退火（cosine annealing）策略
def warmup_cosine_schedule(epoch, warmup_epochs, total_epochs, last_epoch=-1):
    if epoch < warmup_epochs:
        # 线性预热策略
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        # 余弦退火策略
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))