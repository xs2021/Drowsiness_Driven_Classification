import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision
import numpy as np
from tqdm import tqdm
from utils.my_dataset import get_loader
from models.vision_transformer import VisionTransformer
from utils.my_utils import set_seed, AverageMeter, WarmupCosineSchedule
import os

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def predict_single_image(image_path, model, transform, classes):
    im = Image.open(image_path)
    im = transform(im).to(DEVICE)   # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # 增加一个维度放到第一列, 变成[batch, C, H, W]

    model.eval()  # 评估模式

    with torch.no_grad():
        outputs = model(im)
        pre_class_i = outputs.argmax(1).item()

    print("File:", image_path)
    print("softmax输出：", outputs)
    print("预测结果(取最大概率的下标): ", pre_class_i)
    print("预测结果： ", classes[pre_class_i])
    print("-" * 50)


def predict():
    transform = transforms.Compose([
        # transforms.Resize((256,256)),
        # transforms.CenterCrop((224,224)),  # 加上后预测结果稍微变化
        transforms.Resize((224,224)),
        transforms.ToTensor(),   # 转换PIL.Image or numpy.ndarray 为 torch 张量
        transforms.Normalize([0.151, 0.136, 0.129], [0.058, 0.051, 0.048])
    ])

    classes = ('closed', 'no_yawn', 'open', 'yawn')

    config = {
        'seed': 42,
        'img_size': [224, 224],
        'num_classes': 4,
        'train_batch_size': 32,
        'val_batch_size': 16,
        'test_batch_size': 16,

        'patch_size': [16, 16],
        'hidden_size': 768,
        'mlp_dim': 3072,
        'num_heads': 12,
        'num_layers': 12,
        'dropout_rate': 0.1,
        'attention_dropout_rate': 0.0
    }
    train_loader, val_loader, test_loader = get_loader(config)

    model = VisionTransformer(config)
    model.to(DEVICE)

    # 加载模型并测试
    model.load_state_dict(torch.load('./checkpoints/vit_b_16_3266_0.9735449735449735.pth')) 

    # Path to the folder containing images
    folder_path = '../pic_predict'

    # Loop through each file in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            predict_single_image(image_path, model, transform, classes)


if __name__ == '__main__':
    predict()
