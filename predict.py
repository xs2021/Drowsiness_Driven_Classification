import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import torchvision

NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # # VGG16
    # from model.model_vgg import vggNet
    # model = vggNet('vgg16', in_channels=3 ,num_classes=NUM_CLASSES, init_weights=True).to(DEVICE)
    # state_dict = torch.load("./weights/0302BestVGG16.pth")
    # model.load_state_dict(state_dict)

    # ResNet65
    from model.model_res import resnet65
    model = resnet65(NUM_CLASSES).to(DEVICE)
    state_dict = torch.load("./weights/0302BestRes65.pth")
    model.load_state_dict(state_dict)

    # VGG16_Pre
    # model = torchvision.models.vgg16().to(DEVICE)  # 模型实例化
    # last_layer_in_features = model.classifier[-1].in_features    # 获取最后一层全连接层的输入特征数量
    # model.classifier[-1] = nn.Linear(last_layer_in_features, 1024)      # 修改全连接层，使得全连接层的输出与当前数据集类别数对应
    # model.classifier.add_module("last_Linear", module=nn.Linear(1024, NUM_CLASSES))
    # model = model.to(DEVICE)  # 将模型移动到设备上（如 GPU）
    # state_dict = torch.load("./weights/0303BestVGG16_Pre.pth")
    # model.load_state_dict(state_dict)
    
    # GoogLeNet
    # model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(DEVICE)
    # last_layer_in_features = model.fc.in_features
    # model.fc = nn.Linear(last_layer_in_features, NUM_CLASSES).to(DEVICE) # 修改全连接层，使得全连接层的输出与当前数据集类别数对应
    # state_dict = torch.load("./weights/0304BestGoogle_Pre.pth")
    # model.load_state_dict(state_dict)

    # ResNet_Pre
    # model = torchvision.models.resnet50(pretrained=True).to(DEVICE)
    # last_layer_in_features = model.fc.in_features
    # model.fc = nn.Linear(last_layer_in_features, NUM_CLASSES).to(DEVICE) # 修改全连接层，使得全连接层的输出与当前数据集类别数对应
    # state_dict = torch.load("./weights/0303BestResNet50_Pre.pth")
    # model.load_state_dict(state_dict)

    # Path to the folder containing images
    folder_path = './pic_predict'

    # Loop through each file in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            predict_single_image(image_path, model, transform, classes)


if __name__ == '__main__':
    predict()
