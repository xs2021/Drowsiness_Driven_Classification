import torch
import torch.nn as nn

# first version
# class VGGNet(nn.Module):
#     def __init__(self, in_channels=3, num_classes=10):
#         super(VGGNet, self).__init__()
#         # 特征提取层  input(3*224*224)
#         self.features = nn.Sequential(
#             # bolck1: 64*112*112
#             nn.Conv2d(in_channels, 64,3,1,1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2,2),
#             # bolck2: 128*56*56
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#             # bolck3: 256*28*28
#             nn.Conv2d(128, 256, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#             # bolck4: 512*14*14
#             nn.Conv2d(256,512,3,1,1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512,512,3,1,1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2,2),
#             # bolck5: 512*7*7
#             nn.Conv2d(512, 512, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2,2)
#         )
#         # 全连接层
#         self.classifier = nn.Sequential(
#             nn.Linear(512*7*7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5, inplace=True),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5, inplace=True),
#             nn.Linear(4096, num_classes)
#         )
#
#     # 前向传播
#     def forward(self, x):
#         x = self.features(x)
#         print(x.shape)
#         x = nn.Flatten()(x)
#         return self.classifier(x)


# best version
class VGGNet(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
        super(VGGNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, num_classes)
        )
        if init_weights:
            self.init_weights_bias()
            
    def forward(self, x):
        x = self.features(x)  # batch x 3 x 224 x 224 --> batch x 512 x 7 x 7
        x = nn.Flatten()(x)   # batch x 512 x 7 x 7 --> batch x 512*7*7(4096)
        return self.classifier(x)  # batch x 512*7*7(4096) --> batch x num_classes
    
    def init_weights_bias(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight,0,1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def build_features(values: list, in_channels) -> nn.Module:
    layers = []
    for v in values:
        if v=='M':
            layers+= [nn.MaxPool2d(2,2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,3,1,1)
            layers+= [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]  # 自己增加一个归一化
            in_channels = v  # 上一次的输出作为下一次的输入
    # print(*layers, sep='\n')
    return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,'M'],
}

def vggNet(model_name='vgg11',in_channels=3, **kwargs):
    assert model_name in cfgs, 'Unknown model'
    cfg_list = cfgs[model_name]
    cfg_features = build_features(cfg_list, in_channels)
    model = VGGNet(cfg_features, **kwargs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


# import torch
# x = torch.randn(1, 3, 224, 224)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from torchsummary import summary

# model = VGGNet(in_channels=3, num_classes=10).to(DEVICE)
# print(summary(model, (3,224,224)))
# temp = model(x)
# print(model)


# model2 = vggNet('vgg16', in_channels=3 ,num_classes=10, init_weights=True)
# print(summary(model2, (3,224,224)))
# temp2 = model2(x)
# print(model2)