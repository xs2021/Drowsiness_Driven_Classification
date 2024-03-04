import torch
import torch.nn as nn

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, init_weights=False, aux_softmax=False):
        super(GoogLeNet, self).__init__()
        self.aux_softmax = aux_softmax
        self.conv1 = BasicConv2d(in_channels,64,kernel_size=7,padding=3,stride=2)
        self.maxpool1 = nn.MaxPool2d(3,stride=2,padding=1)
        self.conv2 = BasicConv2d(64,64,kernel_size=1)
        self.conv3 = BasicConv2d(64,192,kernel_size=3,padding=1)
        self.maxpool2 = nn.MaxPool2d(3,stride=2,padding=1)  # 28x28x192

        # in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj
        # 根据深度(通道)组合并行层 [总深度: ch1x1+ch3x3+ch5x5+pool_proj]
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # 输出通道：64+128+32+32 = 256
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # 输出通道：128+192+96+64 = 480
        self.maxpool3 = nn.MaxPool2d(3,stride=2,padding=1)   # 14x14x480

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1) # 7x7x832

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        # 输出通道：384+384+128+128 = 1024,  7x7x1024

        if self.training and self.aux_softmax:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        # 首次使用：自适应(全局)平均池化（可以将任意大小的输入特征图转换为固定大小的输出特征图）
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)  # [-1, 64, 112, 112]
        x = self.maxpool1(x) # [-1, 64, 56, 56]
        x = self.conv2(x)    # [-1, 64, 56, 56]
        x = self.conv3(x)    # [-1, 192, 56, 56]
        x = self.maxpool2(x) # [-1, 192, 28, 28]
        # inception架构不改变卷积核的大小,大小是由里面的最大池化改变
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux_softmax:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_softmax:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)  # [-1, 1024, 7, 7]

        x = self.adaptive_avgpool(x)  # [-1, 1024, 1, 1]
        x = self.dropout(x)   # [-1, 1024, 1, 1]
        # 已经使用了adaptive_avgpool, 还需要展平吗？  ---需要
        x = nn.Flatten()(x)   # [-1, 1024*1*1]
        x = self.fc(x)        # [-1, 10]
        if self.training and self.aux_softmax:  # 保证只在训练模式有效
            return x, aux1, aux2
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Inception 结构: 输出的宽高一致才能进行拼接
# ch1x1: 表示使用1x1卷积核的个数(通道数)
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x) # 输出通道: ch1x1
        branch2 = self.branch2(x) # 输出通道: ch3x3
        branch3 = self.branch3(x) # 输出通道: ch5x5
        branch4 = self.branch4(x) # 输出通道: pool_proj
        # 根据深度(通道)组合并行层 [总深度: ch1x1+ch3x3+ch5x5+pool_proj]
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)  # dim=1表示通道

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(5,3)
        self.conv_k_eq_s = BasicConv2d(in_channels,128,kernel_size=1) # output[batch, 128, 4, 4]
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),  # 这样子写就行, 验证或测试时使用model.eval(), 就可以确保在评估时不会应用Dropout
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        # aux1: Batch x 512 x 14 x 14, aux2: Batch x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: Batch x 512 x 4 x 4, aux2: Batch x 528 x 4 x 4
        x = self.conv_k_eq_s(x)
        # aux1: Batch x 128 x 4 x 4, aux2: Batch x 128 x 4 x 4
        x = nn.Flatten()(x)
        # aux1: Batch x 1024, aux2: Batch x 1024
        return self.classifier(x)

# 通用的结构
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.basic_features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basic_features(x)

# import torch
# x = torch.randn(1, 3, 224, 224)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from torchsummary import summary

# model = GoogLeNet(in_channels=3, num_classes=10, init_weights=True, aux_softmax=True).to(DEVICE)
# print(summary(model, (3,224,224)))
# temp = model(x)
# print(model)

# model = torchvision.models.googlenet(init_weights=True, weights=None).to(DEVICE)
# summary(model, (3, 224, 224))
# print(model)
