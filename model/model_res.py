import torch.nn as nn

# 浅层的 resnet: 18/34
class ResidualBlock(nn.Module):
    # skip_branch_alter=True：表示使用了1x1的卷积对通道或者图片尺寸进行改变
    def __init__(self, in_ch, out_ch, stride, skip_branch_alter=False) -> None:
        super(ResidualBlock, self).__init__()
        # 主分支第一个卷积步长不同
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_ch,out_channels=out_ch, kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # 1x1conv: 对通道升降维或者改变图片尺寸
        if skip_branch_alter: 
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.conv1x1 = None

    def forward(self, x):
        input = x
        if self.conv1x1:  # 使用了跳跃分支的卷积
            input = self.conv1x1(x)
        # 主分支   
        temp_out = self.relu(self.bn1(self.conv1(x)))
        temp_out = self.bn2(self.conv2(temp_out))
        
        final_out = self.relu(temp_out+input)
        return final_out

# 深层的 resnet: 50/101/152
class DeepResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, skip_branch_alter=False) -> None:
        super(DeepResidualBlock, self).__init__()
        self.expansion = 4

        # 主分支第二个卷积步长不同, 用来控制尺寸
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,kernel_size=1, stride=1, padding=0)  # 不变: maintain channels
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch*self.expansion,kernel_size=1, stride=1, padding=0)  # 升维: unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_ch*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # 1x1conv: 对通道升降维 且 改变图片尺寸
        if skip_branch_alter: 
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch,out_channels=out_ch*self.expansion,kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(out_ch*self.expansion),
            )
        else:
            self.conv1x1 = None

    def forward(self,x):
        input = x
        if self.conv1x1:
            input = self.conv1x1(x)
        temp_out = self.relu(self.bn1(self.conv1(x)))  # 维度不变
        temp_out = self.relu(self.bn2(self.conv2(temp_out))) # 改变尺寸
        temp_out = self.bn3(self.conv3(temp_out))  # 升维
        final_out = self.relu(temp_out+input)
        return final_out

class ResNet(nn.Module):
    def __init__(self, in_ch, res1_num, res2_num, res3_num, res4_num, layers, num_class=10) -> None:
        super(ResNet, self).__init__()
        self.layers = layers
        self.start = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=64,kernel_size=7,stride=2,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        # layers: 18,34 
        self.residual_block1 = self.buildResidualBlock(64, 64, 1, False, res1_num)
        self.residual_block2 = self.buildResidualBlock(64, 128, 2, True, res2_num)
        self.residual_block3 = self.buildResidualBlock(128, 256, 2, True, res3_num)
        self.residual_block4 = self.buildResidualBlock(256, 512, 2, True, res4_num)

        # layers: 50,101,152
        self.deep_residual_block1 = self.buildDeepResidualBlock(64, 64, 1, True, res1_num)
        self.deep_residual_block2 = self.buildDeepResidualBlock(256, 128, 2, True, res2_num)
        self.deep_residual_block3 = self.buildDeepResidualBlock(512, 256, 2, True, res3_num)
        self.deep_residual_block4 = self.buildDeepResidualBlock(1024, 512, 2, True, res4_num)

        self.adapt_average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*1*1, num_class)
        self.fc2 = nn.Linear(2048*1*1, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,x):
        x = self.start(x)  # [1, 64, 56, 56]
        if self.layers<50: # 50层以下
            x = self.residual_block1(x)
            x = self.residual_block2(x)
            x = self.residual_block3(x)
            x = self.residual_block4(x)
        else:
            # 50 101 152
            x = self.deep_residual_block1(x) # [1, 256, 56, 56]
            x = self.deep_residual_block2(x)
            x = self.deep_residual_block3(x)
            x = self.deep_residual_block4(x) # [1, 2048, 7, 7]
        x = self.adapt_average_pool(x)   # [1, 2048, 1, 1]
        x = x.view(x.size(0), -1)
        if self.layers<50:
            x = self.fc1(x)
        else:
            x = self.fc2(x)
        return x
    

    # classmethod表示是类方法, num 表示残差块的数量
    @classmethod
    def buildResidualBlock(self, in_ch, out_ch, stride, skip_branch_alter, num):
        layer_blocks = []
        next_in_ch = out_ch
        skip_branch_alter = skip_branch_alter
        for i in range(num):
            if stride != 1 and i==0 :  # 控制每一个残差结构中第一个残差块
                layer_blocks.append(ResidualBlock(in_ch=in_ch, out_ch=out_ch, stride=stride, skip_branch_alter=skip_branch_alter))
                next_in_ch = 2*in_ch  # 表示使用了1x1卷积进行变动, 保证下一次的输入与指定的输出通道一致
                skip_branch_alter = False
            else: # 输入与指定的输出通道一致
                layer_blocks.append(ResidualBlock(in_ch=next_in_ch, out_ch=out_ch, stride=1, skip_branch_alter=skip_branch_alter))

        return nn.Sequential(*layer_blocks)

    # classmethod表示是类方法, num 表示残差块的数量
    @classmethod
    def buildDeepResidualBlock(self, in_ch, out_ch, stride, skip_branch_alter, num):
        layer_blocks = []
        next_in_ch = in_ch
        skip_branch_alter = skip_branch_alter
        for i in range(num):
            if stride != 1 and i==0 :  # 控制每一个残差结构中第一个残差块
                layer_blocks.append(DeepResidualBlock(in_ch=next_in_ch, out_ch=out_ch, stride=stride, skip_branch_alter=skip_branch_alter))
                next_in_ch = 4*out_ch
                skip_branch_alter = False
            else:  # 输入与指定的输出通道一致
                layer_blocks.append(DeepResidualBlock(in_ch=next_in_ch, out_ch=out_ch, stride=1, skip_branch_alter=skip_branch_alter))
                next_in_ch = 4*out_ch
                skip_branch_alter = False

        return nn.Sequential(*layer_blocks)

def resnet18(num_classes=10):
    model = ResNet(in_ch=3, res1_num=2, res2_num=2, res3_num=2, res4_num=2, layers=18, num_class=num_classes)
    return model

def resnet34(num_classes=10):
    model = ResNet(in_ch=3, res1_num=3, res2_num=4, res3_num=6, res4_num=3, layers=34, num_class=num_classes)
    return model

def resnet50(num_classes=10):
    model = ResNet(in_ch=3, res1_num=3, res2_num=4, res3_num=6, res4_num=3, layers=50, num_class=num_classes)
    return model

def resnet65(num_classes=10):
    model = ResNet(in_ch=3, res1_num=3, res2_num=4, res3_num=11, res4_num=3, layers=66, num_class=num_classes)
    return model    

def resnet101(num_classes=10):
    model = ResNet(in_ch=3, res1_num=3, res2_num=4, res3_num=23, res4_num=3, layers=101, num_class=num_classes)
    return model

def resnet152(num_classes=10):
    model = ResNet(in_ch=3, res1_num=3, res2_num=8, res3_num=36, res4_num=3, layers=152, num_class=num_classes)
    return model

# import torch
# from torchsummary import summary
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 输入大小超过32x32即可
# input1 = torch.randn(1,3,224,224).to(DEVICE)
# input2 = torch.randn(1,3,64,64).to(DEVICE)
# num_class = 4
# model = resnet18(num_class).to(DEVICE)
# print(model)
# output = model(input1)
# print(summary(model,(3,224,224)))

# import torchvision
# model2 = torchvision.models.resnet50().to(device=DEVICE)
# print(model2)
# print(summary(model2,(3,224,224)))