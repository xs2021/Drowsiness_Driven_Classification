# Drowsiness_Driven_Classification
疲劳驾驶分类(基于Pytorch)

# 说明

> **文件：**
>
> - gain_mean_std.py：获取均值和标准差
> - pre_processing.txt：预处理文件（内置均值和标准差）
> - this.py：训练、验证模型的文件
> - predict.py：输入图片进行预测得到对应类别
> - my_utils.py：自定义的工具包文件(绘图、模型微调等)
> - requirements.txt：依赖的环境包
>
> **文件夹：**
>
> - data：使用的数据集
> - model：自定义的本地模型(VGG、ResNet、GooLeNet)
> - weights：训练好的模型权重
> - results：模型训练的结果可视化
> - pic_predict：用来预测的图片
> - vit_model_drowsy_driven：使用 vit 预训练模型进行训练
>
> **数据集：**
>
> - **训练集：**共2836张（`80%用于训练，20%用于验证`）
>
> - **测试集：**共101张
>
> **标签：**
>
> - 'closed'（闭眼）: 0
> - 'no_yawn'（没有打哈欠）: 1
> - 'open'（睁眼）: 2
> - 'yawn'（打哈欠）: 3

# 模型

> - 整合包：https://pan.quark.cn/s/70536c418824
> - 数据集：https://pan.quark.cn/s/5a4b1140d306
> - VGG16：https://pan.quark.cn/s/46a1d3d1f29b
> - VGG19：https://pan.quark.cn/s/c13604666d41
> - ResNet65：https://pan.quark.cn/s/24ec94d0676a
> - VGG16_Pre：https://pan.quark.cn/s/11a66e0e5d4d
> - ResNet50_Pre：https://pan.quark.cn/s/33c05f29d7a1
> - GoogLeNet_Pre：https://pan.quark.cn/s/e1c1ab0fd170
> - VIT16 预训练模型（放到 pretrain 文件夹内）：https://pan.quark.cn/s/cf0516956931
> - VIT16 训练好的模型（放到 checkpoints 文件夹内）：https://pan.quark.cn/s/703da7c6d738

# 运行

**数据集：**将数据集下载好解压后放到 `data` 文件夹内

**VGG / ResNet65 / GoogLeNet 运行方式：**

将训练好的模型文件下载好放到 `weights` 文件夹内

1、下载所需的环境包

```shell
pip install -r requirements.txt
```

2、进入项目所在路径，运行训练脚本

```shell
python this.py
```

3、运行预测脚本

```shell
python predict.py
```

**VIT16 运行方式：** 

将预训练模型文件下载好放到 `pretrain` 文件夹内

将训练好的模型文件下载好放到 `checkpoints` 文件夹内

配置好环境后，进入项目所在路径，依次执行：

```shell
python train.py
python test.py
python predict.py
```
全部内容请查看README的PDF文件
