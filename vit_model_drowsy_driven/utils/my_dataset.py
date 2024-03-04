from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from torchvision import datasets, transforms

# 数据集 CIFAR10
# def get_loader(config):
#     transform_train = transforms.Compose([
#         transforms.RandomResizedCrop(config['img_size'], scale=(0.05, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
#     transform_test = transforms.Compose([
#         transforms.Resize(config['img_size']),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])

#     trainset = datasets.CIFAR10(root="./data", train=True, transform=transform_train)
#     testset = datasets.CIFAR10(root="./data", train=False, transform=transform_test)

#     train_sampler = RandomSampler(trainset)
#     test_sampler = SequentialSampler(testset)

#     train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=config['train_batch_size'], num_workers=4, pin_memory=True)
#     test_loader = DataLoader(testset, sampler=test_sampler, batch_size=config['test_batch_size'], num_workers=4, pin_memory=True)

#     return train_loader, test_loader

# 数据集：疲劳驾驶
def get_loader(config):
    BATCH_SIZE = 32
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['img_size'], scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.151, 0.136, 0.129], std=[0.058, 0.051, 0.048]) # 需自己计算
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.151, 0.136, 0.129], std=[0.058, 0.051, 0.048]),
    ])

    train_set = 0.0
    test_set = 0.0
    # 加载自定义数据集
    ROOT_TRAIN = "../data/Drowsiness_Driven_Dataset/train/"
    ROOT_TEST = "../data/Drowsiness_Driven_Dataset/test/"
    print(ROOT_TRAIN)
    # ImageFolder 会根据文件夹自动生成不同的标签
    try:
        train_set = datasets.ImageFolder(root=ROOT_TRAIN, transform=transform_train)
        test_set = datasets.ImageFolder(root=ROOT_TEST, transform=transform_test)
        print(train_set.class_to_idx)
        print(test_set.class_to_idx)
    except Exception as e:
        print(f"发生异常: {e}")

    # 划分训练集和验证集
    train_set, val_set = random_split(train_set, [int(round(0.8 * len(train_set))), int(round(0.2 * len(train_set)))])
    train_loader = DataLoader(train_set, batch_size=config['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['val_batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config['test_batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader