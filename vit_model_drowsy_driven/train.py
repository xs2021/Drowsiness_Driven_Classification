import torch
import numpy as np
from tqdm import tqdm
from utils.my_dataset import get_loader
from models.vision_transformer import VisionTransformer
from utils.my_utils import set_seed, AverageMeter, WarmupCosineSchedule
import os


def valid(model, val_loader, device):
    # Validation!
    model.eval()
    all_preds, all_label = [], []
    pbar = tqdm(val_loader, desc='Validating', bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = (all_preds == all_label).mean()
    print('accuracy:', accuracy)

    return accuracy


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    config = {
        'seed': 42,
        'img_size': [224, 224], # 输入尺寸
        'num_classes': 4,       # 几分类
        'train_batch_size': 32,
        'val_batch_size': 16,
        'test_batch_size': 16,

        'learning_rate': 3e-2,
        'weight_decay': 0,    # 权重衰减
        'max_grad_norm': 1.0,

        'freeze_layers': False, # 是否冻结权重(不冻结就重头训练)

        'warmup_steps': 500,  # 学习率衰减
        'num_steps': 3550,    # 总步数
        'val_frequency': 71,  # 多少步后进行验证

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
    print(model)
    # pretrain
    # 方法1：使用strict=False，会加载所有匹配的权重，而忽略不匹配的权重（利用预训练模型的特征提取器部分，并完全重新训练分类头）
    # 从预训练权重中移除分类头的权重（如果存在）
    pretrained_weights = torch.load('./pretrain/vit_b_16_224.pth')
    pretrained_weights.pop('head.weight', None)
    pretrained_weights.pop('head.bias', None)
    model.load_state_dict(pretrained_weights, strict=False)
    print("----------------------")
    print(model)
    
    # freeze_weights
    if config['freeze_layers']:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=config['warmup_steps'], t_total=config['num_steps'])

    model.zero_grad()
    set_seed(config['seed'])
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        pbar = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            loss = model(x, y)

            loss.backward()

            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            pbar.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, config['num_steps'], losses.val)
            )
            if global_step % config['val_frequency'] == 0:
                accuracy = valid(model, val_loader, device)
                if best_acc < accuracy:
                    best_acc = accuracy
                    os.makedirs("./checkpoints", exist_ok=True)
                    torch.save(model.state_dict(), './checkpoints/vit_b_16_' + str(global_step) + '_' + str(best_acc) + '.pth')
                model.train()

            if global_step % config['num_steps'] == 0:
                break
        losses.reset()
        if global_step % config['num_steps'] == 0:
            break
