import torch
import numpy as np
from tqdm import tqdm
from utils.my_dataset import get_loader
from models.vision_transformer import VisionTransformer
from utils.my_utils import set_seed, AverageMeter, WarmupCosineSchedule
import os

def test(model, test_loader, device):
    # test!
    model.eval()
    all_preds, all_label = [], []
    pbar = tqdm(test_loader, desc='testing', bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
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
    model.to(device)

    # 加载模型并测试
    model.load_state_dict(torch.load('./checkpoints/vit_b_16_3195_0.9858906525573192.pth')) 
    test(model, test_loader, device)
