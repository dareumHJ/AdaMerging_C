import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

# --- 1. 설정 ---
TASK1_NAME = 'MNIST' 
TASK2_NAME = 'RESISC45'
MODEL_NAME = 'ViT-B-32'

CHECKPOINT_BASE = './checkpoints'
DATA_LOCATION = './data'

TASKS_TO_EVAL = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

from task_vectors import TaskVector
from modeling import ImageEncoder, ClassificationHead
from args import parse_arguments
from datasets.registry import get_dataset
from heads import get_classification_head

args = parse_arguments()
args.model = MODEL_NAME
args.data_location = DATA_LOCATION
args.save = os.path.join(CHECKPOINT_BASE, MODEL_NAME)
args.openclip_cachedir = './openclip_cache' 

device = "cuda" if torch.cuda.is_available() else "cpu"

pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
finetuned1_checkpoint = os.path.join(args.save, TASK1_NAME, 'finetuned.pt')
finetuned2_checkpoint = os.path.join(args.save, TASK2_NAME, 'finetuned.pt')

print("Loading Task Vectors (T1, T2)...")
# T_1 = theta_ft1 - theta_pre
tv1 = TaskVector(pretrained_checkpoint, finetuned1_checkpoint) 
# T_2 = theta_ft2 - theta_pre
tv2 = TaskVector(pretrained_checkpoint, finetuned2_checkpoint) 
pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu')

print("Moving weights to GPU...")
pretrained_state_dict = {k: v.to(device) for k, v in pretrained_model.state_dict().items()}
tv1_gpu = {k: v.to(device) for k, v in tv1.vector.items()}
tv2_gpu = {k: v.to(device) for k, v in tv2.vector.items()}
print("Task Vectors loaded.")

alpha_range = np.arange(-3.0, 3.0, 0.5)
beta_range = np.arange(-3.0, 3.0, 0.5)
alphas, betas = np.meshgrid(alpha_range, beta_range)

loss_grids = {name: np.zeros_like(alphas) for name in TASKS_TO_EVAL}
loss_grids['Average_5_Tasks'] = np.zeros_like(alphas)

# --- 5. 데이터 로더 및 분류 헤드 준비 ---
print("Loading datasets and classification heads...")
dataloaders = {}
heads = {}
for name in TASKS_TO_EVAL:
    dataset = get_dataset(name, pretrained_model.val_preprocess, location=DATA_LOCATION, batch_size=256, num_workers=8)
    dataloaders[name] = dataset.test_loader
    heads[name] = get_classification_head(args, name).to(device)

loss_fn = nn.CrossEntropyLoss()
print("Dataloaders and heads ready.")

temp_encoder = ImageEncoder(args, keep_lang=False)

if device == "cuda" and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for loss calculation.")
    temp_encoder = nn.DataParallel(temp_encoder)
    
temp_encoder.eval()
temp_encoder.to(device)

pbar = tqdm(total=alphas.size, desc="Calculating Loss Grid (Pretrained-centric)")

with torch.no_grad():
    for i in range(alphas.shape[0]):
        for j in range(alphas.shape[1]):
            alpha = alphas[i, j]
            beta = betas[i, j]

            # model interpolation: theta = theta_pre + alpha * T1 + beta * T2
            current_state_dict = {}
            for key in pretrained_state_dict:
                tv1_vec = tv1_gpu.get(key, 0)
                tv2_vec = tv2_gpu.get(key, 0)
                current_state_dict[key] = pretrained_state_dict[key] + alpha * tv1_vec + beta * tv2_vec

            # temp_encoder.model.load_state_dict(current_state_dict, strict=False)
            if isinstance(temp_encoder, nn.DataParallel):
                # temp_encoder.module.model.load_state_dict(current_state_dict, strict=False)
                temp_encoder.module.load_state_dict(current_state_dict, strict=False)
            else:
                # temp_encoder.model.load_state_dict(current_state_dict, strict=False)
                temp_encoder.load_state_dict(current_state_dict, strict=False)

            total_avg_loss = 0.0
            
            # calculate task losses
            for name in TASKS_TO_EVAL:
                dataloader = dataloaders[name]
                head = heads[name]
                total_loss = 0.0
                total_samples = 0
                for batch in dataloader:
                     images = batch[0].to(device)
                     labels = batch[1].to(device)
                     
                     features = temp_encoder(images)
                     logits = head(features)
                     loss = loss_fn(logits, labels)
                     total_loss += loss.item() * images.size(0)
                     total_samples += images.size(0)
                
                calculated_loss = total_loss / total_samples
                loss_grids[name][i, j] = calculated_loss
                total_avg_loss += calculated_loss

            loss_grids['Average_5_Tasks'][i, j] = total_avg_loss / len(TASKS_TO_EVAL)
            pbar.update(1)
pbar.close()

print("Saving plots...")

def plot_loss_landscape(alphas, betas, loss_grid, title, task1_name, task2_name):
    plt.figure(figsize=(10, 8))
    log_loss = np.log(loss_grid)
    
    contour = plt.contourf(alphas, betas, log_loss, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Log(Loss)')
    plt.contour(alphas, betas, log_loss, levels=30, colors='k', linewidths=0.5)

    # 주요 지점 표시
    plt.scatter([0], [0], marker='o', color='red', s=150, zorder=5, label='Pre-trained (0, 0)')
    plt.scatter([1], [0], marker='*', color='blue', s=200, zorder=5, label=f'Finetuned {task1_name} (1, 0)')
    plt.scatter([0], [1], marker='P', color='magenta', s=150, zorder=5, label=f'Finetuned {task2_name} (0, 1)')
    
    min_idx = np.unravel_index(np.argmin(loss_grid, axis=None), loss_grid.shape)
    min_alpha = alphas[min_idx]
    min_beta = betas[min_idx]
    plt.scatter([min_alpha], [min_beta], marker='X', color='white', s=100, zorder=5, label=f'Optimal point ({min_alpha:.1f}, {min_beta:.1f})')

    plt.xlabel(f'Alpha (Direction of {task1_name} Vector)')
    plt.ylabel(f'Beta (Direction of {task2_name} Vector)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.savefig(f'V1_Pretrained_Plot_{title.replace(" ", "_").replace("(", "").replace(")", "")}.png')
    plt.close()

for name in TASKS_TO_EVAL + ['Average_5_Tasks']:
    plot_loss_landscape(alphas, betas, loss_grids[name], f'Loss ({name})', TASK1_NAME, TASK2_NAME)

print("All V1 (Pretrained-centric) plots saved.")
