import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

# --- 1. 설정 ---
TASK1_NAME = 'MNIST' 
TASK2_NAME = 'RESISC45'
MERGED_MODEL_NAME = 'adamerged_final' # adamerged_final.pt

MODEL_NAME = 'ViT-B-32'
CHECKPOINT_BASE = './checkpoints' # AdaMerging/ 폴더 기준
DATA_LOCATION = './data'       # AdaMerging/ 폴더 기준

# 5개 태스크로 축소 (속도 향상)
TASKS_TO_EVAL = ['MNIST', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN']

# --- 2. 경로 및 인자 설정 ---
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

# --- 3. 모델 및 *방향 벡터* 로드 ---
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
finetuned1_checkpoint = os.path.join(args.save, TASK1_NAME, 'finetuned.pt')
finetuned2_checkpoint = os.path.join(args.save, TASK2_NAME, 'finetuned.pt')
merged_checkpoint = os.path.join(CHECKPOINT_BASE, f'{MERGED_MODEL_NAME}.pt')

print("Loading Models...")
pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu') # 전처리 함수 로드용
ft1_state_dict = torch.load(finetuned1_checkpoint, map_location='cpu').state_dict()
ft2_state_dict = torch.load(finetuned2_checkpoint, map_location='cpu').state_dict()
merged_state_dict = torch.load(merged_checkpoint, map_location='cpu').state_dict()

# 방향 벡터 R1, R2 계산 (Merged 모델 기준)
print("Calculating Direction Vectors (R1, R2)...")
R1_vector = {} # R1 = theta_ft1 - theta_merged
R2_vector = {} # R2 = theta_ft2 - theta_merged
for key in merged_state_dict:
    if key in ft1_state_dict:
        R1_vector[key] = ft1_state_dict[key] - merged_state_dict.get(key, 0)
    if key in ft2_state_dict:
        R2_vector[key] = ft2_state_dict[key] - merged_state_dict.get(key, 0)
print("Vectors loaded.")

# --- 4. 그리드 정의 (영역 확장) ---
alpha_range = np.arange(-0.5, 1.6, 0.2)
beta_range = np.arange(-0.5, 1.6, 0.2)
alphas, betas = np.meshgrid(alpha_range, beta_range)

loss_grids = {name: np.zeros_like(alphas) for name in TASKS_TO_EVAL}
loss_grids['Average_5_Tasks'] = np.zeros_like(alphas)

# --- 5. 데이터 로더 및 분류 헤드 준비 ---
print("Loading datasets and classification heads...")
dataloaders = {}
heads = {}
for name in TASKS_TO_EVAL:
    dataset = get_dataset(name, pretrained_model.val_preprocess, location=DATA_LOCATION, batch_size=128)
    dataloaders[name] = dataset.test_loader
    heads[name] = get_classification_head(args, name).to(device)

loss_fn = nn.CrossEntropyLoss()
print("Dataloaders and heads ready.")

# --- 6. 손실 계산 루프 ---
temp_encoder = ImageEncoder(args, keep_lang=False)
temp_encoder.eval()
temp_encoder.to(device)

pbar = tqdm(total=alphas.size, desc="Calculating Loss Grid (Merged-centric)")

with torch.no_grad():
    for i in range(alphas.shape[0]):
        for j in range(alphas.shape[1]):
            alpha = alphas[i, j]
            beta = betas[i, j]

            # 모델 보간: theta = theta_merged + alpha * R1 + beta * R2
            current_state_dict = {}
            for key in merged_state_dict:
                R1_vec = R1_vector.get(key, 0)
                R2_vec = R2_vector.get(key, 0)
                current_state_dict[key] = merged_state_dict[key] + alpha * R1_vec + beta * R2_vec

            temp_encoder.model.load_state_dict(current_state_dict, strict=False)

            total_avg_loss = 0.0
            
            # 5개 태스크 손실 계산
            for name in TASKS_TO_EVAL:
                dataloader = dataloaders[name]
                head = heads[name]
                total_loss = 0.0
                total_samples = 0
                for batch in dataloader:
                     images = batch[0].to(device) # 튜플 인덱스 사용
                     labels = batch[1].to(device) # 튜플 인덱스 사용
                     
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

# --- 7. 시각화 ---
print("Saving plots...")

def plot_loss_landscape(alphas, betas, loss_grid, title, task1_name, task2_name):
    plt.figure(figsize=(10, 8))
    log_loss = np.log(loss_grid)
    
    contour = plt.contourf(alphas, betas, log_loss, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Log(Loss)')
    plt.contour(alphas, betas, log_loss, levels=30, colors='k', linewidths=0.5)

    # 주요 지점 표시 (원점이 Merged 모델)
    plt.scatter([0], [0], marker='^', color='lime', s=150, zorder=5, label=f'AdaMerged (0, 0)')
    plt.scatter([1], [0], marker='*', color='blue', s=200, zorder=5, label=f'Finetuned {task1_name} (1, 0)')
    plt.scatter([0], [1], marker='P', color='magenta', s=150, zorder=5, label=f'Finetuned {task2_name} (0, 1)')
    
    min_idx = np.unravel_index(np.argmin(loss_grid, axis=None), loss_grid.shape)
    min_alpha = alphas[min_idx]
    min_beta = betas[min_idx]
    plt.scatter([min_alpha], [min_beta], marker='X', color='white', s=100, zorder=5, label=f'Optimal point ({min_alpha:.1f}, {min_beta:.1f})')

    plt.xlabel(f'Alpha (Direction to {task1_name})')
    plt.ylabel(f'Beta (Direction to {task2_name})')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.savefig(f'V2_Merged_Plot_{title.replace(" ", "_").replace("(", "").replace(")", "")}.png')
    plt.close()

for name in TASKS_TO_EVAL + ['Average_5_Tasks']:
    plot_loss_landscape(alphas, betas, loss_grids[name], f'Loss ({name})', TASK1_NAME, TASK2_NAME)

print("All V2 (Merged-centric) plots saved.")
