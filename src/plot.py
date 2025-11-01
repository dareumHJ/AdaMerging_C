import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

# --- 1. 설정 (사용자 지정 필요) ---
# X축으로 사용할 Task 1
TASK1_NAME = 'MNIST' 
# Y축으로 사용할 병합 모델
MERGED_MODEL_NAME = 'adamerged_final'

MODEL_NAME = 'ViT-B-32'
CHECKPOINT_BASE = './checkpoints' # AdaMerging 폴더 기준 상대 경로
DATA_LOCATION = './data'       # AdaMerging 폴더 기준 상대 경로

# 실험할 8개 태스크 목록 (main_...py 파일과 동일하게)
EXAM_DATASETS = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

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
args.openclip_cachedir = './openclip_cache' # 캐시 폴더 지정

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. 모델 및 Task Vector 로드 ---
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
finetuned1_checkpoint = os.path.join(args.save, TASK1_NAME, 'finetuned.pt')
merged_checkpoint = os.path.join(args.save,  f'{MERGED_MODEL_NAME}.pt') # .pt 파일 직접 지정

print("Loading Task Vectors...")
tv1 = TaskVector(pretrained_checkpoint, finetuned1_checkpoint)
tv_merged = TaskVector(pretrained_checkpoint, merged_checkpoint)
pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu')
pretrained_state_dict = pretrained_model.state_dict()
print("Task Vectors loaded.")

# --- 4. 그리드 정의 ---
alpha_range = np.arange(-0.5, 1.6, 0.2) # 범위를 조금 넓게, 간격을 0.2로 (속도 향상)
beta_range = np.arange(-0.5, 1.6, 0.2)
alphas, betas = np.meshgrid(alpha_range, beta_range)

# 8개 태스크 + 평균 손실을 저장할 딕셔너리
loss_grids = {name: np.zeros_like(alphas) for name in EXAM_DATASETS}
loss_grids['Average'] = np.zeros_like(alphas)

# --- 5. 데이터 로더 및 분류 헤드 준비 ---
print("Loading datasets and classification heads...")
dataloaders = {}
heads = {}
for name in EXAM_DATASETS:
    dataset = get_dataset(name, pretrained_model.val_preprocess, location=DATA_LOCATION, batch_size=128)
    dataloaders[name] = dataset.test_loader
    heads[name] = get_classification_head(args, name).to(device)

loss_fn = nn.CrossEntropyLoss()
print("Dataloaders and heads ready.")

# --- 6. 손실 계산 루프 ---
temp_encoder = ImageEncoder(args, keep_lang=False)
temp_encoder.eval()
temp_encoder.to(device)

pbar = tqdm(total=alphas.size, desc="Calculating Loss Grid")

with torch.no_grad():
    for i in range(alphas.shape[0]):
        for j in range(alphas.shape[1]):
            alpha = alphas[i, j]
            beta = betas[i, j]

            # 모델 보간
            current_state_dict = {}
            for key in pretrained_state_dict:
                # tv_merged.vector에 키가 있는지 확인
                tv1_vec = tv1.vector.get(key, 0)
                tv_merged_vec = tv_merged.vector.get(key, 0)
                
                current_state_dict[key] = pretrained_state_dict[key] + alpha * tv1_vec + beta * tv_merged_vec

            temp_encoder.model.load_state_dict(current_state_dict, strict=False)

            total_avg_loss = 0.0
            
            # 8개 태스크 손실 모두 계산
            for name in EXAM_DATASETS:
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

            loss_grids['Average'][i, j] = total_avg_loss / len(EXAM_DATASETS)
            pbar.update(1)
pbar.close()

# --- 7. 시각화 ---
print("Saving plots...")

def plot_loss_landscape(alphas, betas, loss_grid, title, task1_name, merged_name):
    plt.figure(figsize=(10, 8))
    
    # 로그 스케일로 변환 (손실 차이가 클 경우 대비)
    log_loss = np.log(loss_grid)
    
    contour = plt.contourf(alphas, betas, log_loss, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Log(Loss)')
    plt.contour(alphas, betas, log_loss, levels=30, colors='k', linewidths=0.5)

    # 주요 지점 표시
    plt.scatter([0], [0], marker='o', color='red', s=150, zorder=5, label='Pre-trained (0, 0)')
    plt.scatter([1], [0], marker='*', color='blue', s=200, zorder=5, label=f'Fine-tuned {task1_name} (1, 0)')
    plt.scatter([0], [1], marker='^', color='lime', s=150, zorder=5, label=f'{merged_name} (0, 1)')
    
    # 최적점 표시 (해당 손실 그리드에서 손실이 가장 낮은 지점)
    min_idx = np.unravel_index(np.argmin(loss_grid, axis=None), loss_grid.shape)
    min_alpha = alphas[min_idx]
    min_beta = betas[min_idx]
    plt.scatter([min_alpha], [min_beta], marker='X', color='white', s=100, zorder=5, label=f'Optimal point ({min_alpha:.1f}, {min_beta:.1f})')

    plt.xlabel(f'Alpha (Direction of {task1_name} Vector)')
    plt.ylabel(f'Beta (Direction of {merged_name} Vector)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.savefig(f'Loss_Landscape_{title.replace(" ", "_").replace("(", "").replace(")", "")}.png')
    plt.close() # 메모리 해제

# 9개의 플롯 생성
for name in EXAM_DATASETS + ['Average']:
    plot_loss_landscape(alphas, betas, loss_grids[name], f'Loss Landscape ({name} Loss)', TASK1_NAME, MERGED_MODEL_NAME)

print("All loss landscape plots saved.")