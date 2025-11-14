import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from sklearn.decomposition import PCA
import copy

# --- 1. 설정 ---
MODEL_NAME = 'ViT-B-32'
CHECKPOINT_BASE = './checkpoints' # AdaMerging/src 폴더 기준
DATA_LOCATION = './data'       # AdaMerging/src 폴더 기준

# 시각화 및 평가에 사용할 태스크 목록 (main_...py 파일과 동일하게)
EXAM_DATASETS = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

# --- 2. 경로 및 인자 설정 ---
from task_vectors import TaskVector
from modeling import ImageEncoder, ClassificationHead
from args import parse_arguments
from datasets.registry import get_dataset
from heads import get_classification_head
# TIES 유틸리티 임포트 (벡터 <-> state_dict 변환용)
from ties_merging_utils import state_dict_to_vector, vector_to_state_dict

args = parse_arguments()
args.model = MODEL_NAME
args.data_location = DATA_LOCATION
args.save = os.path.join(CHECKPOINT_BASE, MODEL_NAME)
args.openclip_cachedir = './openclip_cache' 

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. 모델 및 Task Vector 로드 ---
print("Loading Pretrained Model and Finetuned Task Vectors...")
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
# torch.load(..., weights_only=True) 경고를 무시하고 진행 (PyTorch 2.x+)
# 이 프로젝트의 .pt 파일은 state_dict가 아닌 전체 모델 객체를 저장하므로 weights_only=False가 맞습니다.
pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu')
pretrained_state_dict = pretrained_model.state_dict()

task_vectors = []
task_names = []
for name in EXAM_DATASETS:
    finetuned_checkpoint = os.path.join(args.save, name, 'finetuned.pt')
    if os.path.exists(finetuned_checkpoint):
        task_vectors.append(TaskVector(pretrained_checkpoint, finetuned_checkpoint))
        task_names.append(name)
    else:
        print(f"Warning: Checkpoint not found for {name}, skipping.")

print(f"Loaded {len(task_vectors)} task vectors: {task_names}")

# --- 4. PCA 수행 ---
print("Performing PCA on task vectors...")
flat_tvs = []
for tv in task_vectors:
    flat_tvs.append(state_dict_to_vector(tv.vector, remove_keys=[]))
X = torch.stack(flat_tvs).cpu().numpy()
pca = PCA(n_components=2)
pca.fit(X)
projected_coords = pca.transform(X)
pc1_flat = pca.components_[0]
pc2_flat = pca.components_[1]
ref_state_dict = task_vectors[0].vector 
dir1_vector_dict = vector_to_state_dict(torch.from_numpy(pc1_flat), ref_state_dict, remove_keys=[])
dir2_vector_dict = vector_to_state_dict(torch.from_numpy(pc2_flat), ref_state_dict, remove_keys=[])
print("PCA complete. PC1 and PC2 directions computed.")

# --- 5. 그리드 정의 (영역 확장) ---
min_alpha = min(projected_coords[:, 0].min(), 0) - 0.5
max_alpha = max(projected_coords[:, 0].max(), 0) + 0.5
min_beta = min(projected_coords[:, 1].min(), 0) - 0.5
max_beta = max(projected_coords[:, 1].max(), 0) + 0.5
alpha_range = np.linspace(min_alpha, max_alpha, 7) # 7x7 grid로 속도 향상
beta_range = np.linspace(min_beta, max_beta, 7)
alphas, betas = np.meshgrid(alpha_range, beta_range)
loss_grids = {name: np.zeros_like(alphas) for name in task_names}
loss_grids['Average_Loss'] = np.zeros_like(alphas)

# --- 6. 데이터 로더, 분류 헤드, *배치 캐싱* 준비 ---
print("Loading datasets, classification heads, and caching batches...")
dataloaders = {}
heads = {}
cached_batches = {} 
loss_fn = nn.CrossEntropyLoss()

for name in task_names:
    dataset = get_dataset(name, pretrained_model.val_preprocess, location=DATA_LOCATION, batch_size=128)
    dataloader = dataset.test_loader
    dataloaders[name] = dataloader
    heads[name] = get_classification_head(args, name).to(device)
    
    cached_batches[name] = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3: # 3 배치만 캐싱하여 속도 향상
            break
        
        images = None
        labels = None
        if isinstance(batch, dict):
            images = batch['images'].cpu()
            labels = batch['labels'].cpu()
        else:
            images = batch[0].cpu()
            labels = batch[1].cpu()

        cached_batches[name].append({'images': images, 'labels': labels})

print(f"Dataloaders, heads, and {len(task_names) * 3} batches cached.")

# --- 7. 손실 계산 루프 (캐시된 배치 사용) ---
temp_encoder = ImageEncoder(args, keep_lang=False)
temp_encoder.eval()

pbar = tqdm(total=alphas.size, desc="Calculating Loss Grid (PCA-centric)")

with torch.no_grad():
    for i in range(alphas.shape[0]):
        for j in range(alphas.shape[1]):
            alpha = alphas[i, j]
            beta = betas[i, j]

            # 모델 보간 (CPU에서 수행)
            current_state_dict = {}
            for key in pretrained_state_dict:
                dir1_vec = dir1_vector_dict.get(key, 0)
                dir2_vec = dir2_vector_dict.get(key, 0)
                current_state_dict[key] = pretrained_state_dict[key] + alpha * dir1_vec + beta * dir2_vec
                
            stripped_state_dict = {}
            for key, value in current_state_dict.items():
                if key.startswith("model."):
                    stripped_state_dict[key.replace("model.", "", 1)] = value
                else:
                    stripped_state_dict[key] = value # 혹시 모를 예외 처리

            # CPU 모델의 '내부' 모델(temp_encoder.model)에 접두사가 제거된 가중치를 로드합니다.
            try:
                temp_encoder.model.load_state_dict(stripped_state_dict, strict=False)
            except Exception as e:
                print(f"Error loading state dict at step (i={i}, j={j}): {e}")
                print(f"Stripped keys: {list(stripped_state_dict.keys())[:5]}")
                print(f"Model keys: {list(temp_encoder.model.state_dict().keys())[:5]}")
                raise e
            
            # 모델을 GPU로 이동
            temp_encoder.to(device)

            total_avg_loss = 0.0
            
            # N개 태스크 손실 계산
            for name in task_names:
                head = heads[name] # 이미 GPU에 있음
                total_loss = 0.0
                total_samples = 0
                
                for batch in cached_batches[name]:
                     images = batch['images'].to(device)
                     labels = batch['labels'].to(device)
                     
                     features = temp_encoder(images)
                     logits = head(features)
                     loss = loss_fn(logits, labels)
                     
                     total_loss += loss.item() * images.size(0)
                     total_samples += images.size(0)
                
                if total_samples == 0:
                    continue

                calculated_loss = total_loss / total_samples
                loss_grids[name][i, j] = calculated_loss
                total_avg_loss += calculated_loss

            if len(task_names) > 0:
                loss_grids['Average_Loss'][i, j] = total_avg_loss / len(task_names)
            
            temp_encoder.to('cpu') 
            pbar.update(1)
            
pbar.close()

# --- 8. 시각화 ---
print("Saving plots...")

def plot_loss_landscape(alphas, betas, loss_grid, title, projected_coords, task_names):
    plt.figure(figsize=(12, 10))
    safe_log_loss = np.log(np.maximum(loss_grid, 1e-9))
    
    contour = plt.contourf(alphas, betas, safe_log_loss, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Log(Loss)')
    plt.contour(alphas, betas, safe_log_loss, levels=30, colors='k', linewidths=0.5)

    plt.scatter([0], [0], marker='o', color='red', s=200, zorder=5, label='Pre-trained (0, 0)')
    
    colors = plt.cm.jet(np.linspace(0, 1, len(task_names)))
    for i, name in enumerate(task_names):
        alpha_i = projected_coords[i, 0]
        beta_i = projected_coords[i, 1]
        plt.scatter([alpha_i], [beta_i], marker='*', color=colors[i], s=250, zorder=5, label=f'Finetuned {name} ({alpha_i:.1f}, {beta_i:.1f})')
        plt.text(alpha_i + 0.05, beta_i + 0.05, name, fontsize=9, color='white')

    min_idx = np.unravel_index(np.argmin(loss_grid, axis=None), loss_grid.shape)
    min_alpha = alphas[min_idx]
    min_beta = betas[min_idx]
    plt.scatter([min_alpha], [min_beta], marker='X', color='white', s=150, zorder=5, label=f'Optimal point ({min_alpha:.1f}, {min_beta:.1f})')

    plt.xlabel('Principal Component 1 (alpha)')
    plt.ylabel('Principal Component 2 (beta)')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.savefig(f'PCA_Plot_{title.replace(" ", "_").replace("(", "").replace(")", "")}.png', bbox_inches='tight')
    plt.close()

for name in task_names + ['Average_Loss']:
    plot_loss_landscape(alphas, betas, loss_grids[name], f'Loss Landscape ({name} Loss)', projected_coords, task_names)

print("All PCA loss landscape plots saved.")