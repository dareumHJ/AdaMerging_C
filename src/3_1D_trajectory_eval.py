import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import copy

# --- 0. Add 'src' directory to Python path ---
script_dir = os.getcwd()
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- 1. 설정 ---
TASK1_NAME = 'Cars'
TASK2_NAME = 'SUN397'
ALL_TASK_NAME = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
# OMEGA_1, OMEGA_2는 가중 평균 손실 계산에만 사용
OMEGA_1 = 0.5
OMEGA_2 = 0.5

MODEL_NAME = 'ViT-B-32'
CHECKPOINT_BASE = './checkpoints' # 스크립트 실행 위치 기준
DATA_LOCATION = './data'       # 스크립트 실행 위치 기준

# --- 2. 경로 및 인자 설정 ---
try:
    from task_vectors import TaskVector
    from modeling import ImageEncoder, ClassificationHead
    from args import parse_arguments
    from datasets.registry import get_dataset
    from heads import get_classification_head
except ImportError as e:
    print(f"Error importing modules: {e}")
    exit(1)

args = parse_arguments()
args.model = MODEL_NAME
args.data_location = DATA_LOCATION
args.save = os.path.join(CHECKPOINT_BASE, MODEL_NAME)
args.openclip_cachedir = './openclip_cache' 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 3. 모델 및 Task Vector 로드 ---
print(f"Loading vectors for {TASK1_NAME} (t=0) and {TASK2_NAME} (t=1)...")
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
finetuned1_checkpoint = os.path.join(args.save, TASK1_NAME, 'finetuned.pt')
finetuned2_checkpoint = os.path.join(args.save, TASK2_NAME, 'finetuned.pt')

pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu')
pretrained_state_dict = pretrained_model.state_dict()

tv1 = TaskVector(pretrained_checkpoint, finetuned1_checkpoint) # tau_Cars
tv2 = TaskVector(pretrained_checkpoint, finetuned2_checkpoint) # tau_MNIST
print("Task Vectors loaded.")

# --- 4. 데이터 로더 및 분류 헤드 준비 ---
print("Loading datasets and classification heads...")
dataloaders = {}
heads = {}
for name in ALL_TASK_NAME:
    dataset = get_dataset(name, pretrained_model.val_preprocess, location=DATA_LOCATION, batch_size=128)
    dataloaders[name] = dataset.test_loader
    heads[name] = get_classification_head(args, name).to(device)

loss_fn = nn.CrossEntropyLoss()
print("Dataloaders and heads ready.")

# --- 5. 손실 계산 루프 (1D 태스크 간 경로) ---
temp_encoder = ImageEncoder(args, keep_lang=False)
temp_encoder.eval()
temp_encoder.to(device)

# t=0 (Cars) 에서 t=1 (MNIST) 까지 이동
t_range = np.linspace(-0.2, 1.2, 25)
results = {
    't': [],
    'loss_task1(main)': [],
    'loss_task2(main)': [],
    'loss_task3': [],
    'loss_task4': [],
    'loss_task5': [],
    'loss_task6': [],
    'loss_task7': [],
    'loss_task8': [],
    'loss_weighted': []
}

pbar = tqdm(total=len(t_range), desc="Calculating 1D Inter-Task Path Loss")

with torch.no_grad():
    for t in t_range:
        # 모델 보간: theta = theta_pre + (1-t) * tau_Cars + t * tau_MNIST
        current_state_dict = copy.deepcopy(pretrained_state_dict)
        for key in pretrained_state_dict:
            vec1 = tv1.vector.get(key, 0)
            vec2 = tv2.vector.get(key, 0)
            
            # (1-t) * tau_1 + t * tau_2
            interp_vec = (1 - t) * vec1 + t * vec2
            
            current_state_dict[key] = pretrained_state_dict[key] + interp_vec

        temp_encoder.load_state_dict(current_state_dict, strict=False)
        
        loss_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        total_samples = [0, 0, 0, 0, 0, 0, 0, 0]
        for i, task_name in enumerate(ALL_TASK_NAME):
            total_samples[i] = 0
            for batch_idx, batch in enumerate(dataloaders[task_name]):
                if batch_idx >= 5: # 속도를 위해 5 배치만 테스트
                    break
                try:
                    images, labels = batch[0].to(device), batch[1].to(device)
                except:
                    batch = batch if isinstance(batch, dict) else {'images': batch[0], 'labels': batch[1]}
                    images, labels = batch['images'].to(device), batch['labels'].to(device)

                features = temp_encoder(images)
                logits = heads[TASK1_NAME](features)
                loss = loss_fn(logits, labels)
                loss_list[i] += loss.item() * images.size(0)
                total_samples[i] += images.size(0)
            
            if total_samples[i] > 0:
                loss_list[i] /= total_samples[i]

        # --- Task 1 (Cars) 손실 계산 ---
        # loss_1 = 0.0
        # total_samples_1 = 0
        # for batch_idx, batch in enumerate(dataloaders[TASK1_NAME]):
        #     if batch_idx >= 5: # 속도를 위해 5 배치만 테스트
        #          break
        #     try:
        #         images, labels = batch[0].to(device), batch[1].to(device)
        #     except:
        #         batch = batch if isinstance(batch, dict) else {'images': batch[0], 'labels': batch[1]}
        #         images, labels = batch['images'].to(device), batch['labels'].to(device)

        #     features = temp_encoder(images)
        #     logits = heads[TASK1_NAME](features)
        #     loss = loss_fn(logits, labels)
        #     loss_1 += loss.item() * images.size(0)
        #     total_samples_1 += images.size(0)
        
        # if total_samples_1 > 0:
        #     loss_1 /= total_samples_1

        # --- Task 2 (MNIST) 손실 계산 ---
        # loss_2 = 0.0
        # total_samples_2 = 0
        # for batch_idx, batch in enumerate(dataloaders[TASK2_NAME]):
        #     if batch_idx >= 5: # 속도를 위해 5 배치만 테스트
        #          break
        #     try:
        #         images, labels = batch[0].to(device), batch[1].to(device)
        #     except:
        #         batch = batch if isinstance(batch, dict) else {'images': batch[0], 'labels': batch[1]}
        #         images, labels = batch['images'].to(device), batch['labels'].to(device)

        #     features = temp_encoder(images)
        #     logits = heads[TASK2_NAME](features)
        #     loss = loss_fn(logits, labels)
        #     loss_2 += loss.item() * images.size(0)
        #     total_samples_2 += images.size(0)
        
        # if total_samples_2 > 0:
        #     loss_2 /= total_samples_2

        # --- 결과 저장 ---
        weighted_loss = OMEGA_1 * loss_list[1] + OMEGA_2 * loss_list[0]
        results['t'].append(t)
        results['loss_task1(main)'].append(loss_list[1])
        results['loss_task2(main)'].append(loss_list[0])
        results['loss_task3'].append(loss_list[2])
        results['loss_task4'].append(loss_list[3])
        results['loss_task5'].append(loss_list[4])
        results['loss_task6'].append(loss_list[5])
        results['loss_task7'].append(loss_list[6])
        results['loss_task8'].append(loss_list[7])
        results['loss_weighted'].append(weighted_loss)
        
        pbar.update(1)

pbar.close()
print("Loss calculation complete.")

# --- 7. 시각화 ---
print("Saving 1D loss plot...")
plt.figure(figsize=(10, 6))
plt.plot(results['t'], results['loss_task1(main)'], label=f'{TASK1_NAME} Loss (t=0 optimal)', linestyle='--', marker='o', markersize=4)
plt.plot(results['t'], results['loss_task2(main)'], label=f'{TASK2_NAME} Loss (t=1 optimal)', linestyle='--', marker='s', markersize=4)
plt.plot(results['t'], results['loss_task3'], label=f'{ALL_TASK_NAME[2]} Loss (t=2)', linestyle='--', marker='s', markersize=2)
plt.plot(results['t'], results['loss_task4'], label=f'{ALL_TASK_NAME[3]} Loss (t=3)', linestyle='--', marker='s', markersize=2)
plt.plot(results['t'], results['loss_task5'], label=f'{ALL_TASK_NAME[4]} Loss (t=4)', linestyle='--', marker='s', markersize=2)
plt.plot(results['t'], results['loss_task6'], label=f'{ALL_TASK_NAME[5]} Loss (t=5)', linestyle='--', marker='s', markersize=2)
plt.plot(results['t'], results['loss_task7'], label=f'{ALL_TASK_NAME[6]} Loss (t=6)', linestyle='--', marker='s', markersize=2)
plt.plot(results['t'], results['loss_task8'], label=f'{ALL_TASK_NAME[7]} Loss (t=7)', linestyle='--', marker='s', markersize=2)
plt.plot(results['t'], results['loss_weighted'], label=f'Weighted Average Loss (0.5*T1 + 0.5*T2)', linestyle='-', marker='x', markersize=6, linewidth=2.5, color='black')

plt.xlabel(f'Interpolation Coefficient (t) [0={TASK1_NAME}, 1={TASK2_NAME}]')
plt.ylabel('Loss (Cross Entropy)')
plt.title(f'Linear Interpolation Loss Barrier between $\Theta_{{{TASK1_NAME}}}$ and $\Theta_{{{TASK2_NAME}}}$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(0, color='blue', lw=1.0, linestyle='-.', label=f'Optimal {TASK1_NAME} (t=0)')
plt.axvline(1, color='red', lw=1.0, linestyle='-.', label=f'Optimal {TASK2_NAME} (t=1)')
plt.legend()

save_path = 'linear_path_barrier_INTERTASK_cars_mnist_and_others.png'
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"Plot saved successfully to {save_path}")