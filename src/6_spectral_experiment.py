import os
import sys
import torch
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

# --- Import src modules ---
script_dir = os.getcwd()
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)

from task_vectors import TaskVector
from modeling import ImageEncoder
from args import parse_arguments
from datasets.registry import get_dataset
from heads import get_classification_head
from spectral_utils import decompose_task_vector, spectral_merge

# --- Configuration ---
# 실험할 태스크 3개 (가장 간섭이 심할 것으로 예상되는 조합 추천)
TASKS = ['DTD', 'GTSRB', 'SVHN'] 
MODEL = 'ViT-B-32'

# TSV 논문 권장: sv_reduction = 1 / num_tasks
# 3개 태스크이므로 약 0.33 (상위 33%만 사용하고 나머지 노이즈 제거)
SV_REDUCTION_RATIO = 1.0 / len(TASKS) 
SCALING_FACTOR = 1.0 

def evaluate_model(model, heads, dataloaders, device):
    """모델 하나를 모든 태스크에 대해 빠르게 평가"""
    model.eval()
    accuracies = {}
    
    with torch.no_grad():
        for task in TASKS:
            head = heads[task]
            loader = dataloaders[task]
            
            correct = 0
            total = 0
            # 속도를 위해 10개 배치만 평가 (경향성 확인용)
            # 전체 정확도를 보려면 이 제한을 푸세요.
            for i, batch in enumerate(loader):
                if i >= 10: break 
                
                if isinstance(batch, dict):
                    img, lbl = batch['images'], batch['labels']
                else:
                    img, lbl = batch[0], batch[1]
                
                img, lbl = img.to(device), lbl.to(device)
                logits = head(model(img))
                pred = logits.argmax(dim=1)
                correct += (pred == lbl).sum().item()
                total += img.size(0)
            
            accuracies[task] = correct / total if total > 0 else 0
    return accuracies

def main():
    args = parse_arguments()
    args.model = MODEL
    args.save = f'./checkpoints/{MODEL}'
    args.data_location = './data'
    # TSV 코드에서 사용하는 캐시 경로 호환
    args.openclip_cachedir = os.path.expanduser("~/openclip-cachedir/open_clip")
    device = args.device
    
    print(f"Experimental Setup: {TASKS} on {device}")
    print(f"SVD Rank Reduction Ratio: {SV_REDUCTION_RATIO:.4f}")

    # 1. Load Models & Compute Task Vectors
    print("\n[1] Loading checkpoints...")
    ckpt_pre = os.path.join(args.save, 'zeroshot.pt')
    if not os.path.exists(ckpt_pre):
        print(f"Error: Zeroshot checkpoint not found at {ckpt_pre}")
        return

    model_pre = torch.load(ckpt_pre, map_location='cpu')
    sd_pre = model_pre.state_dict()
    
    raw_task_vectors = []
    for t in TASKS:
        ckpt_ft = os.path.join(args.save, t, 'finetuned.pt')
        print(f"Loading {t} from {ckpt_ft}...")
        tv = TaskVector(ckpt_pre, ckpt_ft)
        raw_task_vectors.append(tv.vector) 

    # 2. SVD Decomposition (Phase 1) - Pre-computation
    print("\n[2] Performing SVD Decomposition (Cached)...")
    decomposed_tvs = []
    for i, tv_dict in enumerate(raw_task_vectors):
        print(f" >> Decomposing Task {TASKS[i]}...")
        # 여기서 reduction_ratio를 적용하여 노이즈를 미리 제거합니다.
        decomp = decompose_task_vector(tv_dict, reduction_ratio=SV_REDUCTION_RATIO)
        decomposed_tvs.append(decomp)
    
    # 원본 메모리 해제
    del raw_task_vectors
    torch.cuda.empty_cache()
    print("SVD Done. Decomposed vectors cached in CPU RAM.")

    # 3. Prepare Evaluation (Heads & Loaders)
    print("\n[3] Preparing Evaluators...")
    model_eval = ImageEncoder(args, keep_lang=False).to(device)
    heads = {}
    dataloaders = {}
    for t in TASKS:
        heads[t] = get_classification_head(args, t).to(device)
        ds = get_dataset(t, model_pre.val_preprocess, location=args.data_location, batch_size=128)
        dataloaders[t] = ds.test_loader

    # 4. Define Preference Points (Phase 2 & 3)
    # Simplex Grid Sampling (3 Tasks)
    # 극단값, 중간값, 균형값 등을 골고루 섞음
    test_points = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], # Single Task
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], # Pairwise Balanced
        [0.33, 0.33, 0.34], # All Balanced (Approx)
        [0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7], # Task Dominant
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]  # Extreme Dominant
    ]
    
    results = []
    
    print(f"\n[4] Starting Spectral Pareto Merging ({len(test_points)} points)...")
    for prefs in tqdm(test_points, desc="Evaluating Preferences"):
        # A. Merge (Spectral) - GPU 연산
        merged_tv_dict = spectral_merge(
            decomposed_tvs, 
            prefs, 
            scaling_factor=SCALING_FACTOR, 
            device=device
        )
        
        # B. Apply to Model
        # Base Model(sd_pre)에 병합된 벡터(merged_tv_dict)를 더함
        sd_merged = copy.deepcopy(sd_pre)
        for k, v in merged_tv_dict.items():
            if k in sd_merged:
                # v는 이미 device에 있거나 spectral_merge에서 처리됨
                sd_merged[k] = sd_merged[k].to(device) + v.to(device)
        
        model_eval.load_state_dict(sd_merged, strict=False)
        
        # C. Evaluate
        accs = evaluate_model(model_eval, heads, dataloaders, device)
        
        # Result Logging
        res_entry = {'prefs': prefs}
        res_entry.update(accs)
        results.append(res_entry)
        
        # Memory Cleanup
        del merged_tv_dict
        del sd_merged
        torch.cuda.empty_cache()

    # 5. Save Results
    df_data = []
    for r in results:
        row = {}
        for i, t in enumerate(TASKS):
            row[f'w_{t}'] = r['prefs'][i]
        for t in TASKS:
            row[f'acc_{t}'] = r[t]
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    print("\n=== Results Summary ===")
    print(df)
    
    csv_path = "spectral_merging_results_3tasks.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()