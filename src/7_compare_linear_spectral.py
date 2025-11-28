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
# 기존 작성한 유틸리티 재사용
from spectral_utils import decompose_task_vector, spectral_merge

# --- Configuration ---
TASKS = ['DTD', 'GTSRB', 'SVHN']
MODEL = 'ViT-B-32'
SV_REDUCTION_RATIO = 1.0 / len(TASKS) # 약 0.33
SV_REDUCTION_RATIO = 1.0
SCALING_FACTOR = 1.0

def linear_merge(task_vectors, preferences, scaling_factor=1.0, device='cuda'):
    """
    Method A: Linear Merging (Baseline)
    단순 가중 합: W_new = sum( w_i * W_i )
    """
    merged_tv = {}
    # 첫 번째 태스크의 키를 기준으로 순회
    all_keys = task_vectors[0].keys()
    
    for key in all_keys:
        # 초기화 (첫 번째 태스크의 shape과 동일한 0 텐서)
        ref_tensor = task_vectors[0][key]
        weighted_sum = torch.zeros_like(ref_tensor, device=device)
        
        for i, tv in enumerate(task_vectors):
            w = preferences[i]
            if w > 0:
                weighted_sum += tv[key].to(device) * w
        
        merged_tv[key] = weighted_sum * scaling_factor
        
    return merged_tv

def evaluate_model(model, heads, dataloaders, device):
    """모델 평가 (Fast Mode: 10 batches)"""
    model.eval()
    accuracies = {}
    with torch.no_grad():
        for task in TASKS:
            head = heads[task]
            loader = dataloaders[task]
            correct = 0
            total = 0
            for i, batch in enumerate(loader):
                if i >= 10: break # 속도를 위해 10개 배치만
                
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

def apply_and_evaluate(model_pre, merged_tv_dict, model_eval, heads, dataloaders, device):
    """병합된 벡터를 모델에 적용하고 평가"""
    sd_merged = copy.deepcopy(model_pre.state_dict())
    for k, v in merged_tv_dict.items():
        if k in sd_merged:
            sd_merged[k] = sd_merged[k].to(device) + v.to(device)
    
    model_eval.load_state_dict(sd_merged, strict=False)
    return evaluate_model(model_eval, heads, dataloaders, device)

def main():
    args = parse_arguments()
    args.model = MODEL
    args.save = f'./checkpoints/{MODEL}'
    args.data_location = './data'
    args.openclip_cachedir = os.path.expanduser("~/openclip-cachedir/open_clip")
    device = args.device
    
    print(f"Comparison Experiment: {TASKS} on {device}")

    # 1. Load Checkpoints
    print("\n[1] Loading Task Vectors...")
    ckpt_pre = os.path.join(args.save, 'zeroshot.pt')
    model_pre = torch.load(ckpt_pre, map_location='cpu')
    
    raw_task_vectors = []
    for t in TASKS:
        ckpt_ft = os.path.join(args.save, t, 'finetuned.pt')
        tv = TaskVector(ckpt_pre, ckpt_ft)
        raw_task_vectors.append(tv.vector)

    # 2. SVD Pre-computation (For Spectral)
    print("\n[2] SVD Decomposition for Spectral Method...")
    decomposed_tvs = []
    for i, tv_dict in enumerate(raw_task_vectors):
        decomp = decompose_task_vector(tv_dict, reduction_ratio=SV_REDUCTION_RATIO)
        decomposed_tvs.append(decomp)
    print("SVD Ready.")

    # 3. Prepare Evaluation
    print("\n[3] Preparing Evaluators...")
    model_eval = ImageEncoder(args, keep_lang=False).to(device)
    heads = {}
    dataloaders = {}
    for t in TASKS:
        heads[t] = get_classification_head(args, t).to(device)
        ds = get_dataset(t, model_pre.val_preprocess, location=args.data_location, batch_size=128)
        dataloaders[t] = ds.test_loader

    # 4. Test Points (Preferences)
    test_points = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], 
        [0.33, 0.33, 0.34], # Balanced Case (Critical!)
        [0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]
    ]
    
    results = []
    
    print(f"\n[4] Running Comparison Loop ({len(test_points)} points)...")
    for prefs in tqdm(test_points, desc="Comparing Methods"):
        row = {}
        for i, t in enumerate(TASKS):
            row[f'w_{t}'] = prefs[i]

        # --- Method A: Linear Merging ---
        merged_lin = linear_merge(raw_task_vectors, prefs, SCALING_FACTOR, device)
        acc_lin = apply_and_evaluate(model_pre, merged_lin, model_eval, heads, dataloaders, device)
        
        for t in TASKS:
            row[f'lin_{t}'] = acc_lin[t]
            
        # --- Method B: Spectral Merging ---
        merged_spec = spectral_merge(decomposed_tvs, prefs, SCALING_FACTOR, device)
        acc_spec = apply_and_evaluate(model_pre, merged_spec, model_eval, heads, dataloaders, device)
        
        for t in TASKS:
            row[f'spec_{t}'] = acc_spec[t]
            # 성능 차이 계산 (Spectral - Linear)
            row[f'diff_{t}'] = acc_spec[t] - acc_lin[t]

        results.append(row)
        
        # Memory Cleanup
        del merged_lin, merged_spec
        torch.cuda.empty_cache()

    # 5. Save & Summary
    df = pd.DataFrame(results)
    
    # 컬럼 순서 정리
    cols = [f'w_{t}' for t in TASKS] + \
           [f'lin_{t}' for t in TASKS] + \
           [f'spec_{t}' for t in TASKS] + \
           [f'diff_{t}' for t in TASKS]
    df = df[cols]
    
    print("\n=== Comparison Summary (Balanced Case) ===")
    balanced_idx = 6 # [0.33, 0.33, 0.34]
    print(df.iloc[balanced_idx])
    
    csv_path = "comparison_linear_vs_spectral.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")

if __name__ == "__main__":
    main()