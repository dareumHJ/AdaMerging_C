import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import copy

# --- src 경로 추가 ---
script_dir = os.getcwd()
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from task_vectors import TaskVector
    from modeling import ImageEncoder
    from args import parse_arguments
    from datasets.registry import get_dataset
    from heads import get_classification_head
    from ties_merging_utils import state_dict_to_vector, vector_to_state_dict
except ImportError:
    print("Error: 'src' modules not found. Make sure you are running this script from the project root.")
    sys.exit(1)

# --- 설정 ---
TASK1 = 'Cars'
TASK2 = 'SVHN'
MODEL = 'ViT-B-32'
STEPS = 11  # 0.0, 0.1, ..., 1.0
RESET_THRESH = 20 # TIES Trimming: 하위 80% 제거 (상위 20% 보존)


# --- [Visualization] Plotting Function ---
def plot_pareto_frontier(results_linear, results_ties, task1_name, task2_name, save_path='pareto_frontier.png'):
    """
    Pareto Frontier 시각화 함수
    """
    # Numpy 변환
    pts_lin = np.array(results_linear)
    pts_ties = np.array(results_ties)
    
    plt.figure(figsize=(10, 10))
    
    # Plot Lines
    plt.plot(pts_lin[:, 0], pts_lin[:, 1], 'r--o', label='Linear Merging (Baseline)', alpha=0.7)
    plt.plot(pts_ties[:, 0], pts_ties[:, 1], 'b-s', label='Weighted TIES (Ours)', linewidth=2)
    
    # 시작점(Task 2 Only)과 끝점(Task 1 Only) 강조
    # pts_lin[0]은 w1=0 (Task2 100%), pts_lin[-1]은 w1=1 (Task1 100%)
    plt.scatter(pts_lin[0, 0], pts_lin[0, 1], s=150, marker='*', c='green', label=f'{task2_name} Only', zorder=5)
    plt.scatter(pts_lin[-1, 0], pts_lin[-1, 1], s=150, marker='*', c='orange', label=f'{task1_name} Only', zorder=5)
    
    plt.xlabel(f'{task1_name} Accuracy')
    plt.ylabel(f'{task2_name} Accuracy')
    plt.title(f'Pareto Frontier: {task1_name} vs {task2_name}')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Pareto plot saved to {save_path}")


# --- [Helper] Top-K Trimming ---
def topk_values_mask(M, K=20):
    k_fraction = K / 100.0
    n, d = M.shape
    k_idx = int(d * k_fraction)
    
    kth_values, _ = M.abs().kthvalue(d - k_idx, dim=1, keepdim=True)
    mask = M.abs() >= kth_values
    return M * mask

# --- [Core] Weighted TIES Algorithm ---
def ties_merging_weighted(flat_checks, omega_weights, reset_thresh=20, scaling_factor=1.0):
    # 1. TRIM
    updated_checks = topk_values_mask(flat_checks, K=reset_thresh)
    
    # 2. ELECT SIGN (Weighted)
    weighted_sum_for_sign = torch.sum(updated_checks * omega_weights.unsqueeze(-1), dim=0)
    elected_signs = torch.sign(weighted_sum_for_sign)
    elected_signs[elected_signs == 0] = 1 

    # 3. DISJOINT MERGE (Weighted)
    rows_to_keep = (torch.sign(updated_checks) == elected_signs.unsqueeze(0))
    weighted_values = updated_checks * omega_weights.unsqueeze(-1)
    selected_values = weighted_values * rows_to_keep
    
    surviving_weights_sum = torch.sum(rows_to_keep * omega_weights.unsqueeze(-1), dim=0)
    # merged_tv = torch.sum(selected_values, dim=0) / (surviving_weights_sum + 1e-6)
    merged_tv = torch.sum(selected_values, dim=0)
    
    return merged_tv * scaling_factor


# --- 메인 실행 ---
def main():
    # 1. Arguments & Setup
    args = parse_arguments()
    args.model = MODEL
    args.save = f'./checkpoints/{MODEL}'
    args.data_location = './data'
    args.openclip_cachedir = './openclip_cache'
    
    device = args.device
    print(f"Using device: {device}")

    # 2. Load Models & Vectors
    print("Loading vectors...")
    ckpt_pre = os.path.join(args.save, 'zeroshot.pt')
    ckpt_t1 = os.path.join(args.save, TASK1, 'finetuned.pt')
    ckpt_t2 = os.path.join(args.save, TASK2, 'finetuned.pt')
    
    model_pre = torch.load(ckpt_pre, map_location='cpu')
    sd_pre = model_pre.state_dict()
    
    tv1 = TaskVector(ckpt_pre, ckpt_t1)
    tv2 = TaskVector(ckpt_pre, ckpt_t2)
    
    flat_tv1 = state_dict_to_vector(tv1.vector, []).to(device)
    flat_tv2 = state_dict_to_vector(tv2.vector, []).to(device)
    stacked_tvs = torch.stack([flat_tv1, flat_tv2]) 

    # 3. Prepare Evaluation (Dataloaders & Heads)
    dataloaders = {}
    heads = {}
    model_eval = ImageEncoder(args, keep_lang=False).to(device)
    
    for t in [TASK1, TASK2]:
        ds = get_dataset(t, model_pre.val_preprocess, location=args.data_location, batch_size=128)
        dataloaders[t] = ds.test_loader
        heads[t] = get_classification_head(args, t).to(device)
    
    # 내부 평가 함수
    def evaluate_model(merged_state_dict):
        model_eval.load_state_dict(merged_state_dict, strict=False)
        model_eval.eval()
        
        accs = {}
        with torch.no_grad():
            for t_name in [TASK1, TASK2]:
                correct = 0
                total = 0
                for i, batch in enumerate(dataloaders[t_name]):
                    if i >= 5: break # Fast evaluation
                    
                    if isinstance(batch, dict):
                        img, lbl = batch['images'], batch['labels']
                    else:
                        img, lbl = batch[0], batch[1]
                    
                    img, lbl = img.to(device), lbl.to(device)
                    logits = heads[t_name](model_eval(img))
                    pred = logits.argmax(dim=1)
                    correct += (pred == lbl).sum().item()
                    total += img.size(0)
                accs[t_name] = correct / total if total > 0 else 0
        return accs[TASK1], accs[TASK2]

    # 4. Run Experiment (Pareto Sweep)
    results_linear = []
    results_ties = []
    
    omegas = np.linspace(0, 1, STEPS)
    print(f"Starting Pareto Frontier Sweep ({STEPS} steps)...")
    
    for w1 in tqdm(omegas):
        w2 = 1.0 - w1
        omega_tensor = torch.tensor([w1, w2], device=device, dtype=torch.float32)
        
        # --- A. Linear Merging ---
        merged_vec_lin = w1 * flat_tv1 + w2 * flat_tv2
        diff_dict_lin = vector_to_state_dict(merged_vec_lin.cpu(), sd_pre, [])
        
        sd_linear = copy.deepcopy(sd_pre)
        for k in sd_linear:
            if k in diff_dict_lin:
                sd_linear[k] = sd_linear[k] + diff_dict_lin[k]
        
        results_linear.append(evaluate_model(sd_linear))
        
        # --- B. Weighted TIES ---
        merged_vec_ties = ties_merging_weighted(stacked_tvs, omega_tensor, reset_thresh=RESET_THRESH, scaling_factor=1.0)
        diff_dict_ties = vector_to_state_dict(merged_vec_ties.cpu(), sd_pre, [])
        
        sd_ties = copy.deepcopy(sd_pre)
        for k in sd_ties:
            if k in diff_dict_ties:
                sd_ties[k] = sd_ties[k] + diff_dict_ties[k]
                
        results_ties.append(evaluate_model(sd_ties))

    # 5. Visualization
    plot_pareto_frontier(
        results_linear, 
        results_ties, 
        task1_name=TASK1, 
        task2_name=TASK2, 
        save_path='pareto_frontier_final.png'
    )

if __name__ == "__main__":
    main()