import torch
import torch.nn as nn

def compute_svd_and_compress(tensor, reduction_ratio=1.0):
    """
    행렬에 대해 SVD를 수행하고 상위 Singular Value만 남깁니다 (TSV 논문 구현).
    """
    # float32로 변환하여 정밀도 확보
    matrix = tensor.float()
    
    # SVD 수행 (Economy size)
    try:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError as e:
        # SVD 수렴 실패 시 CPU에서 재시도
        print(f"SVD on GPU failed, retrying on CPU...")
        U, S, Vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        U, S, Vh = U.to(matrix.device), S.to(matrix.device), Vh.to(matrix.device)

    # Rank Truncation (TSV 논문의 핵심: 노이즈 제거)
    if reduction_ratio < 1.0:
        k = int(S.shape[0] * reduction_ratio)
        k = max(k, 1) # 최소 1개는 유지
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
        
    return U, S, Vh

def decompose_task_vector(task_vector_dict, reduction_ratio=1.0):
    """
    Task Vector를 SVD 분해합니다.
    TSV 논문 로직: 2D 텐서(Weight)만 분해하고, 나머지는 그대로 둡니다.
    """
    decomposed = {}
    print(f"Executing SVD decomposition (Top {reduction_ratio*100:.1f}%)...")
    
    for key, tensor in task_vector_dict.items():
        # TSV 코드 참조: 2차원 텐서만 SVD 수행 (Linear Layer)
        # text_projection 등은 제외하는 것이 안전함
        if tensor.dim() == 2 and "text_projection" not in key:
            U, S, Vh = compute_svd_and_compress(tensor, reduction_ratio)
            
            decomposed[key] = {
                'U': U.cpu(),   # 메모리 절약을 위해 CPU 저장
                'S': S.cpu(),
                'Vh': Vh.cpu(),
                'type': 'matrix'
            }
        else:
            # Bias, LayerNorm, Conv 등은 SVD 하지 않음
            decomposed[key] = {
                'data': tensor.cpu(), 
                'type': 'vector'
            }
            
    return decomposed

def spectral_merge(decomposed_tasks, preferences, scaling_factor=1.0, device='cuda'):
    """
    Method 2: Spectral Pareto Merging
    Singular Value(Energy)에 Preference를 가중치로 곱하여 병합합니다.
    """
    merged_tv = {}
    # 첫 번째 태스크의 키를 기준으로 순회
    all_keys = decomposed_tasks[0].keys()
    
    for key in all_keys:
        meta_type = decomposed_tasks[0][key]['type']
        
        # A. 단순 벡터/텐서 (Bias, LayerNorm 등) -> Linear Weighted Sum
        if meta_type == 'vector':
            # 초기화
            ref_tensor = decomposed_tasks[0][key]['data']
            weighted_sum = torch.zeros_like(ref_tensor, device=device)
            
            for i, task_dict in enumerate(decomposed_tasks):
                w = preferences[i]
                if w > 0:
                    weighted_sum += task_dict[key]['data'].to(device) * w
            
            merged_tv[key] = weighted_sum * scaling_factor
            
        # B. 행렬 (Weight) -> Spectral Scaling & Reconstruction
        # Formula: Sum ( U_i * (S_i * gamma_i) * Vh_i )
        else:
            final_matrix = None
            
            for i, task_dict in enumerate(decomposed_tasks):
                gamma = preferences[i]
                if gamma <= 1e-4: continue # 가중치 0이면 스킵
                
                comp = task_dict[key]
                # 연산 시점에만 GPU로 이동
                U = comp['U'].to(device)
                S = comp['S'].to(device)
                Vh = comp['Vh'].to(device)
                
                U_full = comp['U']
                S_full = comp['S']
                Vh_full = comp['Vh']
                
                total_rank = S_full.shape[0]
                keep_ratio = max(0.1, gamma) # at least 10%
                k = int(total_rank * keep_ratio)
                
                U = U_full[:, :k].to(device)
                S = S_full[:k].to(device)
                Vh = Vh_full[:k, :].to(device)
                
                
                # Spectral Scaling: 에너지(Singular Value)를 선호도에 따라 조절
                scaled_S = S * gamma
                
                # Reconstruction: U @ diag(S) @ Vh
                # unsqueeze를 이용한 브로드캐스팅 곱셈이 diag행렬 생성보다 빠름
                reconstructed = U @ (scaled_S.unsqueeze(1) * Vh)
                
                if final_matrix is None:
                    final_matrix = reconstructed
                else:
                    final_matrix += reconstructed
            
            if final_matrix is None: # 모든 gamma가 0인 경우 (예외처리)
                ref_u = decomposed_tasks[0][key]['U']
                final_matrix = torch.zeros(ref_u.shape[0], decomposed_tasks[0][key]['Vh'].shape[1], device=device)

            merged_tv[key] = final_matrix * scaling_factor
            
    return merged_tv