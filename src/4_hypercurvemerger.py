import torch
import torch.nn as nn

# (가정) 미분 가능한 NURBS 함수 구현
# (EXTRA 논문의 Eq. 4, 5를 PyTorch로 구현한 버전)
def differentiable_nurbs(control_points, t):
    # control_points: [Batch, Num_Points, N_Tasks]
    # t: [Batch, 1]
    # ... NURBS 계산 로직 ...
    # output: [Batch, N_Tasks] (t 시점의 Lambda 벡터)
    
    # (단순화를 위해, 여기서는 3차 베지어 곡선으로 대체)
    # P0 = (0,0,...)
    P0 = torch.zeros_like(control_points[:, 0, :])
    P1 = control_points[:, 0, :]
    P2 = control_points[:, 1, :]
    
    # t=1일 때의 최종 Lambda를 P3로 사용
    P3 = control_points[:, 2, :] 
    
    t_ = t.unsqueeze(-1) # [Batch, 1, 1]
    
    # (1-t)^3 * P0 + 3(1-t)^2 * t * P1 + 3(1-t) * t^2 * P2 + t^3 * P3
    lambda_t = (1-t_)**3 * P0 + \
               3 * (1-t_)**2 * t_ * P1 + \
               3 * (1-t_) * t_**2 * P2 + \
               t_**3 * P3
               
    return lambda_t.squeeze(1) # [Batch, N_Tasks]

class HyperCurveMerger(nn.Module):
    def __init__(self, num_tasks, num_control_points=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_control_points = num_control_points
        
        # 1. 하이퍼네트워크 (훈련 대상)
        # Input: N-dim omega
        # Output: P * N (제어점 P개의 N차원 좌표)
        self.hypernetwork = nn.Sequential(
            nn.Linear(num_tasks, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_control_points * num_tasks)
        )
        
        # (1단계에서 준비한 재료들은 여기에 멤버 변수로 저장)
        # self.theta_pre = torch.load(...)
        # self.task_vectors_tau = [torch.load(...), ...]
        # self.task_vectors_tau = {key: val.to(device) for tv in ...}
        
    def get_lambdas(self, omega, t):
        """
        omega와 t를 입력받아, t 시점의 N-dim Lambda 벡터를 반환
        """
        # 1. H(omega) -> 제어점 P 생성
        # [Batch, P * N]
        control_points_flat = self.hypernetwork(omega) 
        
        # [Batch, P, N]
        control_points = control_points_flat.view(
            -1, self.num_control_points, self.num_tasks
        )
        
        # 2. NURBS(P, t) -> Lambda(t) 계산
        # [Batch, N_Tasks]
        lambda_t = differentiable_nurbs(control_points, t)
        
        return lambda_t

    def get_merged_model_params(self, omega, t, theta_pre_dict, task_vector_dicts):
        """
        omega와 t에 대한 최종 모델의 state_dict를 조립
        (이 함수 자체는 훈련 중에 호출됨)
        """
        
        # 1. t 시점의 Lambda 벡터 가져오기
        # [Batch=1, N_Tasks]
        lambda_t = self.get_lambdas(omega.unsqueeze(0), t.unsqueeze(0))
        lambda_t = lambda_t.squeeze(0) # [N_Tasks]
        
        # 2. 모델 조립 (미분 그래프가 연결됨)
        # Theta(t) = Theta_pre + sum( lambda_i(t) * tau_i )
        
        final_params = {}
        for key in theta_pre_dict:
            # 기본 파라미터 (Theta_pre)
            final_params[key] = theta_pre_dict[key].clone()
            
            # 태스크 벡터 조립
            for i in range(self.num_tasks):
                # i번째 태스크 벡터에서 key에 해당하는 값 가져오기
                tau_i_vec = task_vector_dicts[i].get(key, 0)
                
                # lambda_i(t) * tau_i
                # lambda_t[i]는 H(omega)에서 왔으므로,
                # 이 연산은 H의 파라미터까지 미분 그래프를 연결시킴
                final_params[key] += lambda_t[i] * tau_i_vec
                
        return final_params

    def forward(self, omega, t):
        # 이 모델의 'forward'는 최종 파라미터를 반환하는 것
        # (실제 훈련 루프에서는 get_merged_model_params를 직접 사용)
        
        # (주의: 실제 훈련에서는 state_dict를 매번 로드하지 않고
        #  파라미터가 비어있는 '모델 껍데기'에 주입해야 함)
        
        # 예시:
        # 1. Phase 1의 재료 로드 (실제로는 클래스 초기화 시 한 번만)
        # theta_pre = torch.load(...) 
        # task_vectors = [torch.load(...) for ...]
        
        # 2. 파라미터 계산
        # final_params = self.get_merged_model_params(omega, t, theta_pre, task_vectors)
        # return final_params
        pass