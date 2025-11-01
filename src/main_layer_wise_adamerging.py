import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import sys
import tqdm
sys.path.append('./taskarithmetic/')

import torch
from task_vectors import TaskVector
from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
from args import parse_arguments
def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
model = 'ViT-B-32'
args = parse_arguments()
args.data_location = './data'
args.model = model
args.save = './checkpoints/' + model
args.logs_path = './logs/' + model
pretrained_checkpoint = './checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_Layer_wise_AdaMerging.txt'.format(str_time_))
args.log = log

task_vectors = [TaskVector(pretrained_checkpoint, './checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets]

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features

from heads import get_classification_head
class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = 0.3
        rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()

model = ModelWrapper(pretrained_model, exam_datasets)
model = model.to(args.device)
_, names = make_functional(model)

paramslist = []
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors
torch.cuda.empty_cache()
adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets)

print('init lambda:')
print(adamerging_mtl_model.lambdas())
print('collect_trainable_params:')
print(list(adamerging_mtl_model.collect_trainable_params()))

epochs = 500
optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)

from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

Total_ACC = 0.
for dataset_name in exam_datasets:
    image_encoder = adamerging_mtl_model.get_image_encoder()
    classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
    metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
    Total_ACC += metrics['top1']
    log.info('Eval: init: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
log.info('Eval: init: ' + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')

for epoch in range(epochs):
    losses = 0.
    for dataset_name in exam_datasets:
        dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=16)
        dataloader = get_dataloader_shuffle(dataset)

        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)

            outputs = adamerging_mtl_model(x, dataset_name)
            loss = softmax_entropy(outputs).mean(0)
            losses += loss

            if i > 0:  
                break

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    print(list(adamerging_mtl_model.lambdas().data))

    if ((epoch+1) % 500) == 0:
        log.info(str(list(adamerging_mtl_model.lambdas().data)))

        Total_ACC = 0.
        for dataset_name in exam_datasets:
            image_encoder = adamerging_mtl_model.get_image_encoder()
            classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
            metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
            Total_ACC += metrics['top1']
            log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
        log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')

log.info("--- 훈련 종료. 최종 병합(merged) 모델 파라미터 계산 및 저장 시작 ---")

# 1. 훈련된 최종 람다(lambda) 값을 가져옵니다. (CPU로 이동)
final_lambdas = adamerging_mtl_model.lambdas().cpu()

# 2. 파라미터 리스트(사전 훈련 + 태스크 벡터)를 가져옵니다.
params_list_of_lists = adamerging_mtl_model.paramslist

# 3. 원본 state_dict에서 파라미터 이름 리스트를 가져옵니다.
# (params_list_of_lists와 순서가 동일함이 보장됩니다)
param_names = list(pretrained_model_dic.keys())

# 4. 최종 람다를 사용해 새로운 파라미터를 계산합니다.
# 각 파라미터 p에 대해 p_merged = lambda_0*p_0 + lambda_1*p_1 + ... 를 수행합니다.
final_params_tuple = tuple(
    sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, final_lambdas[j]))) 
    for j, p in enumerate(zip(*params_list_of_lists))
)

# 5. 파라미터 이름과 계산된 텐서를 묶어 새로운 state_dict를 생성합니다.
final_state_dict = {name: param for name, param in zip(param_names, final_params_tuple)}

log.info("파라미터를 저장하기 위해 원본 체크포인트로부터 새 모델 객체를 로드합니다...")

# 6. 'zeroshot.pt'에서 파라미터가 제거되지 않은 "깨끗한" 모델 객체를 *새로* 로드합니다.
#    (기존의 'pretrained_model' 변수는 make_functional에 의해 파라미터가 제거된 상태입니다.)
fresh_model_shell = torch.load(pretrained_checkpoint)

# 7. 이 깨끗한 모델 객체에 새로 계산한 final_state_dict를 로드합니다.
try:
    fresh_model_shell.load_state_dict(final_state_dict)
    
    # 8. 파라미터가 덮어씌워진 이 *새로운* 모델 객체를 저장합니다.
    save_path = os.path.join(args.save, 'adamerged_final.pt')
    torch.save(fresh_model_shell, save_path)

    log.info(f"--- 최종 병합 모델이 {save_path}에 성공적으로 저장되었습니다. ---")

except Exception as e:
    log.error(f"!!! 모델 저장 중 오류 발생: {e}")
    log.error("state_dict의 키 개수와 모델의 키 개수가 일치하는지 확인하세요.")
    log.error(f"계산된 파라미터 수: {len(final_state_dict)}")
    log.error(f"새로 로드한 모델의 파라미터 수: {len(fresh_model_shell.state_dict())}")

# --- 💡 수정 끝 ---