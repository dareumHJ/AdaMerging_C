import os
from datasets import load_dataset
from tqdm import tqdm
import warnings

# Hugging Face Hub의 데이터셋 로드 시 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.utils._deprecation")

def download_all_datasets(root_path="../../data/hf_datasets"):
    """ 'AdaMerging_C' 프로젝트에 필요한 모든 데이터셋을 Hugging Face Hub에서 다운로드. """
    dataset_hub_names = {
        # "MNIST": "mnist",
        # "SVHN": ("svhn", "cropped_digits"), # SVHN 데이터셋의 특정 버전
        # "EuroSAT": "blanchon/EuroSAT_RGB",
        # "DTD": "jxie/dtd",
        "GTSRB": "ilee0022/GTSRB",
        "StanfordCars": "tanganke/stanford_cars",
        "RESISC45": "blanchon/RESISC45",
        "SUN397": "tanganke/sun397",
    }
    
    split_mapping = {
        "default": {"train": "train", "test": "test"},
        "DTD": {"train": "train", "test": "validation"}, # DTD는 'validation'을 test로 사용
    }
    
    print("Starting dataset downloads...")
    os.makedirs(root_path, exist_ok=True)
    
    for name, hub_id in dataset_hub_names.items():
        try:
            print(f"Downloading {name} dataset...")
            
            if isinstance(hub_id, tuple):
                dataset = load_dataset(hub_id[0], hub_id[1], trust_remote_code=True)
            else:
                dataset = load_dataset(hub_id, trust_remote_code=True)
            
            # Get class names
            class_names = dataset['train'].features['label'].names
            
            # save corresponding directory by split mapping
            splits = split_mapping.get(name, split_mapping["default"])
            for split_name, hf_split_name in splits.items():
                if hf_split_name not in dataset:
                    print(f"Warning: Split '{hf_split_name}' not found in {name} dataset. Skipping this split.")
                    continue
                
                print(f"Processing {split_name} split...")
                split_dataset = dataset[hf_split_name]
                
                for i, example in enumerate(tqdm(split_dataset, desc=f"Saving {split_name} split of {name}")):
                    image = example['image']
                    label = example['label']
                    
                    # Create class directory
                    class_name_str = class_names[label]
                    class_dir = os.path.join(root_path, name, split_name, class_name_str)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # save images (RGB conversion)
                    image_path = os.path.join(class_dir, f"{i}.jpg")
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(image_path)            
            
            print(f"{name} dataset downloaded successfully.")

        except Exception as e:
            print(f"Failed to download {name} dataset. Error: {e}")
            
if __name__ == "__main__":
    download_all_datasets()