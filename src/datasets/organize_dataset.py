import os
import shutil
from datasets import load_dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.utils._deprecation")

def organize_cloned_datasets(input_root, output_root="organized_datasets"):
    """git clone으로 받은 데이터셋 폴더를 ImageFloder 형식으로 재구성."""
    
    dataset_paths = {
        "GTSRB": os.path.join(input_root, "GTSRB"),
        "Cars": os.path.join(input_root, "stanford_cars"),
        "RESISC45": os.path.join(input_root, "RESISC45"),
        "SUN397": os.path.join(input_root, "sun397"),
    }
    
    split_mapping = {
        "default": {"train": "train", "test": "test"},
    }
    
    # GTSRB 데이터셋을 위한 수동 클래스 이름 목록
    gtsrb_class_names = [
        'red and white circle 20 kph speed limit', 'red and white circle 30 kph speed limit',
        'red and white circle 50 kph speed limit', 'red and white circle 60 kph speed limit',
        'red and white circle 70 kph speed limit', 'red and white circle 80 kph speed limit',
        'end / de-restriction of 80 kph speed limit', 'red and white circle 100 kph speed limit',
        'red and white circle 120 kph speed limit', 'red and white circle red car and black car no passing',
        'red and white circle red truck and black car no passing', 'red and white triangle road intersection warning',
        'white and yellow diamond priority road', 'red and white upside down triangle yield right-of-way',
        'stop', 'empty red and white circle', 'red and white circle no truck entry',
        'red circle with white horizonal stripe no entry', 'red and white triangle with exclamation mark warning',
        'red and white triangle with black left curve approaching warning', 'red and white triangle with black right curve approaching warning',
        'red and white triangle with black double curve approaching warning', 'red and white triangle rough / bumpy road warning',
        'red and white triangle car skidding / slipping warning', 'red and white triangle with merging / narrow lanes warning',
        'red and white triangle with person digging / construction / road work warning', 'red and white triangle with traffic light approaching warning',
        'red and white triangle with person walking warning', 'red and white triangle with child and person walking warning',
        'red and white triangle with bicyle warning', 'red and white triangle with snowflake / ice warning',
        'red and white triangle with deer warning', 'white circle with gray strike bar no speed limit',
        'blue circle with white right turn arrow mandatory', 'blue circle with white left turn arrow mandatory',
        'blue circle with white forward arrow mandatory', 'blue circle with white forward or right turn arrow mandatory',
        'blue circle with white forward or left turn arrow mandatory', 'blue circle with white keep right arrow mandatory',
        'blue circle with white keep left arrow mandatory', 'blue circle with white arrows indicating a traffic circle',
        'white circle with gray strike bar indicating no passing for cars has ended', 'white circle with gray strike bar indicating no passing for trucks has ended',
    ]
    
    print(f"Organizing datasets into {output_root}...")
    os.makedirs(output_root, exist_ok=True)
    
    for name, local_path in dataset_paths.items():
        if not os.path.exists(local_path):
            print(f"Warning: Dataset path {local_path} does not exist. Skipping {name}.")
            continue
        try:
            print(f"---'{name}' dataset processing... (from {local_path}) ---")
            
            dataset = load_dataset(local_path, trust_remote_code=True)
            
            if name == "GTSRB":
                class_names = gtsrb_class_names
            else:
                class_names = dataset['train'].features['label'].names
            
            splits = split_mapping.get(name, split_mapping["default"])
            for split_name, hf_split_name in splits.items():
                if hf_split_name not in dataset:
                    print(f"Warning: Split '{hf_split_name}' not found in {name} dataset. Skipping this split.")
                    continue
                
                print(f"Processing {split_name} split...")
                split_dataset = dataset[hf_split_name]
                
                os.makedirs(os.path.join(output_root, name), exist_ok=True)
                
                for i, example in enumerate(tqdm(split_dataset, desc=f"Saving {split_name} split of {name}")):
                    image = example['image']
                    label = example['label']
                    
                    class_name_str = class_names[label]
                    class_dir = os.path.join(output_root, name, split_name, class_name_str)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    image_path_dist = os.path.join(class_dir, f"{i}.jpg")
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(image_path_dist)
                    
            print(f"{name} dataset organized successfully.")
        
        except Exception as e:
            print(f"Failed to organize {name} dataset. Error: {e}")

if __name__ == "__main__":
    cloned_datasets_root = "../../data/hf_cloned_datasets"
    
    organized_datasets_root = "../../data/hf_organized_datasets"
    organize_cloned_datasets(cloned_datasets_root, organized_datasets_root)