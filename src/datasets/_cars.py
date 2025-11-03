import os
import torch
import torchvision.datasets as datasets

import pathlib
from typing import Callable, Optional, Any, Tuple

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset

## Override VisionDataset to fit our GenericDataset structure
class PytorchStanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")
        
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        ### is train or test split?
        self._split = verify_str_arg(split, "split", ("train", "test"))
        ### base directory for the dataset
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"
        
        ### specify annotation path and image path according to split
        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = devkit / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"
        
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        " Most important part of this class: load annotations and prepare samples "
        " Returns (image path, target(correct label)) tuples "
        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1 # Original target mapping starts from 1, hence -1
            )
            ### open .mat file and read annotations
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]
        
        ### load class names by opening cars_meta.mat file
        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        ### create dictionary mapping class name to index
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def __len__(self) -> int:
        return len(self._samples)
    
    ## Called by DataLoader during iteration
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[index]
        pil_image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target
        
    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )
            
    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        print(self._annotations_mat_path)
        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

"""Final Interface for Users"""
class Cars:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=0):
        # Data loading code
        
        self.train_dataset = PytorchStanfordCars(location, 'train', preprocess, download=False)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        self.test_dataset = PytorchStanfordCars(location, 'test', preprocess, download=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]