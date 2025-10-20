""" Manage all dataset classes used in the project. (dataset vending machine)"""

import sys
import inspect
import random
import torch
import copy

from torch.utils.data import random_split

from datasets.cars import Cars
from datasets.dtd import DTD
from datasets.eurosat import EuroSAT, EuroSATVal
from datasets.gtsrb import GTSRB
from datasets.mnist import MNIST
from datasets.resisc45 import RESISC45
from datasets.svhn import SVHN
from datasets.sun397 import SUN397

## Save all class names and their objects imported in this registry.py file
## Important for this file's scalability and extensibility
registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}

class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None

## Split train dataset into **train** and **val** subsets
## Why needed? -> During finetuning, we want to have a validation set to select best model checkpoint
### OR monitor overfitting
def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    ## Sanity check for MNISTVal dataset (for reproducibility)
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044


    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    new_dataset.classnames = copy.deepcopy(dataset.classnames)

    return new_dataset
    
    """
    dataset_name: str, name of the dataset to be fetched (e.g., 'CIFAR10', 'MNIST', 'EuroSATVal', etc.)
    preprocess: torchvision.transforms, preprocessing transformations to be applied
    location: str, path to store/load the dataset
    batch_size: int, batch size for data loaders
    num_workers: int, number of workers for data loaders
    val_fraction: float, fraction of training data to be used as validation set (only for Val datasets)
    max_val_samples: int, maximum number of samples in the validation set (only for Val datasets)
    val_fraction이랑 max_val_samples는 학습 데이터 중 검증 데이터로 얼마나 나눌지를 결정하는 인자.
    """
def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=0, val_fraction=0.1, max_val_samples=5000):
    if dataset_name.endwith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        ## assign dataset class object from registry
        dataset_class = registry[dataset_name]
    ## instantiate dataset class (generate real dataset object)
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    return dataset
    