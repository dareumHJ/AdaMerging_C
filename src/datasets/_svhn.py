import os
import torch
import torchvision.datasets as datasets

import pathlib
from typing import Callable, Optional, Any, Tuple

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset

