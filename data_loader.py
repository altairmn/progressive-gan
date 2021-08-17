from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torchvision
from PIL import Image
import torch
import os
import random
import pandas as pd
from typing import Any, Callable, List, Optional, Union, Tuple

def get_loader(root_dir, image_size=1024, 
               batch_size=16, split='train', num_workers=1, crop_size=178):
    """Build and return a data loader."""
    transform = []
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    dataset = torchvision.datasets.CelebA(
            root = root_dir,
            split = split,
            transform = transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(split=='train'),
                                  num_workers=num_workers)
    return data_loader
