import glob
import os.path as osp
from typing import Callable, Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import cv2
from collections import defaultdict
from torch.nn import functional as F
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.sam import Sam
from typing import Any, Dict, List, Tuple
import json



class SamDataset(Dataset):
    def __init__(self, root_folder: str, dataset_size, test=False):
        self.test = test
        self.dataset_size = dataset_size
        self._root_folder = root_folder
        self._image_paths = sorted(glob.glob(osp.join(root_folder, "*.jpg")))
        self._json_paths = sorted(glob.glob(osp.join(root_folder, "*.json")))
        self.transform = ResizeLongestSide(1024)
        self.sam = sam_model_registry['vit_b'](checkpoint=None)

    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index):

        if not self.test: 
            image = cv2.imread(self._image_paths[index+1000])
        elif self.test:
            image = cv2.imread(self._image_paths[index])
        
        if not self.test: 
            annot = self._json_paths[index+1000]
        elif self.test:
            annot = self._json_paths[index]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()#[None, :, :, :]
        input_image = self.sam.preprocess(transformed_image)

        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])


        if not self.test:
            return {
            "id": self._image_paths[index+1000],
            "input_image": input_image,
            "input_size":input_size,
            "original_image_size":original_image_size,
            "annot":annot
        }
        return {
            "id": self._image_paths[index],
            "input_image": input_image,
            "input_size":input_size,
            "original_image_size":original_image_size,
            "annot":annot
        }

