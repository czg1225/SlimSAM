import glob
import os.path as osp
from torch.utils.data import Dataset
import torch
import cv2
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from typing import Any, Dict, List, Tuple



class SamDataset(Dataset):
    def __init__(self, root_folder: str, dataset_size, val=False):
        self.val = val
        self.dataset_size = dataset_size
        self._root_folder = root_folder
        self._image_paths = sorted(glob.glob(osp.join(root_folder, "*.jpg")))
        self._json_paths = sorted(glob.glob(osp.join(root_folder, "*.json")))
        self.transform = ResizeLongestSide(1024)
        self.sam = sam_model_registry['vit_b'](checkpoint=None)
        # self.sam = Sam(image_encoder=None, prompt_encoder=None, mask_decoder=None, pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375])

    def __len__(self):
        return self.dataset_size
    

    def __getitem__(self, index):

        if not self.val: 
            image = cv2.imread(self._image_paths[index])
        elif self.val:
            image = cv2.imread(self._image_paths[index])
        
        if not self.val: 
            annot = self._json_paths[index]
        elif self.val:
            annot = self._json_paths[index]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()#[None, :, :, :]
        input_image = self.sam.preprocess(transformed_image)

        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])


        if not self.val:
            return {
            "id": self._image_paths[index],
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

