"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

# Dataset code for the DDP training setting.

from io import BytesIO
#import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torch.utils import data
import numpy as np
import random
import re, os
from torchvision import transforms
import torch


class MultiResolutionDataset(Dataset):

    def __init__(self, path, transform, resolution=256):
        self.path = path
        self.resolution = resolution
        self.transform = transform
        self.length = None

    def _open(self): #open and load image path files
        source_root = self.path[:self.path.rfind('/')]
        with open(self.path) as f:
            self.source_paths = f.read().splitlines()
        self.source_paths = [os.path.join(source_root, s) for s in self.source_paths]
        self.length = (len(self.source_paths))
        #print('LEN PATHS: ' + str(self.length))


    def __len__(self):
        if self.length is None:
            self._open()
        return self.length

    def __getitem__(self, index):
        if self.length is None:
            self._open()

        from_path = self.source_paths[index]
        img = Image.open(from_path)
        img = img.convert('RGB')

        img = self.transform(img)

        return img
