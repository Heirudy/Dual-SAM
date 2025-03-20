import itertools

import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from PIL import Image
import argparse
import numpy as np
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
import glob
import os
import re
import matplotlib.pyplot as plt
import tifffile as tf
import random
import torchvision.utils as vutils
import logging
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import Sampler
class SegDataset:
    def __init__(self, img_paths, mask_paths,
                 # transform=None,
                 # ops_weak=None,
                 # ops_strong=None,

                 mask_divide=False, divide_value=255,
                 pixel_mean=[0.5] * 3, pixel_std=[0.5] * 3,
                 img_size=1024) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.length = len(img_paths)
        self.mask_divide = mask_divide
        self.divide_value = divide_value
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_size = img_size
        # self.transform = transform
        # self.ops_weak=ops_weak
        # self.ops_strong=ops_strong
        # self.bbox_shift = bbox_shift

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(img_path).convert("RGB")
        # img = tf.imread(img_path)
        mask = Image.open(mask_path).convert("L")

        img = np.asarray(img)
        mask = np.asarray(mask)
        if self.mask_divide:
            mask = mask // self.divide_value
        # sample = {"image": img, "label": mask}
        # if None not in (self.ops_weak, self.ops_strong):
        #         sample = self.transform(sample, self.ops_weak, self.ops_strong)
        # else:
        #         sample = self.transform(sample)
        transform = Compose(
            [
                ColorJitter(),
                VerticalFlip(),
                HorizontalFlip(),
                Resize(self.img_size, self.img_size),
                Normalize(mean=self.pixel_mean, std=self.pixel_std)
            ]
        )

        aug_data = transform(image=img, mask=mask)
        x = aug_data["image"]
        target = aug_data["mask"]
        if img.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif img.ndim == 2:
            x = np.expand_dims(x, axis=0)
        sample = {"image": torch.from_numpy(x), "label": torch.from_numpy(target)}

        return sample



    # def __len__(self):
    #     return len(self.primary_indices) // self.primary_batch_size

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



