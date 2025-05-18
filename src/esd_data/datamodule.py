""" This module contains the PyTorch Lightning ESDDataModule to use with the
PyTorch ESD dataset."""

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch import Generator
from torch.utils.data import DataLoader, random_split
import torch
from .dataset import DSE
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ..preprocessing.subtile_esd_hw02 import grid_slice
from ..preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_viirs,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
)
from ..preprocessing.file_utils import (
    load_satellite
)
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor
)
from torchvision import transforms
from copy import deepcopy
from typing import List, Tuple, Dict
from src.preprocessing.file_utils import Metadata


def collate_fn(batch):
    Xs = []
    ys = []
    metadatas = []
    for X, y, metadata in batch:
        Xs.append(torch.from_numpy(X) if isinstance(X, np.ndarray) else X)
        ys.append(torch.from_numpy(y) if isinstance(y, np.ndarray) else y)
        metadatas.append(metadata)
    Xs = torch.stack(Xs)
    ys = torch.stack(ys)
    return Xs, ys, metadatas


class ESDDataModule(pl.LightningDataModule):
    """
        PyTorch Lightning ESDDataModule to use with the PyTorch ESD dataset.

        Attributes:
            processed_dir: str | os.PathLike
                Location of the processed data
            raw_dir: str | os.PathLike
                Location of the raw data
            selected_bands: Dict[str, List[str]] | None
                Dictionary mapping satellite type to list of bands to select
            tile_size_gt: int
                Size of the ground truth tiles
            batch_size: int
                Batch size
            seed: int
                Seed for the random number generator
    """
    def __init__(
            self,
            processed_dir: str | os.PathLike,
            raw_dir: str | os.PathLike,
            selected_bands: Dict[str, List[str]] | None = None,
            tile_size_gt=4,
            batch_size=32,
            num_workers=0,
            seed=12378921):
        super().__init__()
        # set transform to a composition of the following transforms:
        # AddNoise, Blur, RandomHFlip, RandomVFlip, ToTensor
        # utilize the RandomApply transform to apply each of the transforms
        # with a probability of 0.5
        self.transform = transforms.Compose([
            transforms.RandomApply([AddNoise(), Blur(), RandomHFlip(), RandomVFlip()], p=0.5),
            ToTensor()
        ])

        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.selected_bands = selected_bands
        self.tile_size_gt = tile_size_gt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def __load_and_preprocess(
            self,
            tile_dir: str | os.PathLike,
            satellite_types: List[str] = ["viirs", "sentinel1", "sentinel2", "landsat", "gt"]
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Metadata]]]:
        """
            Performs the preprocessing step: for a given tile located in tile_dir,
            loads the tif files and preprocesses them just like in homework 1.

            Input:
                tile_dir: str | os.PathLike
                    Location of raw tile data
                satellite_types: List[str]
                    List of satellite types to process

            Output:
                satellite_stack: Dict[str, np.ndarray]
                    Dictionary mapping satellite_type -> (time, band, width, height) array
                satellite_metadata: Dict[str, List[Metadata]]
                    Metadata accompanying the statellite_stack
        """
        preprocess_functions = {
            "viirs": preprocess_viirs,
            "sentinel1": preprocess_sentinel1,
            "sentinel2": preprocess_sentinel2,
            "landsat": preprocess_landsat,
            "gt": lambda x: x
        }

        satellite_stack = {}
        satellite_metadata = {}
        
        # We use the default value of satellite_types since selected_bands selects the bands we want to use
        for satellite_type in satellite_types:
            stack, metadata = load_satellite(tile_dir, satellite_type)

            stack = preprocess_functions[satellite_type](stack)

            satellite_stack[satellite_type] = stack
            satellite_metadata[satellite_type] = metadata

        satellite_stack["viirs_maxproj"] = np.expand_dims(maxprojection_viirs(satellite_stack["viirs"], clip_quantile=0.0), axis=0)
        satellite_metadata["viirs_maxproj"] = deepcopy(satellite_metadata["viirs"])
        for metadata in satellite_metadata["viirs_maxproj"]:
            metadata.satellite_type = "viirs_maxproj"

        return satellite_stack, satellite_metadata

    def prepare_data(self):
        """
            If the data has not been processed before (denoted by whether or not self.processed_dir is an existing directory)

            Split the raw data into a training and validation sets
            For each tile in the training set and validation set, do the following:
                - load and preprocess the data in the tile
                - grid slice the data
                - for each resulting subtile
                    - save the subtile data to self.processed_dir/gt_subtile_size_dim/[Train or Val]
        """
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
            
            parent_images = os.listdir(self.raw_dir)
            
            train_images, val_images = train_test_split(parent_images, test_size=0.2, random_state=self.seed, shuffle=True)
            
            train_dir = os.path.join(self.processed_dir.parent, f"{self.tile_size_gt}x{self.tile_size_gt}", "Train")
            val_dir = os.path.join(self.processed_dir.parent, f"{self.tile_size_gt}x{self.tile_size_gt}", "Val")
            
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            for parent_image in tqdm(train_images, desc="Train Images"):
                satellite_stack, satellite_metadata = self.__load_and_preprocess(os.path.join(self.raw_dir, parent_image))
                subtile_stack = grid_slice(satellite_stack, satellite_metadata, self.tile_size_gt)
                for subtile in subtile_stack:
                    subtile.save(train_dir)
            
            for parent_image in tqdm(val_images, desc="Validation Images"):
                satellite_stack, satellite_metadata = self.__load_and_preprocess(os.path.join(self.raw_dir, parent_image))
                subtile_stack = grid_slice(satellite_stack, satellite_metadata, self.tile_size_gt)
                for subtile in subtile_stack:
                    subtile.save(val_dir)

    def setup(self, stage: str):
        """
            Create self.train_dataset and self.val_dataset.0000ff

            Hint: Use torch.utils.data.random_split to split the Train
            directory loaded into the PyTorch dataset DSE into an 80% training
            and 20% validation set. Set the seed to 1024.
        """
        self.train_dataset = DSE(self.processed_dir / "Train" / "subtiles", self.selected_bands, self.transform)
        self.val_dataset = DSE(self.processed_dir / "Val" / "subtiles", self.selected_bands, self.transform)

    def train_dataloader(self):
        """
            Create and return a torch.utils.data.DataLoader with
            self.train_dataset
        """
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                                           num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def val_dataloader(self):
        """
            Create and return a torch.utils.data.DataLoader with
            self.val_dataset
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
                                           num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))
