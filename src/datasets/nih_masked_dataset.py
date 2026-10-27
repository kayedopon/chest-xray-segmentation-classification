
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.datasets.nih_dataset import DISEASES


class NIHMaskedDataset(Dataset):
    def __init__(self, csv_path, mask_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df[DISEASES].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = Path(self.image_paths[idx])
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Both image and mask must have the same size.
        image = TF.resize(
            image, [224, 224],
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask, [224, 224],
            interpolation=InterpolationMode.NEAREST
        )

        # Keep only pixels inside predicted lungs.
        image_array = np.asarray(image).copy()
        mask_array = np.asarray(mask) > 127

        image_array[~mask_array] = 0
        image = Image.fromarray(image_array)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx])
