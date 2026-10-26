from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.datasets.chexmask import decode_lung_mask


class NIHLungSegmentationDataset(Dataset):
    def __init__(self, split_csv, annotations_df, image_size=224):
        self.df = pd.read_csv(split_csv).copy()
        self.image_size = image_size

        self.df["Image Index"] = self.df["image_path"].map(
            lambda p: Path(p).name
        )

        self.df = self.df.merge(
            annotations_df,
            on="Image Index",
            how="inner",
            validate="one_to_one"
        )

        if len(self.df) != len(pd.read_csv(split_csv)):
            raise ValueError("Some images have no matching masks")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Original chest X-ray
        image = Image.open(row["image_path"]).convert("RGB")

        # Decode and combine both lungs
        mask = decode_lung_mask(row)
        mask = Image.fromarray(mask * 255, mode="L")

        # Resize image and mask
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )

        # Image tensor: 3 x H x W, normalized like ImageNet
        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Mask tensor: 1 x H x W, containing 0 and 1
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask