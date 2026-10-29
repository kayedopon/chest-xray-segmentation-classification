
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from src.datasets.nih_dataset import DISEASES


class NIHFourChannelDataset(Dataset):
    def __init__(self, csv_path, mask_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.mask_dir = Path(mask_dir)
        self.augment = augment

        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df[DISEASES].values.astype(np.float32)

        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = Path(self.image_paths[idx])
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(
            image, [224, 224],
            interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask, [224, 224],
            interpolation=InterpolationMode.NEAREST
        )

        if self.augment:
            # Apply identical geometry to image and mask
            if random.random() < 0.8:
                angle, translations, scale, shear = (
                    transforms.RandomAffine.get_params(
                        degrees=(-7, 7),
                        translate=(0.05, 0.05),
                        scale_ranges=(0.95, 1.05),
                        shears=None,
                        img_size=image.size
                    )
                )

                image = TF.affine(
                    image, angle, translations, scale, shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0
                )

                mask = TF.affine(
                    mask, angle, translations, scale, shear,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0
                )

            # Intensity augmentation affects only the X-ray
            if random.random() < 0.5:
                image = self.color_jitter(image)

        image = TF.to_tensor(image)
        mask = (TF.to_tensor(mask) > 0.5).float()

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Add mask as fourth channel
        x = torch.cat([image, mask], dim=0)

        labels = torch.tensor(
            self.labels[idx], dtype=torch.float32
        )

        return x, labels
