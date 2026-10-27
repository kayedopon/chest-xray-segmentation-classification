
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.datasets.nih_dataset import DISEASES


class NIHCroppedDataset(Dataset):
    def __init__(self, csv_path, mask_dir, transform=None, margin=0.10):
        self.df = pd.read_csv(csv_path)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.margin = margin

        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df[DISEASES].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = Path(self.image_paths[idx])
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # U-Net masks are saved at 224x224.
        # Resize the original X-ray into the same coordinate system.
        image = TF.resize(
            image,
            [224, 224],
            interpolation=InterpolationMode.BILINEAR
        )

        mask_array = np.asarray(mask) > 127

        ys, xs = np.where(mask_array)

        if len(xs) > 0:
            x_min, x_max = xs.min(), xs.max() + 1
            y_min, y_max = ys.min(), ys.max() + 1

            width = x_max - x_min
            height = y_max - y_min

            # Add 10% padding on each side
            pad_x = int(width * self.margin)
            pad_y = int(height * self.margin)

            left = max(0, x_min - pad_x)
            upper = max(0, y_min - pad_y)
            right = min(224, x_max + pad_x)
            lower = min(224, y_max + pad_y)

            image = image.crop((left, upper, right, lower))

        # If U-Net predicts an empty mask, retain the entire X-ray.

        if self.transform is not None:
            image = self.transform(image)

        else:
            image = TF.resize(
                image,
                [224, 224],
                interpolation=InterpolationMode.BILINEAR
            )
            image = TF.to_tensor(image)

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, labels
