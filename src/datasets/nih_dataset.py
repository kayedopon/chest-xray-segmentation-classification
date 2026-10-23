from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


DISEASES = [
    "Atelectasis",
    "Consolidation",
    "Effusion",
    "Pneumothorax",
    "Infiltration",
    "Nodule",
]

class NIHChestXrayDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        self.image_paths = self.df["image_path"].tolist()

        self.labels = self.df[DISEASES].to_numpy(
            dtype=np.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = Path(self.image_paths[idx])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.from_numpy(self.labels[idx].copy())

        return image, labels