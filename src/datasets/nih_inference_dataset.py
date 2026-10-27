from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NIHInferenceDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = Path(self.df.iloc[idx]["image_path"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, image_path.name