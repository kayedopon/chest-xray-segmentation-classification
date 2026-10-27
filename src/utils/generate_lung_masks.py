from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


@torch.inference_mode()
def generate_lung_masks(model, loader, device, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    for images, image_ids in tqdm(loader, desc="Generating masks"):
        images = images.to(device, non_blocking=True)

        logits = model(images)
        masks = (torch.sigmoid(logits) >= 0.5).to(torch.uint8)

        masks = masks.cpu().numpy()

        for mask, image_id in zip(masks, image_ids):
            mask = mask[0] * 255

            Image.fromarray(mask).save(
                output_dir / image_id
            )

            saved += 1

    print(f"Saved {saved} masks to {output_dir}")