import random
import numpy as np
import torch

import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_segmentation_group(
    results,
    dataset,
    model,
    device,
    title
):
    fig, axes = plt.subplots(
        len(results),
        4,
        figsize=(14, 3 * len(results))
    )

    model.eval()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for row_idx, (_, result) in enumerate(results.iterrows()):
        idx = int(result["dataset_index"])

        image, mask = dataset[idx]

        with torch.no_grad():
            logits = model(
                image.unsqueeze(0).to(device)
            )

            prediction = (
                torch.sigmoid(logits) >= 0.5
            ).float().cpu()[0, 0].numpy()

        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np * std + mean, 0, 1)

        target = mask[0].numpy()

        error = np.abs(target - prediction)

        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_ylabel(
            f"Dice={result['dice']:.3f}",
            fontsize=10
        )

        axes[row_idx, 1].imshow(
            target, cmap="gray", vmin=0, vmax=1
        )

        axes[row_idx, 2].imshow(
            prediction, cmap="gray", vmin=0, vmax=1
        )

        axes[row_idx, 3].imshow(image_np)
        axes[row_idx, 3].imshow(
            np.ma.masked_where(error == 0, error),
            cmap="Reds",
            alpha=0.7,
            vmin=0,
            vmax=1
        )

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    for ax, label in zip(
        axes[0],
        ["Original X-ray", "CheXmask target",
         "U-Net prediction", "Prediction errors"]
    ):
        ax.set_title(label)

    fig.suptitle(title, fontsize=16, y=1.01)

    plt.tight_layout()
    plt.show()