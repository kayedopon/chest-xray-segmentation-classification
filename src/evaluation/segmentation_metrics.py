import torch


@torch.no_grad()
def segmentation_metrics(logits, targets, threshold=0.5):
    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= threshold).float()

    dims = (1, 2, 3)

    intersection = (predictions * targets).sum(dim=dims)

    predicted_area = predictions.sum(dim=dims)
    target_area = targets.sum(dim=dims)

    union = predicted_area + target_area - intersection

    eps = 1e-6

    dice = (
        (2 * intersection + eps)
        / (predicted_area + target_area + eps)
    )

    iou = (intersection + eps) / (union + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item()
    }