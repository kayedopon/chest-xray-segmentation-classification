import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Pixel-level classification loss
        bce_loss = self.bce(logits, targets)

        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Calculate soft Dice coefficient per image
        dims = (1, 2, 3)

        intersection = (probabilities * targets).sum(dim=dims)

        denominator = (
            probabilities.sum(dim=dims)
            + targets.sum(dim=dims)
        )

        smooth = 1e-6

        dice = (2 * intersection + smooth) / (
            denominator + smooth
        )

        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss