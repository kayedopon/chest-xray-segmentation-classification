import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def classification_metrics(
    y_true,
    y_prob,
    class_names,
    threshold=0.5
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    results = []

    for i, disease in enumerate(class_names):
        true = y_true[:, i]
        prob = y_prob[:, i]
        pred = y_pred[:, i]

        tn, fp, fn, tp = confusion_matrix(
            true, pred, labels=[0, 1]
        ).ravel()

        auroc = (
            roc_auc_score(true, prob)
            if len(np.unique(true)) == 2
            else np.nan
        )

        auprc = (
            average_precision_score(true, prob)
            if np.any(true == 1)
            else np.nan
        )

        sensitivity = recall_score(
            true, pred, zero_division=0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0 else np.nan
        )

        f1 = f1_score(
            true, pred, zero_division=0
        )

        results.append({
            "Disease": disease,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1": f1
        })

    return pd.DataFrame(results)

import torch


@torch.no_grad()
def evaluate_classifier(model, loader, device, class_names):
    model.eval()

    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.float16
        ):
            logits = model(images)

        probs = torch.sigmoid(logits.float())

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    return classification_metrics(
        y_true,
        y_prob,
        class_names
    )