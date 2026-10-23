from pathlib import Path

import torch
import mlflow
import time
from tqdm.auto import tqdm
import numpy as np

from src.evaluation.metrics import classification_metrics


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
    
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16
        ):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs.float())

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    return val_loss, y_true, y_prob

def fit(model, train_loader, val_loader, loss_fn, 
        optimizer, device, epochs, class_names, checkpoint_path=None):
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }
    start = time.perf_counter()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device)

        val_loss, y_true, y_prob = validate_one_epoch(
            model, val_loader, loss_fn, device
        )

        metrics_df = classification_metrics(
            y_true,
            y_prob,
            class_names
        )

        macro_metrics = metrics_df[
            ["AUROC", "AUPRC", "Sensitivity", "Specificity", "F1"]
        ].mean()
        

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_macro_auroc": macro_metrics["AUROC"],
                    "val_macro_auprc": macro_metrics["AUPRC"],
                    "val_macro_sensitivity": macro_metrics["Sensitivity"],
                    "val_macro_specificity": macro_metrics["Specificity"],
                    "val_macro_f1": macro_metrics["F1"]
                },
                step=epoch + 1
            )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Validation: {val_loss:.4f}"
        )

        if checkpoint_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss

            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_loss": val_loss
                },
                path
            )

            print("Best checkpoint saved.")

    elapsed = time.perf_counter() - start
    print(f"Time taken {elapsed}")


    return history