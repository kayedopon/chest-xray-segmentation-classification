from pathlib import Path

import torch
from tqdm.auto import tqdm


def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16
        ):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = images.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16
        ):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

def fit(model, train_loader, val_loader, loss_fn, 
        optimizer, device, epochs, checkpoint_path=None):
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler
        )

        val_loss = validate_one_epoch(
            model, val_loader, loss_fn, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

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

    return history