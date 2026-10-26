import torch
import mlflow

from src.evaluation.segmentation_metrics import segmentation_metrics

from tqdm.auto import tqdm


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    progress_bar = tqdm(
        loader,
        desc="Training" if training else "Validation",
        leave=False,
        unit="batch"
    )

    for images, masks in progress_bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        metrics = segmentation_metrics(logits.detach(), masks)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_dice += metrics["dice"] * batch_size
        total_iou += metrics["iou"] * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }


def fit_segmentation(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    epochs,
    checkpoint_path
):
    best_dice = -1.0
    history = []

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )

        val_metrics = run_epoch(
            model, val_loader, loss_fn, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        })

        mlflow.log_metrics({
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }, step=epoch)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}",
            flush=True
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_dice,
            }, checkpoint_path)

            print("Best checkpoint saved.", flush=True)

    return history