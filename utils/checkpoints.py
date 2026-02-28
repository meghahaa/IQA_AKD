import os
import torch
import torch.nn as nn


def save_checkpoint(
    save_dir,
    filename,
    model,
    optimizer,
    epoch,
    akd_loss_fn=None,  # ← add this argument
    best_metric=None,
    verbose=True
):
    """
    Save training checkpoint.
    Handles DataParallel wrapper automatically — always saves
    the underlying model state dict without 'module.' prefix.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Unwrap DataParallel if present — saves clean state dict
    model_state = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )

    # But omega lives in akd_loss_fn, so save it separately:
    checkpoint = {
        "epoch":           epoch,
        "model_state":     model_state,
        "optimizer_state": optimizer.state_dict(),
        "akd_loss_state":  akd_loss_fn.state_dict() if akd_loss_fn is not None else None,
        "best_metric":     best_metric,
    }

    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)

    if verbose:
        print(f"[Checkpoint] Saved: {path} (epoch {epoch})")


def load_checkpoint(
    path,
    model,
    optimizer=None,
    akd_loss_fn=None,  # ← add this argument
    device="cpu",
    verbose=True
):
    """
    Load training checkpoint.
    Works regardless of whether the model is currently wrapped
    in DataParallel or not — checkpoint always stores clean keys.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device,weights_only=False)

    # Unwrap DataParallel before loading if present
    target_model = (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )
    target_model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if akd_loss_fn is not None and checkpoint["akd_loss_state"] is not None:
        akd_loss_fn.load_state_dict(checkpoint["akd_loss_state"])

    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", None)

    if verbose:
        print(f"[Checkpoint] Loaded: {path}")
        print(f"[Checkpoint] Resuming from epoch {start_epoch}")

    return start_epoch, best_metric