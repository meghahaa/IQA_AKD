import os
import torch


def save_checkpoint(
    save_dir,
    filename,
    model,
    optimizer,
    epoch,
    best_metric=None,
    verbose=True
):
    """
    Save training checkpoint.

    Args:
        save_dir (str): directory to save checkpoint
        filename (str): checkpoint file name
        model (nn.Module): model to save
        optimizer (Optimizer): optimizer state
        epoch (int): current epoch
        best_metric (float, optional): best validation metric
        verbose (bool): print status
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
    }

    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)

    if verbose:
        print(f"[Checkpoint] Saved: {path} (epoch {epoch})")


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
    verbose=True
):
    """
    Load training checkpoint.

    Args:
        path (str): checkpoint path
        model (nn.Module): model to load weights into
        optimizer (Optimizer, optional): optimizer to restore
        device (str): cpu or cuda
        verbose (bool): print status

    Returns:
        start_epoch (int)
        best_metric (float or None)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", None)

    if verbose:
        print(f"[Checkpoint] Loaded: {path}")
        print(f"[Checkpoint] Resuming from epoch {start_epoch}")

    return start_epoch, best_metric
