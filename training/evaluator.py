import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import stats
from utils.metrics import compute_plcc, compute_srocc


def evaluate_model(
    model,
    test_loader,
    device,
    split_name="Test",
    verbose=True,
):
    """
    Full evaluation of IQA model.
    Computes PLCC and SRCC against MOS labels.

    Args:
        model:       TeacherModel or StudentModel (plain or DataParallel)
        test_loader: DataLoader yielding {"dist": ..., "mos": ...}
                     or {"ref": ..., "dist": ..., "mos": ...} for teacher
        device:      torch.device
        split_name:  string label for printing (e.g. "LIVE", "TID2013")
        verbose:     print results

    Returns:
        dict with keys: plcc, srcc, preds, targets
    """
    model.eval()

    preds_all  = []
    mos_all    = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc=f"[Eval] {split_name}", leave=True)

        for batch in loop:
            mos  = batch["mos"].to(device, non_blocking=True)

            # ── Teacher mode (ref + dist) ─────────────────────────────
            if "ref" in batch:
                ref  = batch["ref"].to(device, non_blocking=True)
                dist = batch["dist"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    preds = model(ref, dist)

            # ── Student / test mode (dist only) ───────────────────────
            else:
                dist = batch["dist"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    preds = model(dist)

            preds_all.append(preds.cpu())
            mos_all.append(mos.cpu())

    # Concatenate all batches
    preds_all = torch.cat(preds_all).float().numpy()   # (N,)
    mos_all   = torch.cat(mos_all).float().numpy()     # (N,)

    # PLCC — Pearson linear correlation
    plcc = compute_plcc(preds_all, mos_all)

    # SRCC — Spearman rank correlation
    srocc = compute_srocc(preds_all, mos_all)

    if verbose:
        print(f"\n{'='*45}")
        print(f"  Evaluation Results — {split_name}")
        print(f"{'='*45}")
        print(f"  PLCC : {plcc:.4f}")
        print(f"  SROCC : {srocc:.4f}")
        print(f"  Samples evaluated: {len(preds_all)}")
        print(f"{'='*45}\n")

    return {
        "plcc":    plcc,
        "srocc":   srocc,
        "preds":   preds_all,
        "targets": mos_all,
    }


def evaluate_on_multiple_datasets(
    model,
    loaders_dict,
    device,
):
    """
    Evaluate model on multiple datasets and print a summary table.

    Args:
        model:        trained model
        loaders_dict: dict mapping dataset name → DataLoader
                      e.g. {"LIVE": live_loader, "TID2013": tid_loader}
        device:       torch.device

    Returns:
        results_dict: dict mapping dataset name → {plcc, srcc, preds, targets}
    """
    results = {}

    for name, loader in loaders_dict.items():
        results[name] = evaluate_model(
            model=model,
            test_loader=loader,
            device=device,
            split_name=name,
            verbose=True,
        )

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"  Summary")
    print(f"{'='*45}")
    print(f"  {'Dataset':<15} {'PLCC':>8} {'SRCC':>8}")
    print(f"  {'-'*35}")

    plcc_vals = []
    srocc_vals = []

    for name, res in results.items():
        print(f"  {name:<15} {res['plcc']:>8.4f} {res['srcc']:>8.4f}")
        plcc_vals.append(res["plcc"])
        srocc_vals.append(res["srocc"])

    avg_plcc = sum(plcc_vals) / len(plcc_vals)
    avg_srocc = sum(srocc_vals) / len(srocc_vals)

    print(f"  {'-'*35}")
    print(f"  {'Average':<15} {avg_plcc:>8.4f} {avg_srocc:>8.4f}")
    print(f"{'='*45}\n")

    return results