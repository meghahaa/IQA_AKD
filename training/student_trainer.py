import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from models.backbone.swin import SwinBackbone
from models.mfr import MultiScaleFeatureRepresentation
from models.mfde import MFDE
from models.cfi import CFI
from models.caf import CAF
from models.regressor import QualityRegressor

from training.losses import ScoreLoss
from utils.metrics import IQAMetrics
from utils.checkpoints import save_checkpoint

class StudentModel(nn.Module):
    """
    No-Reference IQA Student Model (AKD-IQA)

    Pipeline:
        1. Flatten (B, N, 3, H, W) → (B*N, 3, H, W)
        2. Batched backbone: [dist, redist] → 2 × 4 feature maps
        3. MFR → 4 levels × (B*N, 49, 256) per branch
        4. Diff tokens: redist - dist per level (pseudo-reference)
           Note: redist is MORE degraded than dist, so redist-dist
           gives a stable negative-direction difference signal
        5. MFDE(diff per level) → f_diff: list 4 × (B*N, 49, 256)
           + intermediates [4][depth] for AKD loss
        6. CFI(dist_tokens concatenated) → f_dist: (B*N, 196, 256)
        7. CAF(f_diff_cat, f_dist) → fused: (B*N, 196, 256)
        8. Regressor → score: (B,)
    """

    def __init__(
        self,
        embed_dim=256,
        mfde_depth=12,
        cfi_depth=6,
        num_patches=10,
        verbose=False,
    ):
        super().__init__()

        self.verbose     = verbose
        self.num_patches = num_patches

        # Shared backbone — same weights for dist and redist branches
        self.backbone = SwinBackbone(pretrained=True, verbose=verbose)

        self.mfr = MultiScaleFeatureRepresentation(
            in_channels=self.backbone.out_channels,  # [96, 192, 384, 768]
            embed_dim=embed_dim,
            target_spatial_dim=7,
            verbose=verbose,
        )

        # Student MFDE: depth=18 (half of teacher's 36)
        self.mfde = MFDE(
            embed_dim=embed_dim,
            depth=mfde_depth,
            verbose=verbose,
        )

        # Student CFI: depth=9 (half of teacher's 18)
        self.cfi = CFI(
            num_tokens=196,
            embed_dim=embed_dim,
            depth=cfi_depth,
            verbose=verbose,
        )

        self.caf       = CAF(embed_dim=embed_dim, verbose=verbose)
        self.regressor = QualityRegressor(embed_dim=embed_dim, verbose=verbose)

        if self.verbose:
            print(
                f"[StudentModel] Initialized | "
                f"mfde_depth={mfde_depth} | cfi_depth={cfi_depth}"
            )

    def forward(self, dist_patches, redist_patches, store_intermediates=False):
        """
        Args:
            dist_patches:        (B, N, 3, H, W)
            redist_patches:      (B, N, 3, H, W)
            store_intermediates: if True, return MFDE intermediates for AKD loss
                                 set True during training, False during inference

        Returns:
            score:         (B,)
            intermediates: [4][mfde_depth] each (B*N, 49, 256) — only if store_intermediates
        """
        B, N, C, H, W = dist_patches.shape
        BN = B * N

        # ── 1. Flatten ───────────────────────────────────────────────────────
        dist   = dist_patches.view(BN, C, H, W)    # (B*N, 3, 224, 224)
        redist = redist_patches.view(BN, C, H, W)  # (B*N, 3, 224, 224)

        # ── 2. Batched backbone — 1 forward pass for both branches ───────────
        both          = torch.cat([dist, redist], dim=0)   # (2*B*N, 3, 224, 224)
        both_backbone = self.backbone(both)
        # Split back: each is list of 4 × (B*N, C_i, H_i, W_i)
        # dist_backbone  = [f[:BN] for f in both_backbone]
        # redist_backbone = [f[BN:] for f in both_backbone]

        # ── 3. Batched MFR ───────────────────────────────────────────────────
        both_mfr   = self.mfr(both_backbone)
        # both_mfr: list of 4 × (2*B*N, 49, 256)
        dist_feats  = [f[:BN] for f in both_mfr]   # 4 × (B*N, 49, 256)
        redist_feats = [f[BN:] for f in both_mfr]  # 4 × (B*N, 49, 256)

        # ── 4. Pseudo-difference tokens (redist - dist) ──────────────────────
        diff_feats = [r - d for r, d in zip(redist_feats, dist_feats)]
        # 4 × (B*N, 49, 256)

        # ── 5. MFDE — per-level independent processing ───────────────────────
        f_diff_levels, intermediates = self.mfde(
            diff_feats,
            store_selected_only=False,   # student stores ALL layers for KD
        )
        # f_diff_levels: 4 × (B*N, 49, 256)
        # intermediates: [4][mfde_depth] each (B*N, 49, 256)

        # ── 6. CFI — cross-scale integration on dist tokens ──────────────────
        dist_tokens = torch.cat(dist_feats, dim=1)  # (B*N, 196, 256)
        f_dist      = self.cfi(dist_tokens)          # (B*N, 196, 256)

        # ── 7. CAF — cross-attention fusion ──────────────────────────────────
        f_diff = torch.cat(f_diff_levels, dim=1)    # (B*N, 196, 256)
        fused  = self.caf(f_diff, f_dist)           # (B*N, 196, 256)

        # ── 8. Regressor ─────────────────────────────────────────────────────
        score = self.regressor(fused, N)             # (B,)

        if store_intermediates:
            return score, intermediates
        return score, None


# =========================================================
# Training Function
# =========================================================

def train_student(
    student,
    teacher,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_dir,
    akd_loss_fn,
    print_freq=10,
):
    # Model already on device and optionally DataParallel-wrapped
    # before being passed in

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    score_loss_fn = ScoreLoss()
    akd_loss_fn   = akd_loss_fn.to(device)

    scaler    = GradScaler()
    best_plcc = -1e9

    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss       = 0.0
        epoch_score_loss = 0.0
        epoch_akd_loss   = 0.0

        print(f"\n[Student] Epoch {epoch}/{epochs}")
        loop = tqdm(train_loader, desc="Training")

        for i, batch in enumerate(loop):
            ref    = batch["ref"].to(device, non_blocking=True)     # (B, N, 3, H, W)
            dist   = batch["dist"].to(device, non_blocking=True)    # (B, N, 3, H, W)
            redist = batch["redist"].to(device, non_blocking=True)  # (B, N, 3, H, W)
            mos    = batch["mos"].to(device, non_blocking=True)     # (B,)

            optimizer.zero_grad(set_to_none=True)

            # ── Teacher forward (frozen, no grad) ────────────────────────────
            # Teacher uses actual ref-dist difference (not redist)
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    teacher_intermediates = _teacher_forward_for_kd(
                        teacher=teacher,
                        ref_patches=ref,
                        dist_patches=dist,
                        device=device,
                    )
                # teacher_intermediates: [4][num_student_layers] — pre-selected,
                # detached, already on device

            # ── Student forward ───────────────────────────────────────────────
            with torch.amp.autocast("cuda"):
                pred, student_intermediates = student(
                    dist_patches=dist,
                    redist_patches=redist,
                    store_intermediates=True,
                )
                # pred:                  (B,)
                # student_intermediates: [4][mfde_depth]

                # ── Losses ───────────────────────────────────────────────────
                l_score = score_loss_fn(pred, mos)
                l_akd   = akd_loss_fn(
                    teacher_intermediates=teacher_intermediates,
                    student_intermediates=student_intermediates,
                )
                loss = l_score + l_akd

            if i % 10 == 0:  # every 10 iterations to avoid spam
                print(f"\n[Debug] omega_raw values: {akd_loss_fn.omega_raw.data}")
                print(f"[Debug] omega_raw grad:   {akd_loss_fn.omega_raw.grad}")
                print(f"[Debug] omega_raw requires_grad: {akd_loss_fn.omega_raw.requires_grad}")
                print(f"[Debug] omega_raw device: {akd_loss_fn.omega_raw.device}")
                
                # Check if optimizer knows about omega_raw
                param_ids_in_optimizer = {id(p) for group in optimizer.param_groups 
                                        for p in group['params']}
                print(f"[Debug] omega_raw in optimizer: {id(akd_loss_fn.omega_raw) in param_ids_in_optimizer}")
                
                # Check l_akd has grad_fn
                print(f"[Debug] l_akd grad_fn: {l_akd.grad_fn}")
                print(f"[Debug] l_score grad_fn: {l_score.grad_fn}")

            # scaler.step(optimizer)
            # scaler.update()
            scaler.scale(loss).backward()
                        
            scaler.step(optimizer)
            scaler.update()

            epoch_loss       += loss.item()
            epoch_score_loss += l_score.item()
            epoch_akd_loss   += l_akd.item()

            if i % print_freq == 0:
                loop.set_postfix(
                    total=f"{loss.item():.4f}",
                    score=f"{l_score.item():.4f}",
                    akd=f"{l_akd.item():.4f}",
                )

        n = len(train_loader)
        print(
            f"[Student] Epoch {epoch} | "
            f"Total: {epoch_loss/n:.4f} | "
            f"Score: {epoch_score_loss/n:.4f} | "
            f"AKD: {epoch_akd_loss/n:.4f}"
        )

        # ── Validation ───────────────────────────────────────────────────────
        metrics = validate_student(student, val_loader, device)
        print(
            f"[Student] Val | "
            f"PLCC: {metrics['plcc']:.4f} | "
            f"SROCC: {metrics['srocc']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f}"
        )

        if epoch % 2 == 0:   # every epoch, adjust frequency as needed
            weights = akd_loss_fn.get_effective_weights()
            weight_str = " | ".join(
                f"L{k}={v:.4f}" for k, v in weights.items()
            )
            print(f"[AKDLoss] Effective ω: {weight_str}")


        if scheduler is not None:
            scheduler.step()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if metrics["plcc"] > best_plcc:
            best_plcc = metrics["plcc"]
            save_checkpoint(
                save_dir=save_dir,
                filename="student_best.pth",
                model=student,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_plcc,
                akd_loss_fn=akd_loss_fn,
            )


# =========================================================
# Teacher KD Helper
# =========================================================

def _teacher_forward_for_kd(teacher, ref_patches, dist_patches, device):
    """
    Run teacher forward pass to collect pre-selected, detached
    MFDE intermediates for AKD loss.

    Teacher uses actual ref-dist difference (full-reference signal),
    which is richer than the student's redist-dist pseudo-difference.
    This is the knowledge being transferred.

    Args:
        teacher:      frozen TeacherModel (plain or DataParallel)
        ref_patches:  (B, N, 3, H, W)
        dist_patches: (B, N, 3, H, W)

    Returns:
        teacher_intermediates: [4][num_paired_layers]
                               each (B*N, 49, 256), detached
    """
    B, N, C, H, W = dist_patches.shape
    BN = B * N

    ref  = ref_patches.view(BN, C, H, W)
    dist = dist_patches.view(BN, C, H, W)

    # Unwrap DataParallel to access submodules directly
    t = teacher.module if isinstance(teacher, nn.DataParallel) else teacher

    # Batched backbone
    both          = torch.cat([ref, dist], dim=0)
    both_backbone = t.backbone(both)
    # ref_backbone  = [f[:BN] for f in both_backbone]
    # dist_backbone = [f[BN:] for f in both_backbone]

    # Batched MFR
    both_mfr   = t.mfr(both_backbone)
    ref_feats  = [f[:BN] for f in both_mfr]
    dist_feats = [f[BN:] for f in both_mfr]

    # Ref - dist difference (full-reference signal)
    diff_feats = [r - d for r, d in zip(ref_feats, dist_feats)]

    # MFDE with store_selected_only=True — returns only the paired layers, so half
    # (every other layer starting from index 1) already detached
    _, teacher_intermediates = t.mfde(
        diff_feats,
        store_selected_only=True,
    )
    # teacher_intermediates: [4][num_student_layers] — ready for AKDLoss

    return teacher_intermediates


# =========================================================
# Validation
# =========================================================

def validate_student(model, val_loader, device):
    """
    Validate student model on val_loader.
    Uses actual redist from batch — no dummy substitution.
    Computes PLCC, SRCC, RMSE via IQAMetrics.
    """
    model.eval()
    preds_all = []
    mos_all   = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            dist   = batch["dist"].to(device, non_blocking=True)
            redist = batch["redist"].to(device, non_blocking=True)
            mos    = batch["mos"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                pred, _ = model(
                    dist_patches=dist,
                    redist_patches=redist,
                    store_intermediates=False,  # no KD needed at val time
                )

            preds_all.append(pred.cpu())
            mos_all.append(mos.cpu())

    preds_all = torch.cat(preds_all)  # (total_samples,)
    mos_all   = torch.cat(mos_all)    # (total_samples,)

    metrics = IQAMetrics.compute_all_metrics(preds_all, mos_all, verbose=False)
    return metrics