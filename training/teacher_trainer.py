import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbone.swin import SwinBackbone
from models.mfr import MultiScaleFeatureRepresentation
from models.mfde import MFDE
from models.cfi import CFI
from models.caf import CAF
from models.regressor import QualityRegressor

from training.losses import ScoreLoss
from utils.checkpoints import save_checkpoint


class TeacherModel(nn.Module):
    """
    Full-Reference IQA Teacher Model (AKD-IQA)

    Pipeline per forward pass:
        1. Flatten (B, N, 3, H, W) patches → (B*N, 3, H, W)
        2. SwinBackbone → 4 feature maps per image [(B*N, C_i, H_i, W_i)]
        3. MFR → 4 unified tokens per image [(B*N, 49, 256)] each level
        4. Concatenate 4 levels along token dim → (B*N, 196, 256)
        5. Subtract ref tokens from dist tokens → diff tokens (B*N, 196, 256)
        6. MFDE(diff tokens) → F_diff (B*N, 196, 256)  [+ intermediates for KD]
        7. CFI(dist tokens)  → F_dist (B*N, 196, 256)
        8. CAF(F_diff, F_dist) → fused (B*N, 196, 256)
        9. Regressor → score (B, 1)  [averages over N patches]
    """

    def __init__(
        self,
        embed_dim=256,
        mfde_depth=36,
        cfi_depth=18,
        num_patches=10,
        verbose=False
    ):
        super().__init__()

        self.verbose = verbose
        self.num_patches = num_patches

        # Shared backbone — same weights used for both ref and dist branches
        self.backbone = SwinBackbone(pretrained=True, verbose=verbose)

        self.mfr = MultiScaleFeatureRepresentation(
            in_channels=self.backbone.out_channels,  # [96, 192, 384, 768]
            embed_dim=embed_dim,                     # → 256 per level
            target_spatial_dim=7,                    # → 7×7 per level
            verbose=verbose
        )

        # MFDE operates on concatenated diff tokens: 4 levels × 49 tokens = 196
        self.mfde = MFDE(
            num_tokens=196,
            embed_dim=embed_dim,
            depth=mfde_depth,      # 36 for teacher
            verbose=verbose
        )

        # CFI operates on concatenated dist tokens: same shape (B*N, 196, 256)
        self.cfi = CFI(
            num_tokens=196,
            embed_dim=embed_dim,
            depth=cfi_depth,       # 18 for teacher
            verbose=verbose
        )

        self.caf = CAF(embed_dim=embed_dim, verbose=verbose)
        self.regressor = QualityRegressor(embed_dim=embed_dim)

        if self.verbose:
            print("[TeacherModel] Initialized")
            print(f"[TeacherModel] MFDE depth={mfde_depth}, CFI depth={cfi_depth}")
    
    def forward(self, ref_patches, dist_patches):
        """
        Args:
            ref_patches:  (B, N, 3, H, W)
            dist_patches: (B, N, 3, H, W)

        Returns:
            score: (B,)
        """
        B, N, C, H, W = dist_patches.shape

        # ── 1. Flatten patches into batch dim ────────────────────────────────
        ref  = ref_patches.view(B * N, C, H, W)   # (B*N, 3, 224, 224)
        dist = dist_patches.view(B * N, C, H, W)  # (B*N, 3, 224, 224)

        # ── 2. Shared backbone ───────────────────────────────────────────────
        # List of 4: [(B*N, 96,  56, 56),
        #             (B*N, 192, 28, 28),
        #             (B*N, 384, 14, 14),
        #             (B*N, 768,  7,  7)]
        ref_backbone  = self.backbone(ref)
        dist_backbone = self.backbone(dist)

        # ── 3. MFR — project + pool → list of 4 × (B*N, 49, 256) ────────────
        ref_feats  = self.mfr(ref_backbone)   # [(B*N, 49, 256)] × 4
        dist_feats = self.mfr(dist_backbone)  # [(B*N, 49, 256)] × 4

        # ── 4. Concatenate levels along token dim → (B*N, 196, 256) ──────────
        ref_tokens  = torch.cat(ref_feats,  dim=1)  # (B*N, 196, 256)
        dist_tokens = torch.cat(dist_feats, dim=1)  # (B*N, 196, 256)

        # ── 5. Difference tokens (Eq. 2) ─────────────────────────────────────
        diff_tokens = ref_tokens - dist_tokens       # (B*N, 196, 256)

        # ── 6. MFDE on difference tokens ─────────────────────────────────────
        f_diff_levels, _ = self.mfde(diff_tokens)
        # f_diff_levels: list of 4 × (B*N, 49, 256)

        # Concatenate levels → (B*N, 196, 256) for CAF
        f_diff = torch.cat(f_diff_levels, dim=1)         # (B*N, 196, 256)        

        # ── 7. CFI on distortion tokens ──────────────────────────────────────
        f_dist = self.cfi(dist_tokens)               # (B*N, 196, 256)

        # ── 8. Cross-attention fusion (diff→Q,K  |  dist→V) ──────────────────
        fused = self.caf(f_diff, f_dist)             # (B*N, 196, 256)

        # ── 9. Regressor — pool tokens, average patches, predict score ────────
        score = self.regressor(fused, N)             # (B,)

        return score


# =========================================================
# Training Function
# =========================================================

def train_teacher(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    save_dir,
    print_freq=10
):
    model.to(device)
    criterion = ScoreLoss()
    best_plcc = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        print(f"\n[Teacher] Epoch {epoch}/{epochs}")
        loop = tqdm(train_loader, desc="Training")

        for i, batch in enumerate(loop):
            ref  = batch["ref"].to(device)   # (B, N, 3, H, W)
            dist = batch["dist"].to(device)  # (B, N, 3, H, W)
            mos  = batch["mos"].to(device)   # (B,)

            optimizer.zero_grad()
            preds = model(ref, dist)         # (B, 1)
            loss  = criterion(preds, mos)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % print_freq == 0:
                loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Teacher] Train Loss: {avg_loss:.4f}")

        # Validation
        plcc = validate_teacher(model, val_loader, device)
        print(f"[Teacher] Validation PLCC: {plcc:.4f}")

        # Checkpoint
        if plcc > best_plcc:
            best_plcc = plcc
            save_checkpoint(
                save_dir=save_dir,
                filename="teacher_best.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_plcc
            )


# =========================================================
# Validation Function
# =========================================================

def validate_teacher(model, val_loader, device):
    model.eval()
    preds_all, mos_all = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            ref  = batch["ref"].to(device)
            dist = batch["dist"].to(device)
            mos  = batch["mos"].to(device)

            preds = model(ref, dist)         # (B, 1)

            preds_all.append(preds.cpu())
            mos_all.append(mos.cpu())

    preds_all = torch.cat(preds_all).squeeze(1)  # (total_samples,)
    mos_all   = torch.cat(mos_all)               # (total_samples,)

    plcc = torch.corrcoef(
        torch.stack([preds_all, mos_all])
    )[0, 1].item()

    return plcc