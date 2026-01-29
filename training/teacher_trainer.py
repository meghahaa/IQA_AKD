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


# =========================================================
# Teacher Model
# =========================================================

class TeacherModel(nn.Module):
    """
    Full-Reference IQA Teacher Model
    """

    def __init__(
        self,
        embed_dim=256,
        mfde_depth=36,
        num_scales=4,
        num_patches=10,
        verbose=False
    ):
        super().__init__()

        self.verbose = verbose
        self.num_patches = num_patches

        self.backbone = SwinBackbone(pretrained=True)
        self.mfr = MultiScaleFeatureRepresentation(
            in_channels=self.backbone.out_channels,
            embed_dim=embed_dim
        )

        self.mfde = MFDE(
            embed_dim=embed_dim,
            depth=mfde_depth
        )

        self.cfi = CFI(
            num_scales=num_scales,
            embed_dim=embed_dim,
            depth=18
        )

        self.caf = CAF(embed_dim=embed_dim)
        self.regressor = QualityRegressor(embed_dim=embed_dim)

        if self.verbose:
            print("[TeacherModel] Initialized")

    def forward(self, ref_patches, dist_patches):
        """
        Args:
            ref_patches:  [B, N, 3, H, W]
            dist_patches: [B, N, 3, H, W]
        """
        B, N, C, H, W = dist_patches.shape

        # Flatten patches
        ref = ref_patches.view(B * N, C, H, W)
        dist = dist_patches.view(B * N, C, H, W)

        # Backbone + MFR
        ref_feats = self.mfr(self.backbone(ref))
        dist_feats = self.mfr(self.backbone(dist))

        # Difference features (per scale)
        diff_feats = [r - d for r, d in zip(ref_feats, dist_feats)]

        # MFDE per scale
        mfde_outs = []
        for feat in diff_feats:
            out, _ = self.mfde(feat)
            mfde_outs.append(out)

        # Cross-scale integration
        diff_global = self.cfi(mfde_outs)

        # Distorted image global feature
        dist_global = self.cfi(
            [f.flatten(2).transpose(1, 2) for f in dist_feats]
        )

        # Cross-attention fusion
        fused = self.caf(diff_global, dist_global)

        # Regress score
        score = self.regressor(fused, self.num_patches)

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
            ref = batch["ref"].to(device)
            dist = batch["dist"].to(device)
            mos = batch["mos"].to(device)

            optimizer.zero_grad()
            preds = model(ref, dist)
            loss = criterion(preds, mos)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % print_freq == 0:
                loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Teacher] Train Loss: {avg_loss:.4f}")

        # ---------------- Validation ----------------
        plcc = validate_teacher(model, val_loader, device)
        print(f"[Teacher] Validation PLCC: {plcc:.4f}")

        # ---------------- Checkpoint ----------------
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
            ref = batch["ref"].to(device)
            dist = batch["dist"].to(device)
            mos = batch["mos"].to(device)

            preds = model(ref, dist)

            preds_all.append(preds.cpu())
            mos_all.append(mos.cpu())

    preds_all = torch.cat(preds_all)
    mos_all = torch.cat(mos_all)

    # PLCC computation
    plcc = torch.corrcoef(
        torch.stack([preds_all, mos_all])
    )[0, 1].item()

    return plcc
