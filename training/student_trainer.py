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

from training.losses import ScoreLoss, AKDLoss
from utils.metrics import IQAMetrics
from utils.checkpoints import save_checkpoint


# =========================================================
# Student Model
# =========================================================

class StudentModel(nn.Module):
    """
    No-Reference IQA Student Model
    """

    def __init__(
        self,
        embed_dim=256,
        mfde_depth=18,
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
            depth=9
        )

        self.caf = CAF(embed_dim=embed_dim)
        self.regressor = QualityRegressor(embed_dim=embed_dim)

        if self.verbose:
            print("[StudentModel] Initialized")

    def forward(self, dist_patches, redist_patches, return_intermediate=False):
        """
        Args:
            dist_patches:   [B, N, 3, H, W]
            redist_patches: [B, N, 3, H, W]
            return_intermediate (bool): return MFDE intermediates

        Returns:
            score, intermediates (optional)
        """
        B, N, C, H, W = dist_patches.shape

        dist = dist_patches.view(B * N, C, H, W)
        redist = redist_patches.view(B * N, C, H, W)

        # Backbone + MFR
        dist_feats = self.mfr(self.backbone(dist))
        redist_feats = self.mfr(self.backbone(redist))

        # Pseudo difference
        diff_feats = [r - d for r, d in zip(redist_feats, dist_feats)]

        # MFDE per scale
        mfde_outs = []
        mfde_intermediates = []

        for feat in diff_feats:
            out, inter = self.mfde(feat)
            mfde_outs.append(out)
            mfde_intermediates.append(inter)

        # Cross-scale integration
        diff_global = self.cfi(mfde_outs)

        dist_global = self.cfi(
            [f.flatten(2).transpose(1, 2) for f in dist_feats]
        )

        fused = self.caf(diff_global, dist_global)
        score = self.regressor(fused, self.num_patches)

        if return_intermediate:
            return score, mfde_intermediates
        return score


# =========================================================
# Student Training Function
# =========================================================

def train_student(
    student,
    teacher,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    save_dir,
    print_freq=10
):
    student.to(device)
    teacher.to(device)

    teacher.eval()  # freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    score_loss_fn = ScoreLoss()
    akd_loss_fn = AKDLoss(
        num_student_layers=len(student.mfde.mixer_blocks),
        num_teacher_layers=len(teacher.mfde.mixer_blocks),
        verbose=False
    )

    best_plcc = -1e9

    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0.0

        print(f"\n[Student] Epoch {epoch}/{epochs}")
        loop = tqdm(train_loader, desc="Training")

        for i, batch in enumerate(loop):
            dist = batch["dist"].to(device)
            redist = batch["redist"].to(device)
            mos = batch["mos"].to(device)

            optimizer.zero_grad()

            # -------- Teacher Forward --------
            with torch.no_grad():
                _, teacher_inter = teacher_forward_for_kd(
                    teacher, dist, redist
                )

            # -------- Student Forward --------
            pred, student_inter = student(
                dist, redist, return_intermediate=True
            )

            # -------- Loss --------
            l_score = score_loss_fn(pred, mos)
            l_akd = akd_loss_fn(
                flatten_intermediates(teacher_inter),
                flatten_intermediates(student_inter)
            )

            loss = l_score + l_akd
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % print_freq == 0:
                loop.set_postfix(
                    score=l_score.item(),
                    akd=l_akd.item()
                )

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Student] Train Loss: {avg_loss:.4f}")

        # -------- Validation --------
        plcc = validate_student(student, val_loader, device)
        print(f"[Student] Validation PLCC: {plcc:.4f}")

        # -------- Checkpoint --------
        if plcc > best_plcc:
            best_plcc = plcc
            save_checkpoint(
                save_dir=save_dir,
                filename="student_best.pth",
                model=student,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_plcc
            )


# =========================================================
# Helper Functions
# =========================================================

def teacher_forward_for_kd(teacher, dist, redist):
    """
    Forward pass through teacher for KD features
    """
    B, N, C, H, W = dist.shape
    ref = redist.view(B * N, C, H, W)
    dist = dist.view(B * N, C, H, W)

    ref_feats = teacher.mfr(teacher.backbone(ref))
    dist_feats = teacher.mfr(teacher.backbone(dist))
    diff_feats = [r - d for r, d in zip(ref_feats, dist_feats)]

    mfde_inter = []

    for feat in diff_feats:
        _, inter = teacher.mfde(feat)
        mfde_inter.append(inter)

    return None, mfde_inter


def flatten_intermediates(inter_list):
    """
    Convert per-scale intermediate lists into one flat list
    """
    flat = []
    for scale in inter_list:
        flat.extend(scale)
    return flat


def validate_student(model, val_loader, device):
    model.eval()
    preds_all, mos_all = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            dist = batch["dist"].to(device)
            mos = batch["mos"].to(device)

            pred = model(dist, dist)  # dummy redist during test
            preds_all.append(pred.cpu())
            mos_all.append(mos.cpu())

    preds_all = torch.cat(preds_all)
    mos_all = torch.cat(mos_all)

    plcc = torch.corrcoef(
        torch.stack([preds_all, mos_all])
    )[0, 1].item()

    return plcc
