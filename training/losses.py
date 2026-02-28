import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreLoss(nn.Module):
    """
    L1 Loss for IQA score regression
    Used by both teacher and student
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, pred, target):
        """
        Args:
            pred: Tensor [B]
            target: Tensor [B]
        """
        return self.loss_fn(pred, target)


class AKDLoss(nn.Module):
    """
    Inter-Level Adaptive Knowledge Distillation Loss.

    Operates on pre-paired intermediates — layer selection is handled
    inside MFDE via store_selected_only=True on the teacher.
    Therefore teacher and student intermediates are already aligned:
        intermediates[level][k] for k in 0..depth-1

    ω(j) is one learnable scalar weight per level (4 weights total),
    initialized to 0.25 initially.
    """
    # softplus_inverse(0.25) = log(exp(0.25) - 1) ≈ -1.2593
    # Used to initialize raw params so effective weights start at 0.25
    _INIT_RAW = torch.log(torch.tensor(0.25).exp() - 1).item()  # ≈ -1.2593

    def __init__(
        self,
        num_levels=4,
        init_weight=0.25,
        verbose=False,
    ):
        """
        Args:
            num_levels (int):   number of scale levels (always 4)
            init_weight (float): initial value for ω, paper uses 0.25
            verbose (bool):     print per-level loss values
        """
        super().__init__()

        self.num_levels  = num_levels
        self.verbose     = verbose

        # Compute raw init value so softplus(raw) == init_weight
        # softplus_inverse(y) = log(exp(y) - 1)
        raw_init = torch.log(
            torch.tensor(init_weight).exp() - torch.tensor(1.0)
        ).item()

        # Raw learnable parameters — softplus applied in forward
        self.omega_raw = nn.Parameter(
            torch.full((num_levels,), raw_init)
        )

    def forward(self, teacher_intermediates, student_intermediates):
        """
        Args:
            teacher_intermediates: [num_levels][num_paired_layers]
                                   each tensor: (B*N, 49, 256)
                                   pre-selected inside teacher MFDE
                                   (store_selected_only=True, detached)

            student_intermediates: [num_levels][num_student_layers]
                                   each tensor: (B*N, 49, 256)
                                   all student layers

        Returns:
            loss: scalar AKD loss
        """
        assert len(teacher_intermediates) == self.num_levels, (
            f"[AKDLoss] Expected {self.num_levels} teacher levels, "
            f"got {len(teacher_intermediates)}"
        )
        assert len(student_intermediates) == self.num_levels, (
            f"[AKDLoss] Expected {self.num_levels} student levels, "
            f"got {len(student_intermediates)}"
        )

        # Paired layers must match — selection already done in MFDE
        for level in range(self.num_levels):
            assert len(teacher_intermediates[level]) == len(student_intermediates[level]), (
                f"[AKDLoss] Level {level}: teacher has "
                f"{len(teacher_intermediates[level])} paired layers but "
                f"student has {len(student_intermediates[level])} layers. "
                f"Ensure teacher MFDE uses store_selected_only=True and "
                f"teacher depth is exactly 2× student depth."
            )

        # Apply softplus to get positive effective weights
        # Shape: (num_levels,)
        omega = F.softplus(self.omega_raw)

        device = student_intermediates[0][0].device
        loss   = torch.tensor(0.0, device=device)

        for level in range(self.num_levels):
            t_layers = teacher_intermediates[level]  # list of K tensors
            s_layers = student_intermediates[level]  # list of K tensors
            K        = len(s_layers)

            level_loss = torch.tensor(0.0, device=device)

            for k in range(K):
                t_feat = t_layers[k]   # already detached in MFDE
                s_feat = s_layers[k]

                level_loss = level_loss + F.mse_loss(s_feat, t_feat)

            # Average over layers then weight by softplus(omega_raw[level])
            level_loss = level_loss / K
            loss       = loss + omega[level] * level_loss

            if self.verbose:
                print(
                    f"[AKDLoss] Level {level} | "
                    f"ω={omega[level].item():.4f} | "
                    f"level_loss={level_loss.item():.4f}"
                )

        return loss

    def get_effective_weights(self):
        """
        Utility to inspect current effective omega values during training.
        Call this in your training loop to monitor how weights evolve.
        
        Returns:
            dict mapping level index to effective weight value
        """
        with torch.no_grad():
            omega = F.softplus(self.omega_raw)
        return {
            f"omega_level_{j}": omega[j].item()
            for j in range(self.num_levels)
        }