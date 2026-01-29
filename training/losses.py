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
    Inter-level Adaptive Knowledge Distillation Loss
    """

    def __init__(
        self,
        num_student_layers,
        num_teacher_layers,
        init_weight=0.25,
        verbose=False
    ):
        """
        Args:
            num_student_layers (int): depth of student MFDE
            num_teacher_layers (int): depth of teacher MFDE
            init_weight (float): initial weight for each level
            verbose (bool): print debug info
        """
        super().__init__()

        self.verbose = verbose

        # Layer mapping: teacher -> student
        self.layer_map = self._build_layer_map(
            num_student_layers, num_teacher_layers
        )

        # Learnable adaptive weights ω_i
        self.weights = nn.Parameter(
            torch.ones(len(self.layer_map)) * init_weight
        )

        if self.verbose:
            print("[AKDLoss] Initialized")
            print(f"[AKDLoss] Layer map (T -> S): {self.layer_map}")

    def _build_layer_map(self, n_student, n_teacher):
        """
        Map teacher layers to student layers
        Paper rule: sample teacher layers evenly
        """
        step = n_teacher // n_student
        teacher_indices = list(range(step - 1, n_teacher, step))
        teacher_indices = teacher_indices[:n_student]

        student_indices = list(range(n_student))
        return list(zip(teacher_indices, student_indices))

    def forward(self, teacher_feats, student_feats):
        """
        Args:
            teacher_feats: list of Tensors from teacher MFDE
            student_feats: list of Tensors from student MFDE

        Returns:
            akd_loss: scalar
        """
        assert len(student_feats) == len(self.layer_map)

        loss = 0.0

        for i, (t_idx, s_idx) in enumerate(self.layer_map):
            t_feat = teacher_feats[t_idx]
            s_feat = student_feats[s_idx]

            # Align token dimensions by pooling
            t_feat = t_feat.mean(dim=1)
            s_feat = s_feat.mean(dim=1)

            l = F.mse_loss(s_feat, t_feat)
            loss += self.weights[i] * l

            if self.verbose:
                print(
                    f"[AKDLoss] Layer {i} | "
                    f"T:{t_idx} S:{s_idx} | "
                    f"Loss: {l.item():.4f}"
                )

        return loss
