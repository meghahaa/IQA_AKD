import torch
import torch.nn as nn


class QualityRegressor(nn.Module):
    """
    Quality Score Regressor
    Aggregates patch-level features into an image-level quality score.
    """

    def __init__(
        self,
        embed_dim=256,
        hidden_dim=128,
        verbose=False
    ):
        """
        Args:
            embed_dim (int): input feature dimension
            hidden_dim (int): hidden layer dimension
            verbose (bool): print debug info
        """
        super().__init__()

        self.verbose = verbose

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        if self.verbose:
            print("[Regressor] Initialized")

    def forward(self, x, num_patches):
        """
        Args:
            x: Tensor [B*N, C]
            num_patches: int (patches per image)

        Returns:
            scores: Tensor [B]
        """
        # Patch-wise prediction
        patch_scores = self.mlp(x)  # [B*N, 1]

        # Reshape and aggregate
        B = patch_scores.shape[0] // num_patches
        patch_scores = patch_scores.view(B, num_patches)

        scores = patch_scores.mean(dim=1)  # [B]

        if self.verbose:
            print(f"[Regressor] Output shape: {scores.shape}")

        return scores
