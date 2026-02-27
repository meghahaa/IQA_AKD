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
        verbose=False
    ):
        """
        Args:
            embed_dim (int): input feature dimension
            verbose (bool): print debug info
        """
        super().__init__()

        self.verbose = verbose

        self.fc = nn.Linear(embed_dim, 1)

        if self.verbose:
            print("[Regressor] Initialized")

    def forward(self, x, num_patches):
        """
        Args:
            x: Tensor [B*N, N_tokens, C]
            num_patches: int (patches per image) i.e N

        Returns:
            scores: Tensor [B]
        """
        
        # Global average pooling over tokens -> [B*N, C]
        x = x.mean(dim=1)  

        if self.verbose:
            print(f"[Regressor] Input shape: {x.shape}")

        # Patch-wise prediction
        patch_scores = self.fc(x)  # [B*N, 1]

        # Reshape and aggregate
        B = patch_scores.shape[0] // num_patches
        patch_scores = patch_scores.view(B, num_patches)

        scores = patch_scores.mean(dim=1)  # [B]

        if self.verbose:
            print(f"[Regressor] Output shape: {scores.shape}")

        return scores
