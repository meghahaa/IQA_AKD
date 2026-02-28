import torch
import torch.nn as nn
from .backbone.mixerMLP import MixerBlock


class MFDE(nn.Module):
    """
    Multi-Level Feature Discrepancy Extractor (MFDE)
    Processes each of the 4 scale levels INDEPENDENTLY through
    a stack of MLP-Mixer blocks. 

    Each level: (B*N, 49, 256) → MixerBlocks → (B*N, 49, 256)

    Intermediates are stored per-level, giving shape:
        [num_levels][depth] → each (B*N, 49, 256)
    """

    def __init__(
        self,
        num_tokens_per_level=49,   # 7×7 per level
        num_levels=4,
        embed_dim=256,
        depth=18,
        token_mlp_dim=128,
        channel_mlp_dim=512,
        verbose=False,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.num_tokens_per_level = num_tokens_per_level
        self.embed_dim = embed_dim
        self.depth = depth
        self.verbose = verbose

        # Single shared stack — called once per level in forward()
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=num_tokens_per_level,
                embed_dim=embed_dim,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim,
            )
            for _ in range(depth)
        ])

        if self.verbose:
            print(
                f"[MFDE] Initialized | num_levels={num_levels} | "
                f"depth={depth} | tokens_per_level={num_tokens_per_level} | "
                f"embed_dim={embed_dim}"
            )

    def forward(self, diff_feats, store_selected_only=False):
        """
        Args:
            diff_feats:          list of 4 × (B*N, 49, 256)
            store_selected_only: True  → teacher mode: store only odd-indexed layers
                                        (every other starting index 1) — the 
                                        paired layers for KD. Stored detached.
                                False → student mode: store ALL layers for KD.
                                        Gradients preserved.

        Returns:
            final_feats:   list of 4 × (B*N, 49, 256)
            intermediates: [4][num_stored_layers] each (B*N, 49, 256)
                        num_stored_layers = depth//2 if store_selected_only
                                            = depth     if not store_selected_only
        """
        assert len(diff_feats) == self.num_levels, (
            f"[MFDE] Expected {self.num_levels} levels, got {len(diff_feats)}"
        )

        BN    = diff_feats[0].shape[0]
        L     = self.num_levels            # 4
        N_tok = self.num_tokens_per_level  # 49
        C     = self.embed_dim             # 256

        # Stack all levels into batch dim → single CUDA pass per block
        x = torch.stack(diff_feats, dim=1).view(BN * L, N_tok, C)
        # x: (B*N*4, 49, 256)

        # Teacher: odd indices {1, 3, 5, ..., depth-1} — pairs with student 0..depth//2-1
        selected_indices = set(range(1, self.depth, 2))

        intermediates = [[] for _ in range(L)]

        for layer_idx, block in enumerate(self.mixer_blocks):
            x = block(x)

            # Decide whether to store this layer
            should_store = (
                (store_selected_only and layer_idx in selected_indices)  # teacher
                or
                (not store_selected_only)                                 # student — store all
            )

            if should_store:
                x_split = x.view(BN, L, N_tok, C)   # (B*N, 4, 49, 256)
                for level_idx in range(L):
                    feat = x_split[:, level_idx]      # (B*N, 49, 256)
                    intermediates[level_idx].append(
                        feat.detach() if store_selected_only else feat
                        # teacher: detach — no grad needed, saves memory
                        # student: keep grad — needed for backprop through KD loss
                    )

        # Unstack levels
        x = x.view(BN, L, N_tok, C)                  # (B*N, 4, 49, 256)
        final_feats = [x[:, i] for i in range(L)]    # 4 × (B*N, 49, 256)

        return final_feats, intermediates