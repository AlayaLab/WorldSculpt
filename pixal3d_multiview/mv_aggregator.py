"""Multi-view aggregators for per-voxel proj features.

Two learned aggregators, selected via the trainer's `mv_aggregator` config
(`mode` key, see `_ZeroInitMVAggMixin` in trainers.py):

1. `ZeroInitMVAggregator` (mode="zero", default) — ControlNet-style:
       z_agg = z_anchor + zero_proj( masked_mean(others) )
   At init `z_agg == z_anchor` (single-view Pixal3D behavior).

2. `IBRMVAggregator` (mode="ibr") — IBRNet-style per-voxel view weighting,
   ported from GenRecon's `MultiViewFeatAggregator`
   (GenRecon/genrecon/modules/cond_3D/aggregation_net.py, arXiv 2605.23888):
       mu, var = masked mean/var over views          (per voxel)
       h_k     = MLP_feat([z_k, mu, var])            (last layer zero-init)
       w_k     = softmax_k(MLP_w([z_k, mu, var]))    (invalid views -> -1e9)
       z_agg   = mu + sum_k w_k * h_k
   At init `z_agg == mu` == the plain masked mean, i.e. exactly the current
   "avg" fusion — training starts from the verified baseline and learns a
   per-voxel per-view weighted residual on top.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt


class ZeroInitMVAggregator(nn.Module):
    """Zero-init residual aggregator for multi-view features.

    Applies the ControlNet "zero_conv" idea to:
      - per-voxel proj features:  proj_agg   = z_anchor + zero_proj(mean others)
      - (optional) global tokens: global_agg = g_anchor + zero_proj_global(mean others)

    At init both zero layers output 0, so the result == the anchor view's
    features and the model behaves exactly like single-view Pixal3D; the deltas
    grow during training.

    Both branches are computed inside `forward` (a single DDP-wrapped call) so
    DDP all-reduces the gradients of BOTH zero layers — do NOT call a separate
    method on the (DDP-wrapped) module for the global branch.

    Args:
        channels:        feature channels in z_proj (image_cond_model proj dim:
                         1024 for DINOv3-L without NAF, 2048 with NAF).
        global_channels: feature channels of the global token (DINOv3 embed dim,
                         1024 for DINOv3-L). None -> no global aggregation
                         (global stays anchor-only, original behavior).
    """
    def __init__(self, channels: int, global_channels: Optional[int] = None):
        super().__init__()
        self.zero_proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

        self.global_channels = global_channels
        if global_channels is not None:
            self.zero_proj_global = nn.Linear(global_channels, global_channels)
            nn.init.zeros_(self.zero_proj_global.weight)
            nn.init.zeros_(self.zero_proj_global.bias)
        else:
            self.zero_proj_global = None

    def forward(
        self,
        z_anchor: torch.Tensor,                       # (B, V, C)
        z_others_mean: torch.Tensor,                  # (B, V, C)
        g_anchor: Optional[torch.Tensor] = None,      # (B, L, Cg)
        g_others_mean: Optional[torch.Tensor] = None, # (B, L, Cg)
    ):
        """Returns (cond_proj, cond_global). At init both == the anchor inputs."""
        cond_proj = z_anchor + self.zero_proj(z_others_mean)
        if self.zero_proj_global is not None and g_anchor is not None:
            cond_global = g_anchor + self.zero_proj_global(g_others_mean)
        else:
            cond_global = g_anchor
        return cond_proj, cond_global


class IBRMVAggregator(nn.Module):
    """IBRNet-style learned multi-view aggregation (GenRecon port).

    Math is identical to GenRecon's `MultiViewFeatAggregator` core (the
    IBRNet block; conv/self-attention refinements dropped), with the view
    axis moved to dim=-2 so one code path serves both callers:

      - sparse stages (2/3): feats (N, K, C), mask (N, K)   -> (N, C)
        where N = active voxels gathered at `coords` across the batch
      - dense stage (1):     flatten (B, V, K, C) to (B*V, K, C) first

    `feature_mlp`'s last layer is zero-init, so at init the output equals
    the masked mean over views — bit-exact with the existing "avg" fusion.

    Stats (mu/var) are computed in fp32 regardless of input dtype (per-view
    feats are stored bf16 to halve memory; E[x^2]-E[x]^2 in bf16 would be
    garbage). `chunk_size` bounds peak activation memory: chunks are run
    under torch.utils.checkpoint during training (K can be up to ~20 views;
    the 3 saved MLP activations of shape (N, K, C) would otherwise dominate).

    Args:
        channels:   proj feature channels (1024 stage1, 2048 stage2/3).
        chunk_size: voxels per checkpointed chunk; None/0 disables chunking.
    """
    def __init__(self, channels: int, chunk_size: Optional[int] = 16384,
                 global_channels: Optional[int] = None):  # accepted+ignored (anchor-only global)
        super().__init__()
        self.channels = channels
        self.chunk_size = chunk_size or 0

        in_dim = 3 * channels  # [z_k, mu, var]
        self.feature_mlp = nn.Sequential(
            nn.Linear(in_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        nn.init.zeros_(self.feature_mlp[-1].weight)
        nn.init.zeros_(self.feature_mlp[-1].bias)

        self.weight_mlp = nn.Sequential(
            nn.Linear(in_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def _agg(self, feats: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        """feats (n, K, C) any float dtype; mask_f (n, K, 1) float. -> (n, C) fp32."""
        feats = feats.float()
        D = feats.shape[-1]

        sum_valid = mask_f.sum(dim=-2, keepdim=True).clamp_min(1.0)     # (n, 1, 1)
        feats_masked = feats * mask_f
        sum1 = feats_masked.sum(dim=-2, keepdim=True)                   # (n, 1, C)
        sum2 = (feats_masked * feats).sum(dim=-2, keepdim=True)         # (n, 1, C)
        mu = sum1 / sum_valid
        var = (sum2 / sum_valid - mu * mu).clamp_min(0)

        # Split-linear first layer (GenRecon trick): avoids materializing the
        # (n, K, 3C) concat; mu/var are projected at (n, 1, C) and broadcast.
        fW, fb = self.feature_mlp[0].weight, self.feature_mlp[0].bias
        wW, wb = self.weight_mlp[0].weight, self.weight_mlp[0].bias
        h = (F.linear(feats, fW[:, :D]) + F.linear(mu, fW[:, D:2 * D])
             + F.linear(var, fW[:, 2 * D:3 * D]) + fb)
        wl = (F.linear(feats, wW[:, :D]) + F.linear(mu, wW[:, D:2 * D])
              + F.linear(var, wW[:, 2 * D:3 * D]) + wb)
        for i in range(1, len(self.feature_mlp)):
            h = self.feature_mlp[i](h)                                   # (n, K, C)
        for i in range(1, len(self.weight_mlp)):
            wl = self.weight_mlp[i](wl)                                  # (n, K, 1)

        wl = wl.masked_fill(mask_f == 0, -1e9)
        w = torch.softmax(wl, dim=-2)                                    # (n, K, 1)
        return (mu + (h * w).sum(dim=-2, keepdim=True)).squeeze(-2)      # (n, C)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (N, K, C) per-view per-voxel proj features (views at dim=-2).
            mask:  (N, K) bool/float — view validity (sample-level view_mask
                   broadcast per voxel; per-voxel frustum masks plug in here later).
        Returns:
            (N, C) fp32 aggregated features. At init == masked mean over views.
        """
        assert feats.dim() == 3 and mask.dim() == 2, \
            f"expected feats (N,K,C) / mask (N,K), got {feats.shape} / {mask.shape}"
        mask_f = mask.float().unsqueeze(-1)                              # (N, K, 1)

        n = feats.shape[0]
        cs = self.chunk_size
        if cs <= 0 or n <= cs:
            return self._agg(feats, mask_f)

        outs = []
        use_ckpt = self.training and torch.is_grad_enabled()
        for s in range(0, n, cs):
            f_c, m_c = feats[s:s + cs], mask_f[s:s + cs]
            if use_ckpt:
                outs.append(torch_ckpt.checkpoint(self._agg, f_c, m_c, use_reentrant=False))
            else:
                outs.append(self._agg(f_c, m_c))
        return torch.cat(outs, dim=0)
