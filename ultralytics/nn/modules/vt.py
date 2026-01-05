# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRL(torch.autograd.Function):
    """Gradient Reversal Layer (GRL).

    Forward is identity; backward multiplies gradients by -lambda.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # noqa: D401
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg().mul(ctx.lambd), None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GRL.apply(x, lambd)


class VTImageDomainTap(nn.Module):
    """Image-level domain classifier tap (VersatileTeacher-style).

    Matches the original VT `ImageDomainClassifier` structure:
    GRL -> ChannelAttn -> Conv(c->c,3x3) -> ReLU -> Conv(c->c/2,3x3) -> ReLU -> Conv(c/2->c/4,3x3) -> ReLU
    -> Conv(c/4->1,1x1) -> Flatten(HW) -> Linear(HW->2).

    Implemented as a 'tap': stores logits for the Trainer but returns the input feature unchanged.
    """

    def __init__(self, c1: int, size: int, grl_lambda: float = 0.1, num_domains: int = 2):
        super().__init__()
        self.size = int(size)
        self.grl_lambda = float(grl_lambda)
        self.num_domains = int(num_domains)
        self.attn = _ChannelAttention(c1)
        c2 = max(8, c1 // 2)
        c3 = max(8, c2 // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.size, self.size))
        self.fc = nn.Linear(self.size * self.size, self.num_domains)
        self.last_logits: torch.Tensor | None = None
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: during model initialization Ultralytics runs an eval() forward to infer strides.
        # Storing non-leaf tensors (with grad_fn) as attributes breaks deepcopy() used by ModelEMA.
        if not self.training:
            self.last_logits = None
            self.last_attn = None
            return x

        z = grad_reverse(x, self.grl_lambda)
        # Defensive: ensure (B, C, H, W) so Linear always receives a 2D (B, S*S) matrix.
        if z.ndim == 3:
            z = z.unsqueeze(0)
        z = self.attn(z)
        # Cache a spatial attention map for optional alignment loss.
        self.last_attn = z.mean(dim=1, keepdim=True)
        z = self.conv(z)  # (B, 1, H, W)
        z = self.pool(z)  # (B, 1, S, S)
        z = z.flatten(1)  # (B, S*S)
        # Ensure stable 2D logits for CE loss even if a caller provides an unexpected shape.
        self.last_logits = self.fc(z).reshape(-1, self.num_domains)  # (B, 2)
        return x


class VTUnifiedImageDomainTap(nn.Module):
    """Unified multi-scale image-level domain classifier tap (UC-style)."""

    def __init__(
        self,
        c1: list[int] | tuple[int, ...] | int,
        size: int,
        grl_lambda: float = 0.1,
        mid_channels: int = 128,
        num_domains: int = 2,
    ):
        super().__init__()
        if isinstance(c1, int):
            c1 = [c1]
        self.c1 = list(c1)
        self.size = int(size)
        self.grl_lambda = float(grl_lambda)
        self.mid_channels = int(mid_channels)
        self.num_domains = int(num_domains)
        self.enabled = True

        self.reduce = nn.ModuleList([nn.Conv2d(c, self.mid_channels, kernel_size=1) for c in self.c1])
        # Two-stage downsampling convs, applied as needed to match target spatial size.
        self.down = nn.ModuleList(
            [
                nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=2, padding=1),
            ]
        )

        in_ch = self.mid_channels * len(self.c1)
        self.attn = _ChannelAttention(in_ch)
        c2 = max(8, in_ch // 2)
        c3 = max(8, c2 // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, c2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.size, self.size))
        self.fc = nn.Linear(self.size * self.size, self.num_domains)
        self.last_logits: torch.Tensor | None = None
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        if not self.training or not self.enabled:
            self.last_logits = None
            self.last_attn = None
            return x

        if isinstance(x, torch.Tensor):
            feats = [x]
        else:
            feats = list(x)

        reduced = []
        for idx, feat in enumerate(feats):
            z = self.reduce[idx](feat)
            # Downsample with strided convs as needed to reach target size.
            step = 0
            while z.shape[-1] > self.size and step < len(self.down):
                z = self.down[step](z)
                step += 1
            if z.shape[-1] != self.size:
                z = F.adaptive_avg_pool2d(z, (self.size, self.size))
            reduced.append(z)

        fused = torch.cat(reduced, dim=1)
        z = grad_reverse(fused, self.grl_lambda)
        z = self.attn(z)
        self.last_attn = z.mean(dim=1, keepdim=True)
        z = self.conv(z)  # (B, 1, S, S)
        z = self.pool(z)
        z = z.flatten(1)
        self.last_logits = self.fc(z).reshape(-1, self.num_domains)
        return x


class _ChannelAttention(nn.Module):
    """Lightweight channel attention (SE-style) for domain classifiers."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(self.pool(x))
        return x * w


class VTInstanceDomainTap(nn.Module):
    """Instance-level domain classifier that supports an externally provided mask.

    Like VTImageDomainTap, this is a 'tap' that stores logits and returns input unchanged.
    """

    def __init__(self, c1: int, grl_lambda: float = 0.1, shortcut: bool = True):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.shortcut = bool(shortcut)
        self.mask: torch.Tensor | None = None  # expected shape (B, H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self.last_logits: torch.Tensor | None = None  # (B, H, W)

    def set_mask(self, mask: torch.Tensor | None):
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            z = grad_reverse(x, self.grl_lambda)
            if self.mask is not None:
                # broadcast mask to channels: (B,H,W) -> (B,1,H,W) then broadcast
                mask = self.mask.unsqueeze(1).to(dtype=z.dtype, device=z.device)
                if self.shortcut:
                    z = z + z * mask
                else:
                    z = z * mask
            z = self.conv(z)
            self.last_logits = z.squeeze(1)
        else:
            self.last_logits = None
        return x
