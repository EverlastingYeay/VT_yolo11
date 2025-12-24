# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn


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
    GRL -> Conv(c->c,3x3) -> ReLU -> Conv(c->1,1x1) -> Flatten(HW) -> Linear(HW->2).

    Implemented as a 'tap': stores logits for the Trainer but returns the input feature unchanged.
    """

    def __init__(self, c1: int, size: int, grl_lambda: float = 0.1, num_domains: int = 2):
        super().__init__()
        self.size = int(size)
        self.grl_lambda = float(grl_lambda)
        self.num_domains = int(num_domains)
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.size, self.size))
        self.fc = nn.Linear(self.size * self.size, self.num_domains)
        self.last_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: during model initialization Ultralytics runs an eval() forward to infer strides.
        # Storing non-leaf tensors (with grad_fn) as attributes breaks deepcopy() used by ModelEMA.
        if not self.training:
            self.last_logits = None
            return x

        z = grad_reverse(x, self.grl_lambda)
        # Defensive: ensure (B, C, H, W) so Linear always receives a 2D (B, S*S) matrix.
        if z.ndim == 3:
            z = z.unsqueeze(0)
        z = self.conv(z)  # (B, 1, H, W)
        z = self.pool(z)  # (B, 1, S, S)
        z = z.flatten(1)  # (B, S*S)
        # Ensure stable 2D logits for CE loss even if a caller provides an unexpected shape.
        self.last_logits = self.fc(z).reshape(-1, self.num_domains)  # (B, 2)
        return x


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
