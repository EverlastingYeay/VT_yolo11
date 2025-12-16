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
    """Image-level domain classifier that does not change the forward tensor.

    This module is intended to be inserted into the model graph as a 'tap': it runs a
    small classifier on the incoming feature map, stores the logits to be consumed by
    the Trainer, and returns the original feature map unchanged.
    """

    def __init__(self, c1: int, grl_lambda: float = 0.1, num_domains: int = 2):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.num_domains = int(num_domains)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c1, self.num_domains)
        self.last_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: during model initialization Ultralytics runs an eval() forward to infer strides.
        # Storing non-leaf tensors (with grad_fn) as attributes breaks deepcopy() used by ModelEMA.
        if not self.training:
            self.last_logits = None
            return x

        z = grad_reverse(x, self.grl_lambda)
        z = self.pool(z).flatten(1)
        self.last_logits = self.fc(z)
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
