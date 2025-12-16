# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_
import numpy as np
import cv2
from scipy.fftpack import dct, idct

__all__ = "inverse_sigmoid", "multi_scale_deformable_attn_pytorch"


def _get_clones(module, n):
    """Create a list of cloned modules from the given module.

    Args:
        module (nn.Module): The module to be cloned.
        n (int): Number of clones to create.

    Returns:
        (nn.ModuleList): A ModuleList containing n clones of the input module.

    Examples:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value.

    This function calculates the bias initialization value based on a prior probability using the inverse error
    function. It's commonly used in object detection models to initialize classification layers with a specific positive
    prediction probability.

    Args:
        prior_prob (float, optional): Prior probability for bias initialization.

    Returns:
        (float): Bias initialization value calculated from the prior probability.

    Examples:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Bias initialization value: {bias:.4f}")
        Bias initialization value: -4.5951
    """
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init(module):
    """Initialize the weights and biases of a linear module.

    This function initializes the weights of a linear module using a uniform distribution within bounds calculated from
    the input dimension. If the module has a bias, it is also initialized.

    Args:
        module (nn.Module): Linear module to initialize.

    Returns:
        (nn.Module): The initialized module.

    Examples:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> initialized_linear = linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor.

    This function applies the inverse of the sigmoid function to a tensor, which is useful in various neural network
    operations, particularly in attention mechanisms and coordinate transformations.

    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1].
        eps (float, optional): Small epsilon value to prevent numerical instability.

    Returns:
        (torch.Tensor): Tensor after applying the inverse sigmoid function.

    Examples:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Implement multi-scale deformable attention in PyTorch.

    This function performs deformable attention across multiple feature map scales, allowing the model to attend to
    different spatial locations with learned offsets.

    Args:
        value (torch.Tensor): The value tensor with shape (bs, num_keys, num_heads, embed_dims).
        value_spatial_shapes (torch.Tensor): Spatial shapes of the value tensor with shape (num_levels, 2).
        sampling_locations (torch.Tensor): The sampling locations with shape (bs, num_queries, num_heads, num_levels,
            num_points, 2).
        attention_weights (torch.Tensor): The attention weights with shape (bs, num_queries, num_heads, num_levels,
            num_points).

    Returns:
        (torch.Tensor): The output tensor with shape (bs, num_queries, embed_dims).

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()




def dct(x):
    return dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct(x):
    return idct(idct(x, axis=0, norm="ortho"), axis=1, norm="ortho")


def dct_scg(
    img_bgr,
    targets,
    R_L=0.25,
    R_H=0.6,
    bg_strength=(0.35, 0.08, 0.08),   # (Y, U, V)
    fg_strength=(0.0, 0.0, 0.0),
    mix_alpha=0.6,
    freq_gamma=1.0,
):
    """
    Source-only Frequency-domain SCG (Improved)


    Args:
        img_bgr: uint8, BGR, (H, W, 3)   [ONLY SOURCE DOMAIN]
        targets: N x 5, (cls, cx, cy, w, h), normalized
        R_L, R_H: ring band-pass ratios
        bg_strength: per-channel background perturb strength (Y, U, V)
        fg_strength: per-channel foreground perturb strength (Y, U, V)
        mix_alpha: blend ratio
        freq_gamma: frequency weighting exponent

    Returns:
        out_bgr: uint8
    """

    H, W = img_bgr.shape[:2]

    # --------------------------------------------------
    # 1. Foreground / Background mask
    # --------------------------------------------------
    fg_mask = np.zeros((H, W), dtype=np.float32)

    for tgt in targets:
        if len(tgt) >= 5:
            _, cx, cy, bw, bh = tgt[:5]
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            fg_mask[y1:y2, x1:x2] = 1.0

    bg_mask = 1.0 - fg_mask

    # --------------------------------------------------
    # 2. BGR -> YUV
    # --------------------------------------------------
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV).astype(np.float32) / 255.0

    # --------------------------------------------------
    # 3. Frequency grid & ring mask
    # --------------------------------------------------
    u = np.arange(H)[:, None]
    v = np.arange(W)[None, :]
    r = np.sqrt(u ** 2 + v ** 2)
    r_norm = r / (r.max() + 1e-6)

    ring_mask = (
        np.exp(-(r_norm ** 2) / (2 * R_H ** 2))
        - np.exp(-(r_norm ** 2) / (2 * R_L ** 2))
    )
    ring_mask = np.clip(ring_mask, 0.0, 1.0)

    freq_weight = r_norm ** freq_gamma

    # --------------------------------------------------
    # 4. Channel-wise DCT + causal / non-causal perturb
    # --------------------------------------------------
    img_yuv_aug = img_yuv.copy()

    for c in range(3):  # Y, U, V
        X = img_yuv[:, :, c]
        X_dct = dct(X)

        C = X_dct * ring_mask
        S = X_dct * (1.0 - ring_mask)

        noise = np.random.randn(H, W)

        strength_map = (
            bg_strength[c] * bg_mask
            + fg_strength[c] * fg_mask
        )

        S_perturbed = S * (1.0 + noise * strength_map * freq_weight)

        X_dct_new = C + S_perturbed
        X_new = idct(X_dct_new)
        img_yuv_aug[:, :, c] = np.clip(X_new, 0.0, 1.0)

    # --------------------------------------------------
    # 5. YUV -> BGR
    # --------------------------------------------------
    out_aug = cv2.cvtColor(
        (img_yuv_aug * 255.0).astype(np.uint8),
        cv2.COLOR_YUV2BGR
    )

    # --------------------------------------------------
    # 6. Linear blend
    # --------------------------------------------------
    out = (
        mix_alpha * out_aug.astype(np.float32)
        + (1.0 - mix_alpha) * img_bgr.astype(np.float32)
    )
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out
