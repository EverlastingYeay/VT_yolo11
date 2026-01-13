# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import time
import warnings
from copy import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.nn.modules.vt import VTImageDomainTap, VTInstanceDomainTap
from ultralytics.utils import LOGGER, LOCAL_RANK, RANK, TQDM
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywhn
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import autocast, unwrap_model

from .train import DetectionTrainer
from .val import DetectionValidator


def _iter_domain_modules(model: nn.Module):
    # unwrap DDP to access actual module types
    model = unwrap_model(model)
    for m in model.modules():
        if isinstance(m, (VTImageDomainTap, VTInstanceDomainTap)):
            yield m


def _get_vt_logits(model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Collect stored logits from VT taps after a forward pass."""
    image_logits, instance_logits = [], []
    for m in _iter_domain_modules(model):
        if isinstance(m, VTImageDomainTap) and m.last_logits is not None:
            lg = m.last_logits
            # Defensive: image logits should be (B, 2) for CrossEntropyLoss.
            if lg.ndim == 1:
                lg = lg.unsqueeze(0)
            image_logits.append(lg)
        elif isinstance(m, VTInstanceDomainTap) and m.last_logits is not None:
            instance_logits.append(m.last_logits)
    return image_logits, instance_logits


def _set_instance_masks(model: nn.Module, masks: list[torch.Tensor] | None):
    """Set per-scale masks for VTInstanceDomainTap modules in model order."""
    taps = [m for m in _iter_domain_modules(model) if isinstance(m, VTInstanceDomainTap)]
    if masks is None:
        for t in taps:
            t.set_mask(None)
        return
    if len(masks) != len(taps):
        raise ValueError(f"Expected {len(taps)} instance masks, got {len(masks)}.")
    for t, mask in zip(taps, masks):
        t.set_mask(mask)

def _clear_vt_cache(model: nn.Module):
    """Clear cached tensors on VT tap modules to keep EMA deepcopy safe."""
    for m in _iter_domain_modules(model):
        if hasattr(m, "last_logits"):
            m.last_logits = None
        if hasattr(m, "set_mask"):
            m.set_mask(None)


def _build_instance_masks_from_labels(
    batch: dict[str, torch.Tensor],
    strides: torch.Tensor,
) -> list[torch.Tensor]:
    """Create (B,H,W) masks for each stride level from normalized xywh labels (VT-style).

    Aligns with VersatileTeacher `generate_mask_from_labels()`:
    - base noise: N(0,1)/50 + 0.05
    - inside-box region: ~0.85-0.95
    - Gaussian blur per scale (kernel sizes 7/5/3, sigma in [0.5, 1.5])
    """
    imgs = batch["img"]
    bboxes = batch["bboxes"]
    batch_idx = batch["batch_idx"]

    bsz, _, img_h, img_w = imgs.shape
    device = imgs.device

    masks: list[torch.Tensor] = []
    strides_list = [int(s) for s in strides.tolist()]
    kernel_sizes = [7, 5, 3]  # P3/P4/P5 like VT (80/40/20)
    for s in strides_list:
        gh, gw = int(img_h / s), int(img_w / s)
        masks.append(torch.randn((bsz, gh, gw), device=device, dtype=torch.float32) / 50.0 + 0.05)

    if bboxes.numel() == 0:
        return masks

    def _gaussian_blur_bhw(x: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
        if k <= 1:
            return x
        dtype = x.dtype
        ax = torch.arange(k, device=x.device, dtype=dtype) - (k - 1) / 2.0
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * (sigma**2)))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(x.unsqueeze(1), kernel, padding=k // 2).squeeze(1)

    # bboxes are normalized xywh (0..1), batch_idx indexes images in batch
    for bi, (xc, yc, bw, bh) in zip(batch_idx.tolist(), bboxes.tolist()):
        bi = int(bi)
        if bi < 0 or bi >= bsz:
            continue
        for mi, _s in enumerate(strides_list):
            gh, gw = masks[mi].shape[-2:]
            x = int(xc * gw)
            y = int(yc * gh)
            ww = int(bw * gw / 2.0)
            hh = int(bh * gh / 2.0)
            x1 = max(0, x - ww)
            x2 = min(gh - 1, x + ww)
            y1 = max(0, y - hh)
            y2 = min(gw - 1, y + hh)
            if x2 >= x1 and y2 >= y1:
                val = float(0.85 + torch.rand(1, device=device).item() * 0.1)
                masks[mi][bi, x1 : x2 + 1, y1 : y2 + 1] = val

    for mi in range(len(masks)):
        k = kernel_sizes[mi] if mi < len(kernel_sizes) else 3
        sigma = float(torch.empty(1, device=device).uniform_(0.5, 1.5).item())
        masks[mi] = _gaussian_blur_bhw(masks[mi], k=k, sigma=sigma)

    return masks


def _pseudo_labels_from_teacher_nms(
    nms_out: list[torch.Tensor],
    img_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (batch_idx, cls, bboxes, conf) tensors from Ultralytics NMS output."""
    img_h, img_w = img_hw
    batch_idx_list: list[torch.Tensor] = []
    cls_list: list[torch.Tensor] = []
    bboxes_list: list[torch.Tensor] = []
    conf_list: list[torch.Tensor] = []
    for i, det in enumerate(nms_out):
        if det is None or det.numel() == 0:
            continue
        # det: (N,6+extra) -> xyxy, conf, cls
        det = det[:, :6]
        xyxy = det[:, 0:4]
        conf = det[:, 4:5]
        cls = det[:, 5:6]
        bboxes = xyxy2xywhn(xyxy, w=img_w, h=img_h, clip=True)  # normalized xywh
        batch_idx_list.append(torch.full((det.shape[0],), i, device=device, dtype=torch.int64))
        cls_list.append(cls.to(device=device, dtype=torch.float32))
        bboxes_list.append(bboxes.to(device=device, dtype=torch.float32))
        conf_list.append(conf.to(device=device, dtype=torch.float32))

    if not batch_idx_list:
        return (
            torch.zeros((0,), device=device, dtype=torch.int64),
            torch.zeros((0, 1), device=device, dtype=torch.float32),
            torch.zeros((0, 4), device=device, dtype=torch.float32),
            torch.zeros((0, 1), device=device, dtype=torch.float32),
        )

    return torch.cat(batch_idx_list, 0), torch.cat(cls_list, 0), torch.cat(bboxes_list, 0), torch.cat(conf_list, 0)


def _apply_light_color_jitter(
    img: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
) -> torch.Tensor:
    """Apply lightweight color jitter (batch-wise) without changing geometry."""
    if brightness <= 0 and contrast <= 0 and saturation <= 0:
        return img
    out = img
    device = img.device
    if brightness > 0:
        b = float(torch.empty(1, device=device).uniform_(1.0 - brightness, 1.0 + brightness).item())
        out = out * b
    if contrast > 0:
        c = float(torch.empty(1, device=device).uniform_(1.0 - contrast, 1.0 + contrast).item())
        mean = out.mean(dim=(2, 3), keepdim=True)
        out = (out - mean) * c + mean
    if saturation > 0:
        s = float(torch.empty(1, device=device).uniform_(1.0 - saturation, 1.0 + saturation).item())
        gray = out.mean(dim=1, keepdim=True)
        out = (out - gray) * s + gray
    return out.clamp(0.0, 1.0)


def _make_mv_view(
    img: torch.Tensor,
    scale: float,
    hflip_p: float,
    brightness: float,
    contrast: float,
    saturation: float,
    stride: int,
) -> tuple[torch.Tensor, dict]:
    """Build a light multi-view for teacher and return mapping metadata."""
    bsz, _, h, w = img.shape
    scale = max(float(scale), 0.0)
    sf = 1.0
    if scale > 0:
        sf = float(torch.empty(1, device=img.device).uniform_(1.0 - scale, 1.0 + scale).item())
    new_h = max(int(round(h * sf / stride) * stride), stride)
    new_w = max(int(round(w * sf / stride) * stride), stride)
    out = img
    if new_h != h or new_w != w:
        out = F.interpolate(out, size=(new_h, new_w), mode="bilinear", align_corners=False)
    do_flip = False
    if hflip_p > 0 and float(torch.rand(1, device=img.device).item()) < hflip_p:
        out = torch.flip(out, dims=[3])
        do_flip = True
    out = _apply_light_color_jitter(out, brightness, contrast, saturation)
    meta = {
        "orig_hw": (h, w),
        "aug_hw": (new_h, new_w),
        "scale_x": float(new_w / w),
        "scale_y": float(new_h / h),
        "hflip": do_flip,
        "bsz": int(bsz),
    }
    return out, meta


def _remap_nms_boxes_to_orig(
    nms_out: list[torch.Tensor],
    orig_hw: tuple[int, int],
    aug_hw: tuple[int, int],
    scale_x: float,
    scale_y: float,
    hflip: bool,
) -> list[torch.Tensor]:
    """Remap NMS boxes from augmented view back to original image coordinates."""
    orig_h, orig_w = orig_hw
    aug_h, aug_w = aug_hw
    out: list[torch.Tensor] = []
    for det in nms_out:
        if det is None or det.numel() == 0:
            out.append(det)
            continue
        det = det.clone()
        xyxy = det[:, :4]
        if hflip:
            x1 = xyxy[:, 0].clone()
            x2 = xyxy[:, 2].clone()
            xyxy[:, 0] = aug_w - x2
            xyxy[:, 2] = aug_w - x1
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] / max(scale_x, 1e-9)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] / max(scale_y, 1e-9)
        xyxy[:, 0].clamp_(0, orig_w)
        xyxy[:, 2].clamp_(0, orig_w)
        xyxy[:, 1].clamp_(0, orig_h)
        xyxy[:, 3].clamp_(0, orig_h)
        det[:, :4] = xyxy
        out.append(det)
    return out


def _max_iou_per_det(det_ref: torch.Tensor, det_aug: torch.Tensor) -> torch.Tensor:
    """Compute per-det max IoU with same-class boxes."""
    device = det_ref.device
    n = det_ref.shape[0]
    if n == 0:
        return torch.zeros((0, 1), device=device, dtype=torch.float32)
    if det_aug is None or det_aug.numel() == 0:
        return torch.zeros((n, 1), device=device, dtype=torch.float32)
    cls_ref = det_ref[:, 5].to(dtype=torch.long)
    cls_aug = det_aug[:, 5].to(dtype=torch.long)
    ious = torch.zeros((n, 1), device=device, dtype=torch.float32)
    for c in cls_ref.unique().tolist():
        ref_idx = (cls_ref == c).nonzero(as_tuple=False).view(-1)
        aug_idx = (cls_aug == c).nonzero(as_tuple=False).view(-1)
        if ref_idx.numel() == 0 or aug_idx.numel() == 0:
            continue
        iou_mat = box_iou(det_ref[ref_idx, :4], det_aug[aug_idx, :4])
        ious[ref_idx] = iou_mat.max(1, keepdim=True)[0]
    return ious


def _consistency_iou_from_nms(
    nms_ref: list[torch.Tensor],
    nms_aug: list[torch.Tensor] | None,
    device: torch.device,
) -> torch.Tensor:
    """Concatenate per-box IoU consistency for all images in batch."""
    if not nms_ref:
        return torch.zeros((0, 1), device=device, dtype=torch.float32)
    iou_list: list[torch.Tensor] = []
    if nms_aug is None:
        for det_ref in nms_ref:
            if det_ref is None or det_ref.numel() == 0:
                continue
            iou_list.append(_max_iou_per_det(det_ref[:, :6], None))
    else:
        for det_ref, det_aug in zip(nms_ref, nms_aug):
            if det_ref is None or det_ref.numel() == 0:
                continue
            det_ref = det_ref[:, :6]
            det_aug = det_aug[:, :6] if (det_aug is not None and det_aug.numel()) else None
            iou_list.append(_max_iou_per_det(det_ref, det_aug))
    if not iou_list:
        return torch.zeros((0, 1), device=device, dtype=torch.float32)
    return torch.cat(iou_list, 0)


def _cal_density_from_nms(nms_out: list[torch.Tensor], nc: int, device: torch.device) -> torch.Tensor:
    """Compute per-class confidence density proxy from NMS output.

    This follows the VersatileTeacher implementation idea: for each class we build a 10-bin histogram over confidences
    and take the bin with maximum count as the class confidence "density" value.
    """
    density = torch.zeros((nc, 10), device=device)
    for det in nms_out:
        if det is None or det.numel() == 0:
            continue
        det = det[:, :6]  # xyxy, conf, cls
        conf = det[:, 4].clamp(max=0.99)
        cls = det[:, 5].to(dtype=torch.long)
        bins = (conf * 10).to(dtype=torch.long).clamp(0, 9)
        for c, b in zip(cls.tolist(), bins.tolist()):
            density[c, b] += 1
    return torch.argmax(density, dim=1).to(dtype=torch.float32) / 10.0


def _nms_per_class_conf_thres(
    prediction: torch.Tensor,
    cls_conf_thres: torch.Tensor,
    iou_thres: float,
    max_det: int,
    nc: int,
    max_wh: int = 7680,
    max_nms: int = 30000,
) -> list[torch.Tensor]:
    """NMS with per-class confidence thresholding (VT reproduction).

    Ultralytics `non_max_suppression()` currently only accepts float `conf_thres`. VersatileTeacher uses a per-class
    tensor for the second-stage NMS: keep a box if conf(box) > cls_conf_thres[argmax_class]. This function reproduces
    that behavior for YOLO11 outputs (BCN format).
    """
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    bs = prediction.shape[0]
    extra = prediction.shape[1] - nc - 4
    pred = prediction.transpose(-1, -2)  # (B, N, 4+nc+extra)
    pred[..., :4] = xywh2xyxy(pred[..., :4])

    # ensure thresholds on correct device
    cls_conf_thres = cls_conf_thres.to(device=prediction.device, dtype=torch.float32)

    import torchvision  # required for torchvision.ops.nms

    out: list[torch.Tensor] = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    for xi, x in enumerate(pred):
        box, cls, mask = x.split((4, nc, extra), 1)
        conf, j = cls.max(1, keepdim=True)  # (N,1)
        th = cls_conf_thres[j.view(-1).long()].view(-1, 1)
        keep = conf > th
        x = torch.cat((box, conf, j.float(), mask), 1)[keep.view(-1)]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * max_wh
        boxes = x[:, :4] + c
        scores = x[:, 4]
        i = torchvision.ops.nms(boxes, scores, float(iou_thres))[:max_det]
        out[xi] = x[i]
    return out


def _cal_quantile_from_nms(nms_out: list[torch.Tensor], nc: int, q: float, device: torch.device) -> torch.Tensor:
    """Compute per-class confidence quantile from NMS output (post-NMS).

    Returns zeros for classes with no detections.
    """
    confs_by_class: list[list[float]] = [[] for _ in range(nc)]
    for det in nms_out:
        if det is None or det.numel() == 0:
            continue
        det6 = det[:, :6]
        conf = det6[:, 4].clamp(max=0.999).detach().to("cpu")
        cls = det6[:, 5].to(dtype=torch.long).detach().to("cpu")
        for c, s in zip(cls.tolist(), conf.tolist()):
            if 0 <= c < nc:
                confs_by_class[c].append(float(s))

    out = torch.zeros((nc,), device=device, dtype=torch.float32)
    q = float(q)
    for c in range(nc):
        if confs_by_class[c]:
            t = torch.tensor(confs_by_class[c], dtype=torch.float32, device=device)
            out[c] = torch.quantile(t, q).clamp(0.0, 0.999)
    return out


class _CosineDecayWithWarmup:
    """VersatileTeacher-style cosine decay with warmup for CAPS smoothing."""

    def __init__(self, lr_min: float, lr_max: float, nb: int, total_epoch: int, warmup_epoch: int = 3):
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.warmup_nb = int(warmup_epoch * nb)
        self.total_nb = int(total_epoch * nb)

    def cal_alpha_conf(self, ni: int) -> float:
        if ni < self.warmup_nb:
            return ni / max(self.warmup_nb, 1) * self.lr_max
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos((ni - self.warmup_nb) / self.total_nb * np.pi))


def _apply_caps_filter(
    nms_out: list[torch.Tensor], cls_conf_thres: torch.Tensor, device: torch.device
) -> list[torch.Tensor]:
    """Apply class-adaptive confidence thresholding (CAPS) after NMS."""
    out: list[torch.Tensor] = []
    for det in nms_out:
        if det is None or det.numel() == 0:
            out.append(det)
            continue
        det6 = det[:, :6]
        cls = det6[:, 5].to(dtype=torch.long)
        keep = det6[:, 4] >= cls_conf_thres.to(device=device)[cls]
        out.append(det[keep])
    return out


def _linear_warmup(step: int, warmup_steps: int) -> float:
    """Return warmup factor in (0, 1], or 1.0 if disabled."""
    warmup_steps = int(warmup_steps)
    if warmup_steps <= 0:
        return 1.0
    step = max(int(step), 0)
    return min(1.0, (step + 1) / warmup_steps)


class VersatileTeacherTrainer(DetectionTrainer):
    """YOLO11 trainer with VersatileTeacher-style teacher-student domain adaptation."""

    def setup_model(self):
        # Ultralytics behavior: `pretrained=True` does not load weights when `model` is a YAML.

        if isinstance(self.args.pretrained, bool) and self.args.pretrained and bool(getattr(self.args, "vt_auto_pretrained", True)):
            model_path = str(self.args.model)
            if model_path.endswith((".yaml", ".yml")):
                from ultralytics.nn.tasks import yaml_model_load

                # Prefer a local `yolo11{scale}.pt` matching the YAML model scale (n/s/m/l/x), else fallback to `yolo11n.pt`.
                scale = (yaml_model_load(model_path) or {}).get("scale") or "n"
                candidates = [
                    Path.cwd() / f"yolo11{scale}.pt",
                    Path.cwd() / "yolo11n.pt",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        self.args.pretrained = str(candidate)
                        LOGGER.info(f"VT init: using pretrained weights {candidate}")
                        break
        return super().setup_model()

    def get_dataset(self):
        # Accept `data=[src.yaml, tgt.yaml]` (list/tuple) without guessing: rely on check_det_dataset.
        from ultralytics.data.utils import check_det_dataset

        data = self.args.data
        if isinstance(data, (list, tuple)) and len(data) == 2:
            self.data_src = check_det_dataset(data[0])
            self.data_tgt = check_det_dataset(data[1])
            # Some UDA setups provide source train only (no val). Ultralytics trainer expects a val/test split
            # during setup, so fall back to train if missing. We still validate on target by default.
            if not (self.data_src.get("val") or self.data_src.get("test")):
                self.data_src["val"] = self.data_src["train"]
            if "yaml_file" in self.data_src:
                self.args.data = self.data_src["yaml_file"]
            return self.data_src
        raise ValueError("VersatileTeacherTrainer requires `data` to be [source_yaml, target_yaml].")

    def _setup_train(self):
        super()._setup_train()

        self.args.vt_source_weak = bool(getattr(self.args, "vt_source_weak", True))
        self.args.vt_val_target = bool(getattr(self.args, "vt_val_target", True))
        self.args.vt_val_per_class = bool(getattr(self.args, "vt_val_per_class", True))
        self.args.use_instance_masks = bool(getattr(self.args, "use_instance_masks", True))
        self._best_map50 = -1.0
        self._best_map5095 = -1.0
        self._src_empty_steps = 0

        # Source augmentation policy:
        # - Recommended for UDA/VT: keep source "weak" (disable mosaic/mixup/copy-paste + strong HSV/geo),
        #   and do NOT apply the strong transformer on source.
        if self.args.vt_source_weak:
            gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
            batch_size = self.batch_size // max(self.world_size, 1)

            weak_src_args = copy(self.args)
            weak_src_args.mosaic = 0.0
            weak_src_args.mixup = 0.0
            weak_src_args.copy_paste = 0.0
            weak_src_args.cutmix = 0.0
            weak_src_args.hsv_h = 0.0
            weak_src_args.hsv_s = 0.0
            weak_src_args.hsv_v = 0.0
            weak_src_args.degrees = 0.0
            weak_src_args.translate = 0.0
            weak_src_args.scale = 0.0
            weak_src_args.shear = 0.0
            weak_src_args.perspective = 0.0
            weak_src_args.flipud = 0.0
            weak_src_args.fliplr = 0.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset_src = build_yolo_dataset(
                    weak_src_args, self.data["train"], batch=batch_size, data=self.data, mode="train", rect=False, stride=gs
                )
            self.train_loader = build_dataloader(
                dataset_src,
                batch=batch_size,
                workers=self.args.workers,
                shuffle=True,
                rank=LOCAL_RANK,
                drop_last=self.args.compile,
            )

        # Build target-domain train loader (same batch size/workers as source train loader).
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        batch_size = self.batch_size // max(self.world_size, 1)
        # Target weak augmentation: disable mosaic/mixup/copy-paste and strong HSV/geometric perturbations.
        weak_args = copy(self.args)
        weak_args.mosaic = 0.0
        weak_args.mixup = 0.0
        weak_args.copy_paste = 0.0
        weak_args.cutmix = 0.0
        weak_args.hsv_h = 0.0
        weak_args.hsv_s = 0.0
        weak_args.hsv_v = 0.0
        weak_args.degrees = 0.0
        weak_args.translate = 0.0
        weak_args.scale = 0.0
        weak_args.shear = 0.0
        weak_args.perspective = 0.0
        weak_args.flipud = 0.0
        weak_args.fliplr = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset_tgt = build_yolo_dataset(
                weak_args, self.data_tgt["train"], batch=batch_size, data=self.data_tgt, mode="train", rect=False, stride=gs
            )
        self.train_loader_tgt = build_dataloader(
            dataset_tgt,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=True,
            rank=LOCAL_RANK,
            drop_last=self.args.compile,
        )

        # Target-domain val loader/validator (GT is used ONLY for evaluation).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset_tgt_val = build_yolo_dataset(
                copy(self.args),
                self.data_tgt.get("val") or self.data_tgt.get("test"),
                batch=batch_size * 2,
                data=self.data_tgt,
                mode="val",
                rect=True,
                stride=gs,
            )
        self.test_loader_tgt = build_dataloader(
            dataset_tgt_val,
            batch=batch_size * 2,
            workers=self.args.workers * 2,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
        )
        self.validator_tgt = DetectionValidator(
            self.test_loader_tgt, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

        # VT hyperparameters (kept in args for checkpointing)
        self.args.vt_lambda_da = float(getattr(self.args, "vt_lambda_da", 0.1))
        self.args.vt_lambda_consensus = float(getattr(self.args, "vt_lambda_consensus", 0.1))
        self.args.vt_conf = float(getattr(self.args, "vt_conf", 0.25))
        self.args.vt_iou = float(getattr(self.args, "vt_iou", 0.45))
        self.args.vt_phase1_epochs = int(getattr(self.args, "vt_phase1_epochs", 0))
        self.args.vt_caps = bool(getattr(self.args, "vt_caps", True))
        self.args.vt_caps_init = float(getattr(self.args, "vt_caps_init", 0.8))
        self.args.vt_caps_q = float(getattr(self.args, "vt_caps_q", 0.9))
        self.args.vt_caps_gamma = float(getattr(self.args, "vt_caps_gamma", 2.0))
        self.args.vt_caps_tau = float(getattr(self.args, "vt_caps_tau", 0.05))
        self.args.vt_caps_soft = bool(getattr(self.args, "vt_caps_soft", True))
        self.args.vt_caps_hard = bool(getattr(self.args, "vt_caps_hard", True))
        self.args.vt_caps_hard_epochs = int(getattr(self.args, "vt_caps_hard_epochs", -1))
        self.args.vt_pl_warmup_epochs = int(getattr(self.args, "vt_pl_warmup_epochs", 0))
        self.args.vt_pl_weight_cap = float(getattr(self.args, "vt_pl_weight_cap", 1.0))
        self.args.vt_da_warmup_epochs = int(getattr(self.args, "vt_da_warmup_epochs", 0))
        self.args.vt_strong_enable = bool(getattr(self.args, "vt_strong_enable", True))
        self.args.vt_strong_brightness = float(getattr(self.args, "vt_strong_brightness", 0.5))
        self.args.vt_strong_contrast = float(getattr(self.args, "vt_strong_contrast", 0.5))
        self.args.vt_strong_saturation = float(getattr(self.args, "vt_strong_saturation", 0.5))
        self.args.vt_strong_hue = float(getattr(self.args, "vt_strong_hue", 0.5))
        self.args.vt_strong_blur_kernel = int(getattr(self.args, "vt_strong_blur_kernel", 5))
        self.args.vt_strong_erasing_p = float(getattr(self.args, "vt_strong_erasing_p", 0.5))
        self.args.vt_mv_enable = bool(getattr(self.args, "vt_mv_enable", True))
        self.args.vt_mv_scale = float(getattr(self.args, "vt_mv_scale", 0.1))
        self.args.vt_mv_hflip_p = float(getattr(self.args, "vt_mv_hflip_p", 0.5))
        self.args.vt_mv_brightness = float(getattr(self.args, "vt_mv_brightness", 0.1))
        self.args.vt_mv_contrast = float(getattr(self.args, "vt_mv_contrast", 0.1))
        self.args.vt_mv_saturation = float(getattr(self.args, "vt_mv_saturation", 0.1))
        self.args.vt_harmony_beta = float(getattr(self.args, "vt_harmony_beta", 0.5))
        self.args.vt_harmony_iou_min = float(getattr(self.args, "vt_harmony_iou_min", 0.0))
        self.args.vt_vis_enable = bool(getattr(self.args, "vt_vis_enable", False))
        self.args.vt_vis_interval = int(getattr(self.args, "vt_vis_interval", 200))
        self.args.vt_vis_dir = str(getattr(self.args, "vt_vis_dir", "runs/visual"))
        self.args.vt_vis_max_images = int(getattr(self.args, "vt_vis_max_images", 8))

        self._ce = nn.CrossEntropyLoss()
        self._bce = nn.BCEWithLogitsLoss()
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "img_da", "ins_da", "cons")

        # Strong augmentation for target student branch (configurable via YAML).
        strong_ops: list[nn.Module] = []
        if self.args.vt_strong_enable:
            cj = dict(
                brightness=self.args.vt_strong_brightness,
                contrast=self.args.vt_strong_contrast,
                saturation=self.args.vt_strong_saturation,
                hue=self.args.vt_strong_hue,
            )
            if any(float(v) > 0 for v in cj.values()):
                strong_ops.append(transforms.ColorJitter(**cj))
            k = int(self.args.vt_strong_blur_kernel)
            if k and k >= 3:
                if k % 2 == 0:  # GaussianBlur requires odd kernel size
                    k += 1
                strong_ops.append(transforms.GaussianBlur(kernel_size=k))
            if float(self.args.vt_strong_erasing_p) > 0:
                strong_ops.append(transforms.RandomErasing(p=float(self.args.vt_strong_erasing_p)))
        self._strong_transform = transforms.Compose(strong_ops)

        # CAPS (class-adaptive pseudo-label selection) state
        nc = int(getattr(unwrap_model(self.model).model[-1], "nc", 0) or 0)
        self._cls_conf_thres = torch.full((nc,), self.args.vt_caps_init, device=self.device, dtype=torch.float32)
        self._pl_cls_names = self.data_tgt.get("names", None) if isinstance(getattr(self, "data_tgt", None), dict) else None

    @staticmethod
    def _metric_value(metrics: dict, key: str) -> float | None:
        v = metrics.get(key, None)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _save_best_metrics_if_needed(self, metrics: dict):
        """Save best target validation metrics snapshots to JSON in save_dir."""
        if RANK not in {-1, 0}:
            return
        if not getattr(self, "validator_tgt", None):
            return

        map50 = self._metric_value(metrics, "metrics/mAP50(B)")
        map5095 = self._metric_value(metrics, "metrics/mAP50-95(B)")
        if map50 is None or map5095 is None:
            return

        # Structured per-class summary from DetMetrics
        per_class = []
        try:
            per_class = self.validator_tgt.metrics.summary(normalize=True, decimals=5)
        except Exception:
            per_class = []

        payload = {
            "epoch": int(self.epoch + 1),
            "overall": {
                "precision(B)": self._metric_value(metrics, "metrics/precision(B)"),
                "recall(B)": self._metric_value(metrics, "metrics/recall(B)"),
                "mAP50(B)": map50,
                "mAP50-95(B)": map5095,
            },
            "per_class": per_class,
        }
        payload["text"] = (
            f"epoch={payload['epoch']} "
            f"mAP50={payload['overall']['mAP50(B)']} mAP50-95={payload['overall']['mAP50-95(B)']}\n"
            f"per_class={len(per_class)}"
        )

        def _jsonify(obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            return obj

        if map50 > self._best_map50:
            self._best_map50 = map50
            (self.save_dir / "best_mAP50_metrics.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonify)
            )
        if map5095 > self._best_map5095:
            self._best_map5095 = map5095
            (self.save_dir / "best_mAP50_95_metrics.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonify)
            )

    @staticmethod
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return v

    def _save_epoch_csv(self, val_metrics: dict | None):
        """Append one epoch of VT training + target val metrics to results.csv."""
        if RANK not in {-1, 0}:
            return

        # Training losses (VT-extended) are stored in self.tloss (mean over steps), length 6.
        train_vals = {}
        if isinstance(getattr(self, "tloss", None), torch.Tensor) and self.tloss.numel() >= 6:
            names = ("box_loss", "cls_loss", "dfl_loss", "img_da", "ins_da", "cons")
            for k, x in zip(names, self.tloss.detach().cpu().tolist()[:6]):
                train_vals[f"train/{k}"] = self._to_float(x)

        # Target validation metrics (GT only for evaluation).
        val_vals = {}
        if isinstance(val_metrics, dict):
            key_map = {
                "metrics/precision(B)": "val_tgt/precision(B)",
                "metrics/recall(B)": "val_tgt/recall(B)",
                "metrics/mAP50(B)": "val_tgt/mAP50(B)",
                "metrics/mAP50-95(B)": "val_tgt/mAP50-95(B)",
                "val/box_loss": "val_tgt/box_loss",
                "val/cls_loss": "val_tgt/cls_loss",
                "val/dfl_loss": "val_tgt/dfl_loss",
            }
            for src_k, dst_k in key_map.items():
                if src_k in val_metrics:
                    val_vals[dst_k] = self._to_float(val_metrics[src_k])

        # Pseudo-label stats to help analyze VT effectiveness.
        pl_vals = {}
        denom = max(int(getattr(self, "_pl_img_total", 0)), 1)
        pl_vals["train/pl_total"] = int(getattr(self, "_pl_total", 0))
        pl_vals["train/pl_per_img"] = float(getattr(self, "_pl_total", 0) / denom)
        pl_vals["train/pl_img_ratio"] = float(getattr(self, "_pl_img_with", 0) / denom)
        pl_conf_count = int(getattr(self, "_pl_conf_count", 0))
        pl_conf_sum = float(getattr(self, "_pl_conf_sum", 0.0))
        pl_vals["train/pl_mean_conf"] = float(pl_conf_sum / max(pl_conf_count, 1))
        pl_iou_count = int(getattr(self, "_pl_iou_count", 0))
        pl_iou_sum = float(getattr(self, "_pl_iou_sum", 0.0))
        pl_vals["train/pl_mean_iou"] = float(pl_iou_sum / max(pl_iou_count, 1))
        pl_w_count = int(getattr(self, "_pl_w_count", 0))
        pl_w_sum = float(getattr(self, "_pl_w_sum", 0.0))
        pl_vals["train/pl_mean_weight"] = float(pl_w_sum / max(pl_w_count, 1))
        pl_vals["train/pl_weight_nz_ratio"] = float(int(getattr(self, "_pl_w_nz", 0)) / max(pl_w_count, 1))
        pl_vals["train/pl_warmup"] = float(getattr(self, "_pl_warmup", 1.0))
        pl_vals["train/da_warmup"] = float(getattr(self, "_da_warmup", 1.0))
        if hasattr(self, "_cls_conf_thres") and isinstance(self._cls_conf_thres, torch.Tensor):
            pl_vals["train/caps_thres_mean"] = float(self._cls_conf_thres.mean().detach().cpu())
        if hasattr(self, "_pl_conf_hist") and isinstance(self._pl_conf_hist, torch.Tensor):
            for bi, v in enumerate(self._pl_conf_hist.detach().cpu().to(dtype=torch.long).tolist()):
                pl_vals[f"train/pl_conf_hist/{bi}"] = int(v)
        if hasattr(self, "_pl_cls_hist") and isinstance(self._pl_cls_hist, torch.Tensor):
            cls_hist = self._pl_cls_hist.detach().cpu().to(dtype=torch.long).tolist()
            names = getattr(self, "_pl_cls_names", None)
            for ci, v in enumerate(cls_hist):
                if names and ci < len(names):
                    name = str(names[ci]).replace("/", "_").replace("\\", "_").strip() or str(ci)
                    pl_vals[f"train/pl_cls_hist/{name}"] = int(v)
                else:
                    pl_vals[f"train/pl_cls_hist/{ci}"] = int(v)

        # Learning rates
        lr_vals = {f"lr/pg{i}": float(x["lr"]) for i, x in enumerate(self.optimizer.param_groups)}

        self.save_metrics(metrics={**train_vals, **val_vals, **pl_vals, **lr_vals})

    def _maybe_save_pl_vis_json(
        self,
        step: int,
        img_paths: list[str] | None,
        bsz: int,
        img_hw: tuple[int, int],
        p_batch_idx: torch.Tensor,
        p_cls: torch.Tensor,
        p_conf: torch.Tensor,
        p_iou: torch.Tensor,
        p_w: torch.Tensor,
        p_bboxes: torch.Tensor,
    ) -> None:
        """Optionally dump pseudo-labels + weights to JSON for visualization."""
        if not getattr(self.args, "vt_vis_enable", False):
            return
        if RANK not in {-1, 0}:
            return
        interval = int(getattr(self.args, "vt_vis_interval", 0))
        if interval > 0 and (int(step) % interval) != 0:
            return
        vis_dir = Path(getattr(self.args, "vt_vis_dir", "runs/visual"))
        vis_dir.mkdir(parents=True, exist_ok=True)

        bsz = int(bsz)
        max_images = int(getattr(self.args, "vt_vis_max_images", 0)) or bsz
        max_images = max(0, min(max_images, bsz))
        if max_images == 0:
            return

        p_num = int(p_bboxes.shape[0]) if isinstance(p_bboxes, torch.Tensor) else 0
        if p_num:
            b_xyxy = xywh2xyxy(p_bboxes.detach().clone())
            b_xyxy[:, [0, 2]] *= float(img_hw[1])
            b_xyxy[:, [1, 3]] *= float(img_hw[0])
            xyxy = b_xyxy.cpu().tolist()
            bidx = p_batch_idx.detach().cpu().tolist()
            cls = p_cls.detach().view(-1).cpu().tolist()
            conf = p_conf.detach().view(-1).cpu().tolist()
            iou = p_iou.detach().view(-1).cpu().tolist()
            w = p_w.detach().view(-1).cpu().tolist()
        else:
            xyxy, bidx, cls, conf, iou, w = [], [], [], [], [], []

        per_img = [[] for _ in range(max_images)]
        for k in range(len(bidx)):
            bi = int(bidx[k])
            if bi < 0 or bi >= max_images:
                continue
            per_img[bi].append(
                {
                    "xyxy": [float(v) for v in xyxy[k]],
                    "cls": int(cls[k]) if cls else 0,
                    "conf": float(conf[k]) if conf else 0.0,
                    "iou": float(iou[k]) if iou else 0.0,
                    "weight": float(w[k]) if w else 0.0,
                    "name": (
                        str(self._pl_cls_names[int(cls[k])])
                        if getattr(self, "_pl_cls_names", None) and cls and int(cls[k]) < len(self._pl_cls_names)
                        else None
                    ),
                }
            )

        images = []
        img_paths = img_paths or [f"img_{i}" for i in range(max_images)]
        for i in range(max_images):
            images.append(
                {
                    "image": str(img_paths[i]),
                    "height": int(img_hw[0]),
                    "width": int(img_hw[1]),
                    "detections": per_img[i],
                }
            )

        payload = {
            "step": int(step),
            "epoch": int(getattr(self, "epoch", 0)),
            "images": images,
        }
        out_path = vis_dir / f"pl_{int(step):08d}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def validate(self):
        """Validate on target domain by default (GT only for evaluation)."""
        if not getattr(self.args, "vt_val_target", True):
            return super().validate()

        original_data = self.data
        original_validator = self.validator
        try:
            self.data = self.data_tgt
            self.validator = self.validator_tgt
            metrics = self.validator(self)
            if metrics is None:
                return None, None
            self._save_best_metrics_if_needed(metrics)
            fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
            if not self.best_fitness or self.best_fitness < fitness:
                self.best_fitness = fitness
            return metrics, fitness
        finally:
            self.data = original_data
            self.validator = original_validator
    def label_loss_items(self, loss_items=None, prefix="train"):
        # NOTE: Ultralytics Validator accumulates `model.loss(...)[1]` which is the task loss vector
        # (detect: box/cls/dfl => length 3). Our VT trainer additionally tracks DA losses for logging.
        base_names = ("box_loss", "cls_loss", "dfl_loss")
        names = base_names if prefix == "val" else self.loss_names
        keys = [f"{prefix}/{x}" for x in names]
        if loss_items is None:
            return keys

        loss_items = loss_items.detach() if isinstance(loss_items, torch.Tensor) else loss_items
        n = len(loss_items)
        if n == len(base_names):  # validator path
            keys = [f"{prefix}/{x}" for x in base_names]
        elif n != len(names):
            # Fallback: avoid shape mismatch if caller passes unexpected length
            keys = [f"{prefix}/loss{i}" for i in range(n)]
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))

    def _do_train(self):
        # Based on BaseTrainer._do_train(), adapted to joint source/target batches.
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb_src = len(self.train_loader)
        nb_tgt = len(self.train_loader_tgt)
        nb = max(nb_src, nb_tgt)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        cosine_decay = _CosineDecayWithWarmup(lr_min=0.000005, lr_max=0.00005, nb=nb, total_epoch=self.epochs, warmup_epoch=3)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {self.save_dir}\n"
            f"Starting training for {self.epochs} epochs..."
        )

        epoch = self.start_epoch
        self.optimizer.zero_grad()

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
                self.train_loader_tgt.sampler.set_epoch(epoch)

            # Phase transition: after phase-1, initialize student/teacher from phase-1 teacher (EMA) weights.
            if self.args.vt_phase1_epochs > 0 and epoch == self.args.vt_phase1_epochs:
                with torch.no_grad():
                    unwrap_model(self.model).load_state_dict(unwrap_model(self.ema.ema).state_dict(), strict=True)
                from ultralytics.utils.torch_utils import ModelEMA

                _clear_vt_cache(self.model)
                self.ema = ModelEMA(self.model)

            it_src = iter(self.train_loader)
            it_tgt = iter(self.train_loader_tgt)
            pbar = range(nb)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(pbar, total=nb)

            self.tloss = None
            self._pl_total = 0
            self._pl_img_with = 0
            self._pl_img_total = 0
            self._pl_w_sum = 0.0
            self._pl_w_count = 0
            self._pl_w_nz = 0
            self._pl_conf_sum = 0.0
            self._pl_conf_count = 0
            self._pl_iou_sum = 0.0
            self._pl_iou_count = 0
            self._pl_conf_hist = torch.zeros((10,), device="cpu", dtype=torch.long)
            nc = int(getattr(unwrap_model(self.model).model[-1], "nc", 0) or 0)
            self._pl_cls_hist = torch.zeros((nc,), device="cpu", dtype=torch.long)
            for i in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch

                # Warmup (same as BaseTrainer)
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                batch_src = next(it_src, None)
                if batch_src is None:
                    it_src = iter(self.train_loader)
                    batch_src = next(it_src)
                batch_tgt = next(it_tgt, None)
                if batch_tgt is None:
                    it_tgt = iter(self.train_loader_tgt)
                    batch_tgt = next(it_tgt)

                with autocast(self.amp):
                    batch_src = self.preprocess_batch(batch_src)
                    batch_tgt = self.preprocess_batch(batch_tgt)
                    img_tgt_weak = batch_tgt["img"]  # weak: no extra augmentation beyond dataloader pipeline

                    # Sanity-check: UDA requires labeled source. If source has no labels, detection loss collapses and mAP stays 0.
                    n_src = int(batch_src.get("cls", torch.zeros(0)).shape[0])
                    if n_src == 0:
                        self._src_empty_steps += 1
                    else:
                        self._src_empty_steps = 0
                    if self._src_empty_steps >= 50 and RANK in {-1, 0}:
                        raise RuntimeError(
                            "Source batches contain 0 labeled instances for 50 consecutive steps. "
                            "Please verify your source dataset is correctly labeled in YOLO format "
                            "(images + matching label txt files) and that the `train:` path in the source YAML is correct."
                        )

                    # --- Source: supervised detection loss + domain losses
                    strides = unwrap_model(self.model).stride
                    if self.args.use_instance_masks:
                        _set_instance_masks(self.model, _build_instance_masks_from_labels(batch_src, strides))
                    else:
                        _set_instance_masks(self.model, None)
                    loss_src, loss_items_src = self.model(batch_src)  # detection loss (sum over components)

                    img_logits_src, ins_logits_src = _get_vt_logits(self.model)
                    domain_src = torch.zeros((1,), device=self.device, dtype=torch.long)
                    img_da_src = sum(self._ce(lg, domain_src.expand(lg.shape[0])) for lg in img_logits_src) if img_logits_src else 0.0
                    ins_da_src = (
                        sum(self._bce(lg, torch.zeros_like(lg)) for lg in ins_logits_src) if ins_logits_src else 0.0
                    )

                    # --- Target:
                    # Phase-1 (domain-aware pretraining): no pseudo labels, only domain losses.
                    # Phase-2 (teacher-student): teacher NMS pseudo labels + target detection loss.
                    phase1 = self.args.vt_phase1_epochs > 0 and epoch < self.args.vt_phase1_epochs
                    phase2_step = epoch - int(self.args.vt_phase1_epochs) if not phase1 else 0
                    self._pl_warmup = _linear_warmup(phase2_step, self.args.vt_pl_warmup_epochs) if not phase1 else 0.0
                    self._da_warmup = _linear_warmup(phase2_step, self.args.vt_da_warmup_epochs) if not phase1 else 1.0

                    if phase1:
                        _set_instance_masks(self.model, None)
                        _ = self.model(img_tgt_weak)  # forward only to populate VT logits
                        loss_tgt = torch.zeros_like(loss_src)
                        loss_items_tgt = torch.zeros_like(loss_items_src)
                        img_logits_tgt, ins_logits_tgt = _get_vt_logits(self.model)
                    else:
                        teacher = self.ema.ema
                        teacher.eval()
                        with torch.no_grad():
                            preds = teacher(img_tgt_weak)
                            nc = int(getattr(unwrap_model(self.model).model[-1], "nc", 0) or 0)
                            # VT reproduction: CAPS uses density (hist argmax) + hard per-class threshold + 2nd NMS.
                            # 1) NMS at a fixed low threshold (0.2) to estimate confidence density.
                            nms_for_density = non_max_suppression(
                                preds,
                                conf_thres=0.2,
                                iou_thres=self.args.vt_iou,
                                max_det=1000,
                                nc=nc,
                            )
                            hard_caps = bool(self.args.vt_caps_hard) and (
                                int(self.args.vt_caps_hard_epochs) < 0
                                or int(phase2_step) < int(self.args.vt_caps_hard_epochs)
                            )
                            if self.args.vt_caps:
                                alpha_conf = 1.0 - cosine_decay.cal_alpha_conf(ni)
                                conf_delta = _cal_density_from_nms(nms_for_density, nc=nc, device=self.device)
                                # If density is ~0, keep old threshold (no objects in this batch for that class).
                                missing = conf_delta < 0.05
                                conf_delta[missing] = self._cls_conf_thres[missing]
                                self._cls_conf_thres = self._cls_conf_thres * alpha_conf + conf_delta * (1.0 - alpha_conf)
                                if hard_caps:
                                    # 2) Second-stage NMS with per-class confidence thresholds (hard filter).
                                    nms_out = _nms_per_class_conf_thres(
                                        preds,
                                        cls_conf_thres=self._cls_conf_thres,
                                        iou_thres=self.args.vt_iou,
                                        max_det=1000,
                                        nc=nc,
                                    )
                                else:
                                    # Pure soft-CAPS: avoid per-class hard filtering.
                                    nms_out = non_max_suppression(
                                        preds,
                                        conf_thres=self.args.vt_conf,
                                        iou_thres=self.args.vt_iou,
                                        max_det=1000,
                                        nc=nc,
                                    )
                            else:
                                # fallback: plain NMS
                                nms_out = non_max_suppression(
                                    preds,
                                    conf_thres=self.args.vt_conf,
                                    iou_thres=self.args.vt_iou,
                                    max_det=1000,
                                    nc=nc,
                                )
                            nms_out_mv = None
                            if self.args.vt_mv_enable:
                                mv_stride = int(strides.max().item()) if isinstance(strides, torch.Tensor) else int(strides)
                                img_tgt_mv, mv_meta = _make_mv_view(
                                    img_tgt_weak,
                                    scale=self.args.vt_mv_scale,
                                    hflip_p=self.args.vt_mv_hflip_p,
                                    brightness=self.args.vt_mv_brightness,
                                    contrast=self.args.vt_mv_contrast,
                                    saturation=self.args.vt_mv_saturation,
                                    stride=mv_stride,
                                )
                                preds_mv = teacher(img_tgt_mv)
                                if self.args.vt_caps and hard_caps:
                                    nms_out_mv = _nms_per_class_conf_thres(
                                        preds_mv,
                                        cls_conf_thres=self._cls_conf_thres,
                                        iou_thres=self.args.vt_iou,
                                        max_det=1000,
                                        nc=nc,
                                    )
                                else:
                                    nms_out_mv = non_max_suppression(
                                        preds_mv,
                                        conf_thres=self.args.vt_conf,
                                        iou_thres=self.args.vt_iou,
                                        max_det=1000,
                                        nc=nc,
                                    )
                                nms_out_mv = _remap_nms_boxes_to_orig(
                                    nms_out_mv,
                                    orig_hw=mv_meta["orig_hw"],
                                    aug_hw=mv_meta["aug_hw"],
                                    scale_x=mv_meta["scale_x"],
                                    scale_y=mv_meta["scale_y"],
                                    hflip=mv_meta["hflip"],
                                )
                            _, _, img_h, img_w = img_tgt_weak.shape

                            p_batch_idx, p_cls, p_bboxes, p_conf = _pseudo_labels_from_teacher_nms(
                                nms_out, img_hw=(img_h, img_w), device=batch_tgt["img"].device
                            )
                            if self.args.vt_mv_enable:
                                p_iou = _consistency_iou_from_nms(nms_out, nms_out_mv, device=batch_tgt["img"].device)
                            else:
                                p_iou = p_conf.clone()
                            if p_conf.numel() and p_iou.numel() == 0:
                                p_iou = p_conf.clone()
                            if p_iou.numel():
                                p_iou = p_iou.clamp(min=float(self.args.vt_harmony_iou_min), max=1.0)
                            beta = float(self.args.vt_harmony_beta)
                            p_conf_clamped = p_conf.clamp(0.0, 1.0)
                            w = (
                                (p_conf_clamped**beta) * (p_iou**(1.0 - beta))
                                if p_conf.numel()
                                else torch.zeros_like(p_conf)
                            )
                            if self.args.vt_caps and self.args.vt_caps_soft and p_conf.numel():
                                cls_ids = p_cls.detach().view(-1).to(dtype=torch.long)
                                cls_ids = cls_ids.clamp(0, self._cls_conf_thres.numel() - 1)
                                thr = self._cls_conf_thres.to(device=p_conf.device, dtype=p_conf.dtype)[cls_ids]
                                tau = max(float(self.args.vt_caps_tau), 1e-6)
                                w_caps = torch.sigmoid((p_conf.view(-1) - thr) / tau).view_as(p_conf)
                                gamma = float(self.args.vt_caps_gamma)
                                if gamma != 1.0:
                                    w_caps = w_caps**gamma
                                w = w * w_caps
                            if float(self.args.vt_pl_weight_cap) > 0:
                                w = w.clamp(max=float(self.args.vt_pl_weight_cap))
                            w_mean = float(w.mean().detach().cpu()) if w.numel() else 0.0
                            # Pseudo-label statistics for analysis
                            bsz = int(img_tgt_weak.shape[0])
                            self._pl_total += int(p_cls.shape[0])
                            self._pl_img_total += bsz
                            if p_batch_idx.numel():
                                self._pl_img_with += int(torch.unique(p_batch_idx).numel())
                            if p_conf.numel():
                                self._pl_conf_sum += float(p_conf.detach().sum().cpu())
                                self._pl_conf_count += int(p_conf.numel())
                                conf_bins = (p_conf.detach().flatten().cpu().clamp(0.0, 0.99) * 10.0).to(dtype=torch.long)
                                self._pl_conf_hist += torch.bincount(conf_bins, minlength=10).to(dtype=torch.long)
                            if p_iou.numel():
                                self._pl_iou_sum += float(p_iou.detach().sum().cpu())
                                self._pl_iou_count += int(p_iou.numel())
                            if w.numel():
                                self._pl_w_sum += float(w.detach().sum().cpu())
                                self._pl_w_count += int(w.numel())
                                self._pl_w_nz += int((w.detach() > 0).sum().cpu())
                            if p_cls.numel():
                                cls_ids = p_cls.detach().flatten().cpu().to(dtype=torch.long)
                                self._pl_cls_hist += torch.bincount(cls_ids, minlength=self._pl_cls_hist.numel()).to(dtype=torch.long)

                        self._maybe_save_pl_vis_json(
                            step=ni,
                            img_paths=batch_tgt.get("im_file", None),
                            bsz=int(img_tgt_weak.shape[0]),
                            img_hw=(img_h, img_w),
                            p_batch_idx=p_batch_idx,
                            p_cls=p_cls,
                            p_conf=p_conf,
                            p_iou=p_iou,
                            p_w=w,
                            p_bboxes=p_bboxes,
                        )

                        pseudo_batch = dict(batch_tgt)
                        pseudo_batch["batch_idx"] = p_batch_idx
                        pseudo_batch["cls"] = p_cls
                        pseudo_batch["bboxes"] = p_bboxes
                        pseudo_batch["weights"] = w
                        # strong: apply additional strong augmentation only on the student branch
                        pseudo_batch["img"] = self._strong_transform(img_tgt_weak) if self.args.vt_strong_enable else img_tgt_weak

                        if self.args.use_instance_masks:
                            _set_instance_masks(self.model, _build_instance_masks_from_labels(pseudo_batch, strides))
                        else:
                            _set_instance_masks(self.model, None)
                        loss_tgt, loss_items_tgt = self.model(pseudo_batch)
                        # VT reproduction: hard CAPS -> no soft weighting. Optional warmup can still be enabled via cfg.
                        loss_tgt = loss_tgt * float(self._pl_warmup)

                        img_logits_tgt, ins_logits_tgt = _get_vt_logits(self.model)

                    domain_tgt = torch.ones((1,), device=self.device, dtype=torch.long)
                    img_da_tgt = sum(self._ce(lg, domain_tgt.expand(lg.shape[0])) for lg in img_logits_tgt) if img_logits_tgt else 0.0
                    ins_da_tgt = (
                        sum(self._bce(lg, torch.ones_like(lg)) for lg in ins_logits_tgt) if ins_logits_tgt else 0.0
                    )

                    # Consensus: encourage instance domain prob-map to align with image-level domain prob
                    cons = 0.0
                    if img_logits_src and ins_logits_src:
                        # Use P(domain=1) from image logits as a per-image target
                        for il, sl in zip(img_logits_src, ins_logits_src):
                            p = il.softmax(1)[:, 1].view(-1, 1, 1)
                            cons = cons + torch.mean((sl.sigmoid() - p) ** 2)
                    if img_logits_tgt and ins_logits_tgt:
                        for il, sl in zip(img_logits_tgt, ins_logits_tgt):
                            p = il.softmax(1)[:, 1].view(-1, 1, 1)
                            cons = cons + torch.mean((sl.sigmoid() - p) ** 2)

                    img_da = img_da_src + img_da_tgt
                    ins_da = ins_da_src + ins_da_tgt
                    da_scale = float(self._da_warmup)
                    vt_da = da_scale * (self.args.vt_lambda_da * (img_da + ins_da) + self.args.vt_lambda_consensus * cons)

                    loss = loss_src.sum() + (0.0 if phase1 else loss_tgt.sum()) + vt_da
                    loss_items = torch.cat(
                        (
                            loss_items_src.detach(),
                            torch.tensor(
                                [float(img_da.detach()), float(ins_da.detach()), float(cons.detach())],
                                device=self.device,
                                dtype=loss_items_src.dtype,
                            ),
                        ),
                        0,
                    )
                    self.loss = loss
                    # Keep task loss items (box/cls/dfl) for Ultralytics Validator compatibility.
                    self.loss_items = loss_items_src.detach()
                    # Track VT-extended items separately for progress display.
                    self.vt_loss_items = loss_items

                    # Important: clear cached tensors so ModelEMA deepcopy remains safe (phase switch etc.).
                    # The autograd graph is already retained by `loss` and does not rely on module attributes.
                    _clear_vt_cache(self.model)
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = (
                        self.vt_loss_items
                        if self.tloss is None
                        else (self.tloss * i + self.vt_loss_items) / (i + 1)
                    )

                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    losses = self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *losses,
                            batch_src["cls"].shape[0],
                            batch_src["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")

                self.run_callbacks("on_train_batch_end")

            self.epoch_time = time.time() - self.epoch_time_start
            self.epoch_time_start = time.time()
            self.run_callbacks("on_train_epoch_end")

            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                self.save_model()
                self.metrics, self.fitness = self.validate()
                self._save_epoch_csv(self.metrics)
                self.stop |= (epoch + 1) >= self.epochs

            self.run_callbacks("on_fit_epoch_end")
            if self.stop:
                break
            epoch += 1
