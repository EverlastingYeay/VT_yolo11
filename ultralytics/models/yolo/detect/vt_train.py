# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import time
import warnings
from copy import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.nn.modules.vt import VTImageDomainTap, VTInstanceDomainTap
from ultralytics.utils import LOGGER, LOCAL_RANK, RANK, TQDM
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xyxy2xywhn
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
            image_logits.append(m.last_logits)
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


def _build_instance_masks_from_labels(
    batch: dict[str, torch.Tensor],
    strides: torch.Tensor,
) -> list[torch.Tensor]:
    """Create (B,H,W) masks for each stride level from normalized xywh labels."""
    imgs = batch["img"]
    bboxes = batch["bboxes"]
    batch_idx = batch["batch_idx"]

    bsz, _, img_h, img_w = imgs.shape
    device = imgs.device

    # base mask values
    masks: list[torch.Tensor] = []
    for s in strides.tolist():
        gh, gw = int(img_h / s), int(img_w / s)
        masks.append(torch.full((bsz, gh, gw), 0.05, device=device, dtype=torch.float32))

    if bboxes.numel() == 0:
        return masks

    # bboxes are normalized xywh (0..1), batch_idx indexes images in batch
    for bi, (xc, yc, bw, bh) in zip(batch_idx.tolist(), bboxes.tolist()):
        bi = int(bi)  # batch_idx is float in Ultralytics Format() then collated; cast for indexing
        if bi < 0 or bi >= bsz:
            continue
        # convert to pixel xyxy
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h

        for mi, s in enumerate(strides.tolist()):
            gh, gw = masks[mi].shape[-2:]
            gx1 = int(max(0, math.floor(x1 / s)))
            gy1 = int(max(0, math.floor(y1 / s)))
            gx2 = int(min(gw - 1, math.ceil(x2 / s)))
            gy2 = int(min(gh - 1, math.ceil(y2 / s)))
            if gx2 >= gx1 and gy2 >= gy1:
                masks[mi][bi, gy1 : gy2 + 1, gx1 : gx2 + 1] = 0.95

    return masks


def _pseudo_labels_from_teacher_nms(
    nms_out: list[torch.Tensor],
    img_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (batch_idx, cls, bboxes) tensors from Ultralytics NMS output."""
    img_h, img_w = img_hw
    batch_idx_list: list[torch.Tensor] = []
    cls_list: list[torch.Tensor] = []
    bboxes_list: list[torch.Tensor] = []
    for i, det in enumerate(nms_out):
        if det is None or det.numel() == 0:
            continue
        # det: (N,6+extra) -> xyxy, conf, cls
        det = det[:, :6]
        xyxy = det[:, 0:4]
        cls = det[:, 5:6]
        bboxes = xyxy2xywhn(xyxy, w=img_w, h=img_h, clip=True)  # normalized xywh
        batch_idx_list.append(torch.full((det.shape[0],), i, device=device, dtype=torch.int64))
        cls_list.append(cls.to(device=device, dtype=torch.float32))
        bboxes_list.append(bboxes.to(device=device, dtype=torch.float32))

    if not batch_idx_list:
        return (
            torch.zeros((0,), device=device, dtype=torch.int64),
            torch.zeros((0, 1), device=device, dtype=torch.float32),
            torch.zeros((0, 4), device=device, dtype=torch.float32),
        )

    return torch.cat(batch_idx_list, 0), torch.cat(cls_list, 0), torch.cat(bboxes_list, 0)


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


class VersatileTeacherTrainer(DetectionTrainer):
    """YOLO11 trainer with VersatileTeacher-style teacher-student domain adaptation."""

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
        self._best_map50 = -1.0
        self._best_map5095 = -1.0

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

        self._ce = nn.CrossEntropyLoss()
        self._bce = nn.BCEWithLogitsLoss()
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "img_da", "ins_da", "cons")

        # Strong augmentation for target student branch (matches VersatileTeacher torchvison implementation).
        self._strong_transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.GaussianBlur(kernel_size=5),
                transforms.RandomErasing(p=0.5),
            ]
        )

        # CAPS (class-adaptive pseudo-label selection) state
        nc = int(getattr(unwrap_model(self.model).model[-1], "nc", 0) or 0)
        self._cls_conf_thres = torch.full((nc,), 0.8, device=self.device, dtype=torch.float32)

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

                self.ema = ModelEMA(self.model)

            it_src = iter(self.train_loader)
            it_tgt = iter(self.train_loader_tgt)
            pbar = range(nb)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(pbar, total=nb)

            self.tloss = None
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

                    # --- Source: supervised detection loss + domain losses
                    strides = unwrap_model(self.model).stride
                    _set_instance_masks(self.model, _build_instance_masks_from_labels(batch_src, strides))
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
                            nms_out = non_max_suppression(
                                preds,
                                conf_thres=self.args.vt_conf,
                                iou_thres=self.args.vt_iou,
                                max_det=300,
                                nc=int(getattr(unwrap_model(self.model).model[-1], "nc", 0) or 0),
                            )
                            # CAPS update + filter (Ultralytics NMS is global-threshold; CAPS is applied post-NMS)
                            alpha_conf = 1.0 - cosine_decay.cal_alpha_conf(ni)
                            conf_delta = _cal_density_from_nms(nms_out, nc=self._cls_conf_thres.numel(), device=self.device)
                            low = conf_delta < 0.05
                            conf_delta[low] = self._cls_conf_thres[low]
                            self._cls_conf_thres = self._cls_conf_thres * alpha_conf + conf_delta * (1.0 - alpha_conf)
                            nms_out = _apply_caps_filter(nms_out, self._cls_conf_thres, device=self.device)

                            _, _, h, w = img_tgt_weak.shape
                            p_batch_idx, p_cls, p_bboxes = _pseudo_labels_from_teacher_nms(
                                nms_out, img_hw=(h, w), device=batch_tgt["img"].device
                            )

                        pseudo_batch = dict(batch_tgt)
                        pseudo_batch["batch_idx"] = p_batch_idx
                        pseudo_batch["cls"] = p_cls
                        pseudo_batch["bboxes"] = p_bboxes
                        # strong: apply additional strong augmentation only on the student branch
                        pseudo_batch["img"] = self._strong_transform(img_tgt_weak)

                        _set_instance_masks(self.model, _build_instance_masks_from_labels(pseudo_batch, strides))
                        loss_tgt, loss_items_tgt = self.model(pseudo_batch)

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
                    vt_da = self.args.vt_lambda_da * (img_da + ins_da) + self.args.vt_lambda_consensus * cons

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
                self.stop |= (epoch + 1) >= self.epochs

            self.run_callbacks("on_fit_epoch_end")
            if self.stop:
                break
            epoch += 1
