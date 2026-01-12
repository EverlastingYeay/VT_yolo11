from ultralytics.models.yolo.detect import VersatileTeacherTrainer


BASE_OVERRIDES = {
    # Common VT setup.
    "model": "ultralytics/cfg/models/11/yolo11s-vt.yaml",
    "pretrained": "yolo11s.pt",
    "vt_lambda_da": 0.1,
    "vt_lambda_consensus": 0.1,
    "vt_iou": 0.45,
    "vt_caps": True,
    "vt_loc_quality": True,
    "vt_loc_q_weight": True,
    "vt_cons_iou": True,
    "vt_cons_iou_thr": 0.0,
    "vt_cons_iou_gamma": 1.0,
    "vt_cons_iou_alpha": 0.2,
    "vt_quality_min": 0.1,
    "vt_conf_floor": 0.05,
    "vt_pl_min_per_img": 1,
    "use_instance_masks": True,
    "vt_source_weak": False,
    "vt_strong_enable": True,
    "batch": 4,
}

PRESETS = {
    # Moderate shift, multi-class.
    "cityscapes_to_foggy": {
        "data": ["cityscapes.yaml", "cityscapes_foggy.yaml"],
        "vt_phase1_epochs": 120,
        "vt_conf": 0.2,
        "vt_caps_init": 0.75,
        "vt_caps_gamma": 1.6,
        "vt_caps_tau": 0.06,
        "vt_sched_epochs": 80,
        "vt_caps_init_final": 0.55,
        "vt_caps_gamma_final": 1.2,
        "vt_caps_tau_final": 0.08,
        "vt_loc_q_thr": 0.1,
        "vt_loc_q_gamma": 1.2,
        "vt_loc_q_thr_final": 0.0,
        "vt_loc_q_gamma_final": 1.0,
        "vt_pl_warmup_epochs": 5,
        "vt_da_warmup_epochs": 5,
        "vt_cls_curriculum": True,
        "vt_cls_thres_min": 0.3,
        "vt_cls_thres_max": 0.8,
        "vt_cls_delta": 0.02,
        "vt_cls_tau": 0.1,
    },
    # Large shift, synthetic -> real, single-class.
    "sim10k_to_cityscapes_car": {
        "data": ["sim10k.yaml", "cityscapes.yaml"],
        "vt_phase1_epochs": 200,
        "vt_conf": 0.25,
        "vt_caps_init": 0.85,
        "vt_caps_gamma": 2.0,
        "vt_caps_tau": 0.04,
        "vt_sched_epochs": 120,
        "vt_caps_init_final": 0.6,
        "vt_caps_gamma_final": 1.5,
        "vt_caps_tau_final": 0.08,
        "vt_loc_q_thr": 0.2,
        "vt_loc_q_gamma": 1.5,
        "vt_loc_q_thr_final": 0.05,
        "vt_loc_q_gamma_final": 1.2,
        "vt_pl_warmup_epochs": 10,
        "vt_da_warmup_epochs": 10,
        "vt_cls_curriculum": False,
    },
    # Medium shift, real -> real, single-class.
    "kitti_to_cityscapes_car": {
        "data": ["kitti.yaml", "cityscapes.yaml"],
        "vt_phase1_epochs": 150,
        "vt_conf": 0.22,
        "vt_caps_init": 0.78,
        "vt_caps_gamma": 1.6,
        "vt_caps_tau": 0.05,
        "vt_sched_epochs": 100,
        "vt_caps_init_final": 0.55,
        "vt_caps_gamma_final": 1.2,
        "vt_caps_tau_final": 0.08,
        "vt_loc_q_thr": 0.12,
        "vt_loc_q_gamma": 1.3,
        "vt_loc_q_thr_final": 0.03,
        "vt_loc_q_gamma_final": 1.0,
        "vt_pl_warmup_epochs": 5,
        "vt_da_warmup_epochs": 8,
        "vt_cls_curriculum": False,
    },
}


def build_overrides(preset_name: str) -> dict:
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset '{preset_name}'. Available: {sorted(PRESETS)}")
    overrides = dict(BASE_OVERRIDES)
    overrides.update(PRESETS[preset_name])
    return overrides


def main():
    # Select a preset and adjust dataset paths if your YAML names differ.
    preset = "cityscapes_to_foggy"
    trainer = VersatileTeacherTrainer(overrides=build_overrides(preset))
    trainer.train()


if __name__ == "__main__":
    # Required on Windows for PyTorch DataLoader multiprocessing ("spawn").
    main()
