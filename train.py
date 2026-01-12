from ultralytics.models.yolo.detect import VersatileTeacherTrainer


def main():
    trainer = VersatileTeacherTrainer(
        overrides={
            # Reproduction preset (closest to original VersatileTeacher defaults).
            # If you switch to yolo11m/yolo11l, ensure the matching pretrained weights exist locally
            # (e.g. yolo11m.pt/yolo11l.pt) or set `pretrained` to the correct path explicitly.
            "model": "ultralytics/cfg/models/11/yolo11-vt.yaml",
            "data": ["cityscapes.yaml", "cityscapes_foggy.yaml"],
            "pretrained": "yolo11l.pt",
            "vt_phase1_epochs": 150,
            "vt_lambda_da": 0.1,
            "vt_lambda_consensus": 0.1,
            "vt_conf": 0.2,
            "vt_iou": 0.45,
            "vt_caps": True,
            "vt_caps_init": 0.8,
            "vt_mv_enable": True,
            "vt_mv_scale": 0.1,
            "vt_mv_hflip_p": 0.5,
            "vt_mv_brightness": 0.1,
            "vt_mv_contrast": 0.1,
            "vt_mv_saturation": 0.1,
            "vt_harmony_beta": 0.5,
            "vt_harmony_iou_min": 0.0,
            "use_instance_masks": True,
            "vt_source_weak": False,
        }
    )
    trainer.train()


if __name__ == "__main__":
    # Required on Windows for PyTorch DataLoader multiprocessing ("spawn").
    main()
