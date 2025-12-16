from ultralytics.models.yolo.detect import VersatileTeacherTrainer


def main():
    trainer = VersatileTeacherTrainer(
        overrides={
            "model": "ultralytics/cfg/models/11/yolo11-vt.yaml",
            "data": ["cityscapes.yaml", "cityscapes_foggy.yaml"],
            "vt_lambda_da": 0.1,
            "vt_lambda_consensus": 0.1,
            "vt_conf": 0.25,
            "vt_iou": 0.45,
            "vt_phase1_epochs": 50,
        }
    )
    trainer.train()


if __name__ == "__main__":
    # Required on Windows for PyTorch DataLoader multiprocessing ("spawn").
    main()
