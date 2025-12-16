# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .vt_train import VersatileTeacherTrainer
from .val import DetectionValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "VersatileTeacherTrainer"
