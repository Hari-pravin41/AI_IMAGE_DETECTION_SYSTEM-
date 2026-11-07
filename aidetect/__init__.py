"""aidetect - multi-signal forensic detection of AI-generated images."""

__version__ = "1.0.0"

from .config import Config  # noqa: E402
from .detector import Detector, Prediction  # noqa: E402

__all__ = ["Config", "Detector", "Prediction", "__version__"]
