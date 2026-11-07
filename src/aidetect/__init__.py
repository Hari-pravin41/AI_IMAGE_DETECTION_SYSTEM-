"""aidetect - forensic AI-generated image detection you can train on your own data."""
__version__ = "1.0.0"

__all__ = ["__version__", "Config", "Detector", "train_model", "evaluate_model"]


def __getattr__(name):          # lazy imports keep ``import aidetect`` fast
    if name == "Config":
        from .config import Config
        return Config
    if name == "Detector":
        from .detector import Detector
        return Detector
    if name == "train_model":
        from .train import train_model
        return train_model
    if name == "evaluate_model":
        from .evaluate import evaluate_model
        return evaluate_model
    raise AttributeError(f"module 'aidetect' has no attribute {name!r}")
