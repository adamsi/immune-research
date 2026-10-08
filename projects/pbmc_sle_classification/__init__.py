import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "configs", "default.yaml")


def path(rel: str) -> str:
    """Resolve a config path relative to this project folder."""
    return os.path.join(PROJECT_DIR, rel)
