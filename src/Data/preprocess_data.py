import tensorflow_hub as hub
import os
import sys
import scipy.io as scio
import numpy as np
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_encoder(model_path: Path):
    """
    Load Universal Sentence Encoder model.

    Args:
        model_path: Path to the Universal Sentence Encoder model directory

    Returns:
        Loaded encoder model with 512-dimensional output

    Raises:
        FileNotFoundError: If model path does not exist
        Exception: If model fails to load
    """
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        logger.info(f"Loading encoder from {model_path}")
        encoder = hub.load(str(model_path))
        logger.info("Encoder loaded successfully")
        return encoder
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load encoder: {e}")
        raise

if __name__ == "__main__":
    try:
        encoder = load_encoder(config.ENCODER_MODEL_PATH)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
