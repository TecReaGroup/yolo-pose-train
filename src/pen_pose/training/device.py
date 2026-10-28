"""Training and evaluation device selection."""
import logging

import torch

LOGGER = logging.getLogger(__name__)


def select_device(requested: str | None) -> str:
    """Prefer CUDA when available and preserve explicit device selections."""
    if requested is not None:
        LOGGER.info('Using requested device: %s', requested)
        return requested
    if torch.cuda.is_available():
        LOGGER.info('Using CUDA device 0: %s', torch.cuda.get_device_name(0))
        return '0'
    LOGGER.warning('CUDA is unavailable in this Python environment; using CPU')
    return 'cpu'
