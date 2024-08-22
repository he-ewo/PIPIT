import torch

from . import (
    classifier,
    gan,
    logger,
    path,
)

# random seed
seed = 0

# device used for training
device: str = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
