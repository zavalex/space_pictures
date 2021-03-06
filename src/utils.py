import torch
import numpy as np
import random

def torch_fix_seed(seed=42) -> None:
    '''Fixes some of random seeds.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.
    '''
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True