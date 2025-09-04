# -*- encoding: utf-8 -*-
'''

Test script

@File    :   test.py
@Time    :   2025/09/02 15:14:44
@Author  :   Yangshuo He
@Contact :   sugarhe58@gmail.com
'''

import torch
import numpy as np
from pbb.utils import test_exp

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA: {torch.cuda.is_available()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print("MPS: ", torch.backends.mps.is_available())
    print(f"Using device: {device}")

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device == 'mps':
        torch.use_deterministic_algorithms(True)

    test_exp(outage=0.1, device=device)

    test_exp(outage=0.2, device=device)

    test_exp(outage=0.3, device=device)

    test_exp(outage=0.4, device=device)

    test_exp(outage=0.5, device=device)

    print('All tests done!')