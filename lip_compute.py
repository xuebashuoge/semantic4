# -*- encoding: utf-8 -*-
'''

Lipschitz constant computation functions

@File    :   lip_compute.py
@Time    :   2025/11/17 11:38:05
@Author  :   Yangshuo He
@Contact :   sugarhe58@gmail.com
'''

import json
import torch
import argparse
import numpy as np
from pbb.utils import compute_lipschitz_constant_new, set_device, set_seed
from pbb.data import loaddataset, loadbatches

if __name__ == '__main__':
    
    # --- Load Config ---
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        args_dict = json.load(f)

    # input parser
    parser = argparse.ArgumentParser(description='Test script for semantic4')

    # add args to the parser
    for key, value in args_dict.items():
        # For boolean arguments, use a different action
        if isinstance(value, bool):
            # This creates --key and --no-key arguments
            parser.add_argument(f'--{key}', type=bool, default=value, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value, help=f'Set the {key}')

    parser.add_argument('--gpu', type=int, default=2, help='GPU id to use (if available)')

    args = parser.parse_args()

    # set device and seed
    device = set_device()
    set_seed(args.seed, device)

    # load data
    loader_kargs = {'num_workers': args.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': args.num_workers}

    train, test = loaddataset(args.name_data)

    train_loader, test_loader, valid_loader, _, _, bound_loader, lip_all_loader, lip_test_loader = loadbatches(train, test, loader_kargs, args.batch_size, args.lip_bs, prior=True, perc_train=args.perc_train, perc_prior=args.perc_prior)

    # compute Lipschitz constant
    lip_constant = compute_lipschitz_constant_new(args, loader=lip_all_loader, mc_samples=args.mc_samples, pmin=args.pmin, clamping=args.clamping, chunk_size=args.chunk_size, device=device)

    print(f'Computed Lipschitz constant: {lip_constant}')

    # save Lipschitz constant
    with open('lip_constant.json', 'w') as f:
        json.dump({'lip_constant': lip_constant}, f)