# -*- encoding: utf-8 -*-
'''

Lipschitz constant computation functions

@File    :   lip_compute.py
@Time    :   2025/11/17 11:38:05
@Author  :   Yangshuo He
@Contact :   sugarhe58@gmail.com
'''

import os
import json
import torch
import argparse
import numpy as np
from pbb.utils import compute_lipschitz_constant_new, set_device, set_seed
from pbb.data import loaddataset, loadbatches
from pbb.models import select_prior_network

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
    device = set_device(args)
    set_seed(args.seed, device)

    # load data
    loader_kargs = {'num_workers': args.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': args.num_workers}

    train, test = loaddataset(args.name_data)

    train_loader, test_loader, valid_loader, _, _, bound_loader, lip_all_loader, lip_test_loader = loadbatches(train, test, loader_kargs, args.batch_size, args.lip_bs, prior=True, perc_train=args.perc_train, perc_prior=args.perc_prior)


    os.makedirs('results', exist_ok=True)
    if args.channel_type.lower() == 'rayleigh':
        channel_specs = f'noise{args.noise_var}'
    elif args.channel_type.lower() == 'bec':
        channel_specs = f'outage{args.outage}'
    else:
        channel_specs = 'nochannel'

    # load prior if it is not random
    if args.init_prior.lower() != 'random':
        net0 = select_prior_network(args.model, args.layers, args.name_data, args.dropout_prob, device=device)
        
        prior_file = f'results/prior/{args.name}_{args.name_data}_{args.model}-{args.layers}_sig{args.sigma_prior}_pmin{args.pmin}_{args.prior_dist}_epochpri{args.prior_epochs}_bs{args.batch_size}_lrpri{args.learning_rate_prior}_mompri{args.momentum_prior}_drop{args.dropout_prob}_perc{args.perc_prior}/prior_net.pth'

        try:
            net0.load_state_dict(torch.load(prior_file, weights_only=False, map_location=device))
        except Exception as e:
            raise RuntimeError(f'Error loading prior network from {prior_file}: {e}')
    else:
        net0 = None

    # compute Lipschitz constant
    lip_constant, lip_list = compute_lipschitz_constant_new(args, loader=lip_test_loader, mc_samples=args.mc_samples, pmin=args.pmin, clamping=args.clamping, chunk_size=args.chunk_size, init_net=net0, device=device)

    print(f'Computed Lipschitz constant, {args.name_data}, {args.model}-{args.layers}, {args.init_prior}-prior_{args.channel_type.lower()}, {channel_specs}_chan-layer{args.l_0}, {'bounded' if args.clamping else 'unbounded'}-loss: {lip_constant}')

    print(f'Lipschitz constants from each MC sample: {lip_list}')

    # save Lipschitz constant
    with open(f'results/lip_constant_{args.lip_name}_{args.name_data}_{args.model}-{args.layers}_{args.init_prior}-prior_{args.channel_type.lower()}_{channel_specs}_chan-layer{args.l_0}_{'bounded' if args.clamping else 'unbounded'}-loss.json', 'w') as f:
        json.dump({'lip_constant': lip_constant}, f)