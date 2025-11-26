# -*- encoding: utf-8 -*-
'''

Test script

@File    :   test.py
@Time    :   2025/09/02 15:14:44
@Author  :   Yangshuo He
@Contact :   sugarhe58@gmail.com
'''

import json
import torch
import argparse
import numpy as np
from pbb.utils import test_exp, train_and_certificate, my_exp, set_device, set_seed
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

    device = set_device(args)
    set_seed(args.seed, device)

    loader_kargs = {'num_workers': args.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': args.num_workers}

    # mnist
    # args.name_data = 'mnist'
    # args.l_0 = 2
    # args.model = 'cnn'
    # args.layers = 4
    # args.perc_prior = 0.3

    train, test = loaddataset(args.name_data)

    all_train_loader, test_loader, prior_loader, _, _, train_loader, lip_all_loader, lip_test_loader = loadbatches(train, test, loader_kargs, args.batch_size, args.lip_bs, prior=(args.init_prior == 'learnt'), perc_train=args.perc_train, perc_prior=args.perc_prior)

    args.prior_epochs = 5
    args.epochs = 30

    train_and_certificate(args, train_loader=all_train_loader, prior_loader=prior_loader, test_loader=test_loader, empirical_loader=all_train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)
    
    args.prior_epochs = 10
    args.epochs = 30

    train_and_certificate(args, train_loader=all_train_loader, prior_loader=prior_loader, test_loader=test_loader, empirical_loader=all_train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.prior_epochs = 20
    args.epochs = 30

    train_and_certificate(args, train_loader=all_train_loader, prior_loader=prior_loader, test_loader=test_loader, empirical_loader=all_train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.prior_epochs = 30
    args.epochs = 30

    train_and_certificate(args, train_loader=all_train_loader, prior_loader=prior_loader, test_loader=test_loader, empirical_loader=all_train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.prior_epochs = 5
    args.epochs = 50

    train_and_certificate(args, train_loader=all_train_loader, prior_loader=prior_loader, test_loader=test_loader, empirical_loader=all_train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)
    
    args.prior_epochs = 10
    args.epochs = 50

    train_and_certificate(args, train_loader=all_train_loader, prior_loader=prior_loader, test_loader=test_loader, empirical_loader=all_train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.prior_epochs = 20
    args.epochs = 50

    train_and_certificate(args, train_loader=all_train_loader, prior_loader=prior_loader, test_loader=test_loader, empirical_loader=all_train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.prior_epochs = 30
    args.epochs = 50

    train_and_certificate(args, train_loader=all_train_loader, prior_loader=prior_loader, test_loader=test_loader, empirical_loader=all_train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)


    print('All tests done!')