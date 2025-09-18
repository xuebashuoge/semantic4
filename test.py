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
from argparse import Namespace
from pbb.utils import test_exp, train_and_certificate, my_exp
from pbb.data import loaddataset, loadbatches

if __name__ == '__main__':
    # This is the key: a robust, version-agnostic way to select the device.
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("CUDA is available. Using GPU.")
    # Check if MPS is available (for macOS with Apple Silicon)
    # The 'hasattr' check is crucial for compatibility with older PyTorch versions
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device == 'mps':
        torch.use_deterministic_algorithms(True)

    args_dict = {
        'name': 'myexp',
        'name_data': 'mnist',
        'model': 'cnn',
        'layers': 9,
        'prior_dist': 'gaussian',
        'sigma_prior': 0.03,
        'l_0': 2,
        'channel_type': 'bec',
        'outage': 0.1,
        'noise_var': 1,
        'batch_size': 250,
        'lip_bs': 100,
        'perc_train': 1.0,
        'perc_prior': 0.5,
        'prior_epochs': 20,
        'learning_rate_prior': 0.01,
        'momentum_prior': 0.95,
        'epochs': 100,
        'learning_rate': 0.001,
        'momentum': 0.95,
        'dropout_prob': 0.2,
        'mc_samples': 200,
        'clamping': True,
        'pmin': 1e-5,
        'num_workers': 8,
        'chunk_size': 16,    # for efficient Lipschitz constant computation
    }

    args = Namespace(**args_dict)

    loader_kargs = {'num_workers': args.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': args.num_workers}

    train, test = loaddataset(args.name_data)

    train_loader, test_loader, valid_loader, _, _, bound_loader, lip_all_loader, lip_test_loader = loadbatches(train, test, loader_kargs, args.batch_size, args.lip_bs, prior=True, perc_train=args.perc_train, perc_prior=args.perc_prior)

    args.name = 'prior0.5-train1.0-empirical1.0'
    args.l_0 = 2
    args.outage = 0.1

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.name = 'prior0.5-train1.0-empirical1.0'
    args.l_0 = 2
    args.outage = 0.2

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.name = 'prior0.5-train1.0-empirical1.0'
    args.model = 'fcn'
    args.l_0 = 2
    args.outage = 0.1

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.name = 'prior0.5-train1.0-empirical1.0'
    args.model = 'fcn'
    args.l_0 = 2
    args.outage = 0.2
    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    # cifar10
    args.name_data = 'cifar10'
    args.model = 'cnn'
    args.layers = 9

    train, test = loaddataset(args.name_data)

    train_loader, test_loader, valid_loader, _, _, bound_loader, lip_all_loader, lip_test_loader = loadbatches(train, test, loader_kargs, args.batch_size, args.lip_bs, prior=True, perc_train=args.perc_train, perc_prior=args.perc_prior)

    args.name = 'prior0.5-train1.0-empirical1.0'
    args.l_0 = 2
    args.outage = 0.1

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.name = 'prior0.5-train1.0-empirical1.0'
    args.l_0 = 2
    args.outage = 0.2

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.name = 'prior0.5-train1.0-empirical1.0'
    args.model = 'fcn'
    args.l_0 = 2
    args.outage = 0.1

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)

    args.name = 'prior0.5-train1.0-empirical1.0'
    args.model = 'fcn'
    args.l_0 = 2
    args.outage = 0.2

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, lip_loader=lip_all_loader, device=device)


    print('All tests done!')