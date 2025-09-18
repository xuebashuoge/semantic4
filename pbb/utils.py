import math
import os
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from pbb.models import ProbLinear, ProbConv2d, WirelessChannel, NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbNNet4lChannel, ProbCNNet4lChannel, ProbCNNet9l, ProbCNNet9lChannel, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, ProbCNNet13lChannel, ProbCNNet15lChannel, trainNNet, testNNet, Lambda_var, trainPNNet,trainPNNet2, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, compute_empirical_risk
from pbb.bounds import PBBobj
from pbb import data
from pbb.data import loaddataset, loadbatches



def my_exp(args, device='cuda', num_workers=8):
    loader_kargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': num_workers}

    train, test = loaddataset(args.name_data)

    train_loader, test_loader, valid_loader, _, _, _ = loadbatches(train, test, loader_kargs, args.batch_size, prior=True, perc_train=args.perc_train, perc_prior=args.perc_prior)

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, device=device)



def train_and_certificate(args, train_loader, prior_loader, test_loader, empirical_loader, population_loader, lip_loader, device='cuda'):

    rho_prior = math.log(math.exp(args.sigma_prior)-1.0)

    # learn prior
    print('Learning prior...')

    if args.model.lower() == 'cnn':
        if args.name_data.lower() == 'cifar10':
            # fcn for mnist, cnn for cifar10
            if args.layers == 9:
                net0 = CNNet9l(dropout_prob=args.dropout_prob).to(device)
            elif args.layers == 13:
                net0 = CNNet13l(dropout_prob=args.dropout_prob).to(device)
            elif args.layers == 15:
                net0 = CNNet15l(dropout_prob=args.dropout_prob).to(device)
            else:
                raise RuntimeError(f'Wrong number of layers chosen {args.layers}')
        else:
            args.layers = 4
            net0 = CNNet4l(dropout_prob=args.dropout_prob).to(device)
    elif args.model.lower() == 'fcn':
        args.layers = 4
        net0 = NNet4l(dropout_prob=args.dropout_prob).to(device)
    else:
        raise RuntimeError(f'Wrong model chosen {args.model}-{args.layers}')
    
    prior_folder = f'results/prior/{args.name}_{args.name_data}_{args.model}-{args.layers}_sig{args.sigma_prior}_pmin{args.pmin}_{args.prior_dist}_epochpri{args.prior_epochs}_bs{args.batch_size}_lrpri{args.learning_rate_prior}_mompri{args.momentum_prior}_drop{args.dropout_prob}_perc{args.perc_prior}/'

    train_prior(net0, prior_loader, test_loader, prior_folder, args, device)

    # load probabilistic model
    print('Training posterior...')

    if args.model.lower() == 'cnn':
        if args.name_data.lower() == 'cifar10':
            if args.layers == 9:
                net = ProbCNNet9lChannel(rho_prior, prior_dist=args.prior_dist, l_0=args.l_0, channel_type=args.channel_type, outage=args.outage, device=device, init_net=net0).to(device)
            elif args.layers == 13:
                net = ProbCNNet13lChannel(rho_prior, prior_dist=args.prior_dist, l_0=args.l_0, channel_type=args.channel_type, outage=args.outage, device=device, init_net=net0).to(device)
            elif args.layers == 15:
                net = ProbCNNet15lChannel(rho_prior, prior_dist=args.prior_dist, l_0=args.l_0, channel_type=args.channel_type, outage=args.outage, device=device, init_net=net0).to(device)
            else:
                raise RuntimeError(f'Wrong number of layers chosen {args.layers}')
        else:
            net = ProbCNNet4lChannel(rho_prior, prior_dist=args.prior_dist, l_0=args.l_0, channel_type=args.channel_type, outage=args.outage, device=device, init_net=net0).to(device)
    elif args.model.lower() == 'fcn':
        net = ProbNNet4lChannel(rho_prior, prior_dist=args.prior_dist, l_0=args.l_0, channel_type=args.channel_type, outage=args.outage, device=device, init_net=net0).to(device)
    else:
        raise RuntimeError(f'Wrong model chosen {args.model}-{args.layers}')

    posterior_folder = f'results/{args.name}_{args.name_data}_{args.model}-{args.layers}_sig{args.sigma_prior}_pmin{args.pmin}_{args.prior_dist}_epoch{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}_mom{args.momentum}_drop{args.dropout_prob}/'
    kl = train_posterior(net, train_loader, posterior_folder, args, device)

    print('Computing certificate...')
    certificate_folder = f'results/{args.name}_{args.name_data}_{args.model}-{args.layers}_sig{args.sigma_prior}_pmin{args.pmin}_{args.prior_dist}_epoch{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}_mom{args.momentum}_drop{args.dropout_prob}/certificate/'

    # compute empirical and population risks
    compute_certificate(net, empirical_loader, population_loader, lip_loader, certificate_folder, kl, args, device)

    print('Done!')


def test_exp(learning_rate, momentum, epochs, model='cnn', name_data='cifar10', sigma_prior=0.03, learning_rate_prior=0.01, momentum_prior=0.95, prior_epochs=70, dropout_prob=0.2, batch_size=250, perc_train=1.0, perc_prior=0.5, prior_dist='gaussian', l_0=2, channel_type='bec', outage=0.1, noise_var = 1, mc_samples=100, clamping=True, pmin=1e-5, device='cuda', num_workers=8):


    loader_kargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': num_workers}

    train, test = loaddataset(name_data)
    rho_prior = math.log(math.exp(sigma_prior)-1.0)

    
    train_loader, test_loader, valid_loader, _, _, _ = loadbatches(train, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)

    folder = f'results/prior/{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_{prior_dist}_epoch{prior_epochs}_bs{batch_size}_lr{learning_rate_prior}_mom{momentum_prior}_drop{dropout_prob}_percpri{perc_prior}/'

    net0 = CNNet9l(dropout_prob=dropout_prob).to(device)

    train_prior(net0, valid_loader, test_loader, folder, model, name_data, sigma_prior, learning_rate_prior, momentum_prior, prior_epochs, dropout_prob, perc_train, perc_prior, device)

    # load probabilistic model
    net = ProbCNNet9lChannel(rho_prior, prior_dist=prior_dist, l_0=l_0, channel_type=channel_type, outage=outage, device=device, init_net=net0).to(device)

    folder = f'results/{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_{prior_dist}_epoch{epochs}_bs{batch_size}_lr{learning_rate}_mom{momentum}_drop{dropout_prob}/'
    kl = train_posterior(net, train_loader, learning_rate, momentum, epochs, folder, model, name_data, sigma_prior, dropout_prob, clamping, pmin, device)

    folder = f'results/{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_{prior_dist}_epoch{epochs}_bs{batch_size}_lr{learning_rate}_mom{momentum}_drop{dropout_prob}/certificate/'

    # compute empirical and population risks
    compute_certificate(net, train_loader, test_loader, folder, learning_rate, momentum, epochs, kl, model, name_data, sigma_prior, learning_rate_prior, momentum_prior, prior_epochs, dropout_prob, batch_size, perc_train, perc_prior, prior_dist, l_0, channel_type, outage, noise_var, mc_samples, clamping, pmin, device)

    print('Done!')



def train_prior(net0, train_loader, test_loader, folder, args, device='cuda'):
    
    os.makedirs(folder, exist_ok=True)
    prior_file = f'{folder}/prior_net.pth'

    if os.path.exists(prior_file):
        net0.load_state_dict(torch.load(prior_file, weights_only=False, map_location=device))
        try:
            prior_results_dict = torch.load(f'{folder}/prior_results.pth', weights_only=False, map_location=device)
        except TypeError:
            prior_results_dict = torch.load(f'{folder}/prior_results.pth', map_location=device)
        prior_loss_tr = prior_results_dict['prior_loss_tr']
        prior_err_tr = prior_results_dict['prior_err_tr']
        loss_net0 = prior_results_dict['loss_net0']
        error_net0 = prior_results_dict['error_net0']
        print(f"Loaded prior from {prior_file}")
        print(f"train loss last epoch {prior_loss_tr[-1]:.4f}, train err last epoch {prior_err_tr[-1]:.4f}, test loss {loss_net0:.4f}, test err {error_net0:.4f}")
    
    else:
        optimizer = optim.SGD(net0.parameters(), lr=args.learning_rate_prior, momentum=args.momentum_prior)

        prior_loss_tr = torch.zeros(args.prior_epochs)
        prior_err_tr = torch.zeros(args.prior_epochs)
        for epoch in range(args.prior_epochs):
            train_loss, train_err = trainNNet(net0, optimizer, epoch, train_loader, device=device, verbose=True)

            prior_loss_tr[epoch] = train_loss
            prior_err_tr[epoch] = train_err
        
        loss_net0, error_net0 = testNNet(net0, test_loader, device=device)

        plt.figure()
        plt.plot(range(1,args.prior_epochs+1), prior_loss_tr)
        plt.xlabel('Epochs')
        plt.ylabel('Prior NLL loss')
        plt.savefig(f'{folder}/prior_loss.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.plot(range(1,args.prior_epochs+1), prior_err_tr)
        plt.xlabel('Epochs')
        plt.ylabel('Prior 0-1 error')
        plt.savefig(f'{folder}/prior_err.pdf', dpi=300, bbox_inches='tight')

        torch.save(net0.state_dict(), prior_file)

        prior_results_dict = {
            'args': args,
            'loss_net0': loss_net0,
            'error_net0': error_net0,
            'prior_loss_tr': prior_loss_tr,
            'prior_err_tr': prior_err_tr,
        }

    torch.save(prior_results_dict, f'{folder}/prior_results.pth')

def train_posterior(net, train_loader, folder, args, device='cuda'):
    
    os.makedirs(folder, exist_ok=True)
    posterior_file = f'{folder}/posterior_net.pth'

    if os.path.exists(posterior_file):
        net.load_state_dict(torch.load(posterior_file, weights_only=False, map_location=device))
        try:
            posterior_results_dict = torch.load(f'{folder}/posterior_results_dict.pth', weights_only=False, map_location=device)
        except TypeError:
            posterior_results_dict = torch.load(f'{folder}/posterior_results_dict.pth', map_location=device)
        loss_tr = posterior_results_dict['loss_tr']
        err_tr = posterior_results_dict['err_tr']
        kl = posterior_results_dict['kl']

        print(f"Loaded posterior from {posterior_file}")
        print(f"train loss last epoch {loss_tr[-1]:.4f}, train err last epoch {err_tr[-1]:.4f}, kl {kl:.4f}")
    else:

        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

        loss_tr = torch.zeros(args.epochs)
        err_tr = torch.zeros(args.epochs)
        kl_tr = torch.zeros(args.epochs)

        for epoch in range(args.epochs):
            train_loss, train_err, train_kl = trainPNNet2(net, optimizer, epoch, train_loader, device=device, clamping=args.clamping, pmin=args.pmin, verbose=True)
            loss_tr[epoch] = train_loss
            err_tr[epoch] = train_err
            kl_tr[epoch] = train_kl
            
        kl = net.compute_kl()

        plt.figure()
        plt.plot(range(1,args.epochs+1), loss_tr)
        plt.xlabel('Epochs')
        plt.ylabel('NLL loss')
        plt.savefig(f'{folder}/loss.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.plot(range(1,args.epochs+1), err_tr)
        plt.xlabel('Epochs')
        plt.ylabel('0-1 error')
        plt.savefig(f'{folder}/err.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.plot(range(1,args.epochs+1), kl_tr)
        plt.xlabel('Epochs')
        plt.ylabel('KL divergence')
        plt.savefig(f'{folder}/kl.pdf', dpi=300, bbox_inches='tight')

        
        torch.save(net.state_dict(), f'{folder}/posterior_net.pth')

        posterior_results_dict = {
            'args': args,
            'loss_tr': loss_tr,
            'err_tr': err_tr,
            'kl_tr': kl_tr,
            'kl': kl
        }

        torch.save(posterior_results_dict, f'{folder}/posterior_results_dict.pth')
    
    return kl


def compute_certificate(net, empirical_loader, population_loader, lip_loader, folder, kl, args, device='cuda'):

    os.makedirs(folder, exist_ok=True)

    net.eval()

    # compute Lipschitz constant L_w
    L_w = compute_lipschitz_constant_direct(net, lip_loader, args.mc_samples, args.pmin, args.clamping, args.chunk_size, device)

    # compute empirical risk using mc samples
    error_empirical, cross_entropy_empirical = compute_empirical(net, empirical_loader, args, device)

    # compute population risk
    error_population, cross_entropy_population = compute_population(net, population_loader, args, device)


    # # compute empirical risk using mc samples
    # correct_empirical = 0.0
    # cross_entropy_empirical = 0.0

    # for data_batch, target_batch in tqdm(empirical_loader):
    #     data_batch, target_batch = data_batch.to(device), target_batch.to(device)

    #     for _ in range(args.mc_samples):

    #         outputs = net(data_batch, sample=True, wireless=False, clamping=args.clamping, pmin=args.pmin)
    #         loss_ce = compute_empirical_risk(outputs, target_batch, args.pmin, args.clamping)
    #         pred = outputs.max(1, keepdim=True)[1]
    #         correct_empirical += pred.eq(target_batch.view_as(pred)).sum().item()
    #         cross_entropy_empirical += loss_ce.item()

    # cross_entropy_empirical /= (len(empirical_loader) * args.mc_samples)
    # error_empirical = 1.0 - (correct_empirical / (len(empirical_loader) * empirical_loader.batch_size * args.mc_samples))

    # # compute population risk
    # correct_population = 0.0
    # cross_entropy_population = 0.0

    # with torch.no_grad():
    #     for data, target in tqdm(population_loader):
    #         data, target = data.to(device), target.to(device)

    #         for _ in range(args.mc_samples):

    #             outputs = net(data, sample=True, wireless=True, clamping=args.clamping, pmin=args.pmin)

    #             loss_ce = compute_empirical_risk(outputs, target, args.pmin, args.clamping)
    #             pred = outputs.max(1, keepdim=True)[1]
    #             correct_population += pred.eq(target.view_as(pred)).sum().item()
    #             cross_entropy_population += loss_ce.item()

    # cross_entropy_population /= (len(population_loader) * args.mc_samples)
    # error_population = 1.0 - (correct_population / (len(population_loader) * population_loader.batch_size * args.mc_samples))

    

    # bound evaluation
    k = math.sqrt(len(empirical_loader.dataset))
    # sigma-sub-Gaussian is equivalent to bounded in [0, 2*sigma], the loss function here is clamped in [0, log(1/pmin)]
    sigma = math.log(1/args.pmin)/2
    bound_ce = cross_entropy_empirical + k*sigma**2 / (2*len(empirical_loader.dataset)) + 1/k * (kl + math.log(1))


    print(f"***Final results***")
    print(f"Dataset: {args.name_data}, Sigma: {args.sigma_prior :.5f}, pmin: {args.pmin :.5f}, Dropout: {args.dropout_prob :.5f}, Perc_train: {args.perc_train :.5f}, Perc_prior: {args.perc_prior :.5f}, L_0: {args.l_0}, Channel: {args.channel_type}, Outage: {args.outage :.5f}, MC samples: {args.mc_samples}, Clamping: {args.clamping}, Empirical error: {error_empirical :.5f}, Empirical CE loss: {cross_entropy_empirical :.5f}, Population error: {error_population :.5f}, Population CE loss: {cross_entropy_population :.5f}, KL: {kl :.5f}, L_w: {L_w :.5f}")

    results_dict = {
        'args': args,
        'error_empirical': error_empirical,
        'cross_entropy_empirical': cross_entropy_empirical,
        'error_population': error_population,
        'cross_entropy_population': cross_entropy_population,
        'kl': kl,
        'L_w': L_w,
    }

    if args.channel_type.lower() == 'rayleigh':
        channel_specs = f'noise{args.noise_var}'
    elif args.channel_type.lower() == 'bec':
        channel_specs = f'outage{args.outage}'
    else:
        channel_specs = 'nochannel'

    torch.save(results_dict, f'{folder}/{args.channel_type.lower()}_{channel_specs}_chan-layer{args.l_0}_mcsamples{args.mc_samples}_results.pth')

def compute_lipschitz_constant(net, loaders, mc_samples, pmin, clamping, device):
    """
    Computes the Lipschitz constant of the loss wrt the weights.

    Approximates L_w = sup_{w,z} ||∇_w l(w,z)||_2 by finding the maximum
    gradient norm over the provided dataset and for multiple Monte Carlo
    samples of the model weights.

    Parameters
    ----------
    net : nn.Module
        The trained probabilistic neural network.
    loaders : list of DataLoader
        The data loaders for the dataset (e.g., test or validation set).
    mc_samples : int
        The number of Monte Carlo samples to draw for the weights per data point.
    pmin : float
        The minimum probability value for clamping in the loss function.
    clamping : bool
        Whether to apply clamping in the loss function.
    device : str
        The device to run the computation on ('cuda' or 'cpu').

    Returns
    -------
    float
        The computed Lipschitz constant (maximum gradient norm).
    """
    net.eval()  # Set the model to evaluation mode
    max_grad_norm = 0.0

    # Chain the loaders together to create a single iterator
    combined_iterator = itertools.chain(*loaders)

    print("Computing Lipschitz constant (L_w)...")
    # Iterate over batches in the dataset
    for data_batch, target_batch in tqdm(combined_iterator, desc="Lipschitz computation"):
        data_batch, target_batch = data_batch.to(device), target_batch.to(device)
        batch_size = data_batch.size(0)

        # Loop for Monte Carlo samples of the weights
        for _ in range(mc_samples):
            # 1. Perform one forward pass for the entire batch.
            # `sample=True` draws a new set of weights 'w' from the posterior.
            outputs = net(data_batch, sample=True, clamping=clamping, pmin=pmin)

            # 2. Compute loss for each sample in the batch.
            per_sample_losses = compute_empirical_risk(outputs, target_batch, pmin, clamping, per_sample=True)

            # 3. Compute per-sample gradients and find the max norm in the batch.
            for i in range(batch_size):
                net.zero_grad() # Clear gradients for the next sample's backward pass.

                # We need to retain the graph for all but the last sample in the batch
                # because they all depend on the same 'outputs' tensor from the forward pass.
                is_last_sample = (i == batch_size - 1)
                per_sample_losses[i].backward(retain_graph=not is_last_sample)

                # 4. Collect all parameter gradients into a single flat vector.
                all_grads = [param.grad.view(-1) for param in net.parameters() if param.grad is not None]

                if not all_grads:
                    continue

                flat_grads = torch.cat(all_grads)
                
                # 5. Compute the L2 norm of the gradient vector.
                current_norm = torch.linalg.norm(flat_grads).item()

                # 6. Update the overall maximum norm found so far.
                if current_norm > max_grad_norm:
                    max_grad_norm = current_norm
    
    return max_grad_norm

def compute_lipschitz_constant_efficient(net, loader, mc_samples, pmin, clamping, chunk_size, device):
    """
    Computes the Lipschitz constant efficiently using torch.func.vmap.

    Parameters
    ----------
    net : nn.Module
        The trained probabilistic neural network.
    loader : DataLoader
        The data loaders to iterate over.
    mc_samples : int
        The number of Monte Carlo samples for the weights.
    pmin : float
        The minimum probability value for clamping.
    clamping : bool
        Whether to apply clamping in the loss function.
    device : str
        The device to run the computation on ('cuda' or 'cpu').

    Returns
    -------
    float
        The computed Lipschitz constant (maximum gradient norm).
    """
    net.eval()
    max_grad_norm = 0.0

    # This functional forward pass computes the loss for a single data point
    # given a fixed set of sampled weights.
    def compute_loss_functional(sampled_weights, buffers, x_sample, y_sample):
        # The model's forward needs a batch dimension, so we add it with unsqueeze
        outputs = torch.func.functional_call(
            net, 
            (sampled_weights, buffers), 
            args=(x_sample.unsqueeze(0),), 
            kwargs={'sample': False, 'wireless': True} # We've already sampled the weights!
        )
        # We use the per-sample loss function and take the single resulting value
        loss = compute_empirical_risk(outputs, y_sample.unsqueeze(0), pmin, clamping, per_sample=True)
        return loss.squeeze()

    # Create a function that computes gradients with respect to the weights
    grad_fn = torch.func.grad(compute_loss_functional, argnums=0)
    
    # Use vmap to vectorize the gradient calculation over the batch dimension
    # in_dims specifies which arguments to map over:
    # None for weights/buffers (they are fixed for the batch),
    # 0 for data/targets (iterate along the first dimension).
    vmapped_grad_fn = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0), chunk_size=chunk_size)

    total_batches = len(loader)
    
    print("Computing Lipschitz constant efficiently with torch.func...")
    
    with tqdm(total=total_batches * mc_samples, desc="Processing") as pbar:
        for _ in range(mc_samples):
            # 1. Sample a single set of weights 'w' for the entire network
            # This is done by replacing the probabilistic layers' parameters
            # with a single sample from their respective distributions.
            sampled_weights = {}
            for name, module in net.named_modules():
                if isinstance(module, (ProbLinear, ProbConv2d)):
                    # Get the parameter names for this specific module
                    weight_name = f"{name}.weight.mu"
                    bias_name = f"{name}.bias.mu"
                    # Sample weights and biases and store them
                    sampled_weights[weight_name] = module.weight.sample()
                    sampled_weights[bias_name] = module.bias.sample()
            
            # Get the model's buffers (e.g., for batch norm, though not present here)
            buffers = {name: buf for name, buf in net.named_buffers()}

            for data_batch, target_batch in loader:
                data_batch, target_batch = data_batch.to(device), target_batch.to(device)
                
                # 2. Compute all per-sample gradients in one vectorized call
                per_sample_grads_dict = vmapped_grad_fn(sampled_weights, buffers, data_batch, target_batch)

                # 3. Calculate the norm for each sample's gradient and find the max
                # Flatten the gradients for each sample across all parameters
                flat_grads_per_sample = torch.cat([g.flatten(start_dim=1) for g in per_sample_grads_dict.values()], dim=1) # Shape: (batch_size, num_total_params)
                
                # Compute L2 norm for each row (each sample)
                norms = torch.linalg.norm(flat_grads_per_sample, dim=1)
                
                batch_max_norm = torch.max(norms).item()
                if batch_max_norm > max_grad_norm:
                    max_grad_norm = batch_max_norm

                # --- FIX ADDED HERE ---
                # 1. Explicitly delete the large tensors to free up references.
                del per_sample_grads_dict, flat_grads_per_sample, norms

                # 2. Tell the backend to empty its cache of unused memory.
                if device == 'cuda':
                    torch.cuda.empty_cache()
                elif device == 'mps':
                    torch.mps.empty_cache()
                # ----------------------
                
                pbar.update(1)
            
    
    return max_grad_norm

import time

def compute_lipschitz_constant_direct(net, loader, mc_samples, pmin, clamping, chunk_size, device):
    """
    Computes the Lipschitz constant using the direct ratio method, vectorized with torch.func.vmap.
    K = sup |l(w',z) - l(w,z)| / ||w'^{(l0)} - I||

    Parameters
    ----------
    net : nn.Module
        The trained probabilistic neural network. It must contain the channel layer.
    loader : DataLoader
        The data loader to iterate over.
    mc_samples : int
        The number of Monte Carlo samples for the weights.
    pmin : float
        The minimum probability value for clamping in the loss function.
    clamping : bool
        Whether to apply clamping in the loss function.
    chunk_size : int
        Chunk size for vmap to manage memory usage.
    device : str
        The device to run the computation on ('cuda', 'mps', or 'cpu').

    Returns
    -------
    float
        The estimated Lipschitz constant (maximum ratio).
    """
    net.eval()
    max_k = 0.0

    # Define a function that performs the core calculation for a SINGLE data sample.
    # vmap will then vectorize this across the whole batch.
    def compute_k_for_sample(sampled_weights, buffers, x_sample, y_sample):
        # --- NO CHANNEL (w) ---
        # Forward pass for the ideal network (wireless=False)
        outputs_no_channel = torch.func.functional_call(
            net,
            (sampled_weights, buffers),
            args=(x_sample.unsqueeze(0),),
            kwargs={'sample': False, 'wireless': False, 'return_channel_weight': False}
        )
        loss_no_channel = compute_empirical_risk(outputs_no_channel, y_sample.unsqueeze(0), pmin, clamping, per_sample=True)

        # --- WITH CHANNEL (w') ---
        # Forward pass for the network with the channel (wireless=True)
        outputs_with_channel, channel_params = torch.func.functional_call(
            net,
            (sampled_weights, buffers),
            args=(x_sample.unsqueeze(0),),
            kwargs={'sample': False, 'wireless': True, 'return_channel_weight': True}
        )
        loss_with_channel = compute_empirical_risk(outputs_with_channel, y_sample.unsqueeze(0), pmin, clamping, per_sample=True)
        
        # --- DENOMINATOR ||w' - w||_2 ---
        channel_weight, channel_bias = channel_params
        
        # The 'ideal' weight is 1.0 and 'ideal' bias is 0.0
        # For BEC, channel_bias will be None.
        if channel_bias is not None:
            # Rayleigh channel case
            # Note: .contiguous() can sometimes help vmap performance
            flat_w_diff = (channel_weight - 1.0).contiguous().view(-1)
            flat_b_diff = channel_bias.contiguous().view(-1)
            d_w = torch.linalg.norm(torch.cat((flat_w_diff, flat_b_diff)))
        else:
            # BEC channel case
            d_w = torch.linalg.norm(channel_weight - 1.0)

        # --- RATIO ---
        # The condition for the whole batch
        condition = d_w > 1e-9

        # The value if the condition is True
        # We use torch.ones_like(d_w) to avoid division by zero for the False cases, 
        # but their results will be discarded anyway.
        safe_d_w = torch.where(condition, d_w, torch.ones_like(d_w))
        value_if_true = torch.abs(loss_with_channel - loss_no_channel) / safe_d_w

        # The value if the condition is False
        value_if_false = torch.zeros_like(d_w)

        # Select between the two based on the condition
        k_sample = torch.where(condition, value_if_true, value_if_false)
            
        return k_sample.squeeze()

    # Vectorize our single-sample function to run on a full batch.
    vmapped_k_fn = torch.func.vmap(compute_k_for_sample, in_dims=(None, None, 0, 0), chunk_size=chunk_size, randomness="different")

    # Before the mc_samples loop, get the parameter names once
    param_names = []
    for name, module in net.named_modules():
        if isinstance(module, (ProbLinear, ProbConv2d)):
            param_names.append(f"{name}.weight.mu")
            param_names.append(f"{name}.bias.mu")

    print("Computing Lipschitz constant with the direct method using torch.func...")
    with tqdm(total=len(loader) * mc_samples, desc="Processing") as pbar:
        for _ in range(mc_samples):

            # 1. Sample one set of Bayesian weights for this MC iteration.
            sampled_weights = dict.fromkeys(param_names) # Pre-allocate
            for name, module in net.named_modules():
                if isinstance(module, (ProbLinear, ProbConv2d)):
                    sampled_weights[f"{name}.weight.mu"] = module.weight.sample()
                    sampled_weights[f"{name}.bias.mu"] = module.bias.sample()
            
            buffers = {name: buf for name, buf in net.named_buffers()}


            for data_batch, target_batch in loader:
                data_batch, target_batch = data_batch.to(device), target_batch.to(device)
                
                # 2. Compute all per-sample k values in one vectorized call
                k_values_batch = vmapped_k_fn(sampled_weights, buffers, data_batch, target_batch)
                
                # 3. Find the max k in the current batch and update the global max
                batch_max_k = torch.max(k_values_batch).item()
                if batch_max_k > max_k:
                    max_k = batch_max_k

                pbar.update(1)
            
            # Clean up memory after each MC sample
            del sampled_weights, buffers
            if device == 'cuda': torch.cuda.empty_cache()
            elif device == 'mps': torch.mps.empty_cache()
            
    
    return max_k


def compute_empirical(net, empirical_loader, args, device='cuda'):
    net.eval()
    # compute empirical risk using mc samples
    correct_empirical = 0.0
    cross_entropy_empirical = 0.0

    for data_batch, target_batch in tqdm(empirical_loader):
        data_batch, target_batch = data_batch.to(device), target_batch.to(device)

        for _ in range(args.mc_samples):

            outputs = net(data_batch, sample=True, wireless=False, clamping=args.clamping, pmin=args.pmin)
            loss_ce = compute_empirical_risk(outputs, target_batch, args.pmin, args.clamping)
            pred = outputs.max(1, keepdim=True)[1]
            correct_empirical += pred.eq(target_batch.view_as(pred)).sum().item()
            cross_entropy_empirical += loss_ce.item()

    cross_entropy_empirical /= (len(empirical_loader) * args.mc_samples)
    error_empirical = 1.0 - (correct_empirical / (len(empirical_loader) * empirical_loader.batch_size * args.mc_samples))

    return error_empirical, cross_entropy_empirical

def compute_population(net, population_loader, args, device='cuda'):
    net.eval()

    # compute population risk
    correct_population = 0.0
    cross_entropy_population = 0.0

    with torch.no_grad():
        for data, target in tqdm(population_loader):
            data, target = data.to(device), target.to(device)

            for _ in range(args.mc_samples):

                outputs = net(data, sample=True, wireless=True, clamping=args.clamping, pmin=args.pmin)

                loss_ce = compute_empirical_risk(outputs, target, args.pmin, args.clamping)
                pred = outputs.max(1, keepdim=True)[1]
                correct_population += pred.eq(target.view_as(pred)).sum().item()
                cross_entropy_population += loss_ce.item()

    cross_entropy_population /= (len(population_loader) * args.mc_samples)
    error_population = 1.0 - (correct_population / (len(population_loader) * population_loader.batch_size * args.mc_samples))

    return error_population, cross_entropy_population