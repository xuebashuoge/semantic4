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
from pbb.models import ProbLinear, ProbConv2d, NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, ProbCNNet9lChannel, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet,trainPNNet2, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, compute_empirical_risk
from pbb.bounds import PBBobj
from pbb import data
from pbb.data import loaddataset, loadbatches



def my_exp(args, device='cuda', num_workers=8):
    loader_kargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': num_workers}

    train, test = loaddataset(args.name_data)

    train_loader, test_loader, valid_loader, _, _, _ = loadbatches(train, test, loader_kargs, args.batch_size, prior=True, perc_train=args.perc_train, perc_prior=args.perc_prior)

    train_and_certificate(args, train_loader=train_loader, prior_loader=valid_loader, test_loader=test_loader, empirical_loader=train_loader, population_loader=test_loader, device=device)



def train_and_certificate(args, train_loader, prior_loader, test_loader, empirical_loader, population_loader, device='cuda'):

    rho_prior = math.log(math.exp(args.sigma_prior)-1.0)

    # learn prior
    print('Learning prior...')
    prior_folder = f'results/prior/{args.name}_{args.name_data}_{args.model}-{args.layers}_sig{args.sigma_prior}_pmin{args.pmin}_{args.prior_dist}_epochpri{args.prior_epochs}_bs{args.batch_size}_lrpri{args.learning_rate_prior}_mompri{args.momentum_prior}_drop{args.dropout_prob}_perc{args.perc_prior}/'


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
            net0 = CNNet4l(dropout_prob=args.dropout_prob).to(device)
    elif args.model.lower() == 'fcn':
        net0 = NNet4l(dropout_prob=args.dropout_prob).to(device)
    else:
        raise RuntimeError(f'Wrong model chosen {args.model}')

    train_prior(net0, prior_loader, test_loader, prior_folder, args, device)

    # load probabilistic model
    print('Training posterior...')
    net = ProbCNNet9lChannel(rho_prior, prior_dist=args.prior_dist, l_0=args.l_0, channel_type=args.channel_type, outage=args.outage, device=device, init_net=net0).to(device)

    posterior_folder = f'results/{args.name}_{args.name_data}_{args.model}-{args.layers}_sig{args.sigma_prior}_pmin{args.pmin}_{args.prior_dist}_epoch{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}_mom{args.momentum}_drop{args.dropout_prob}/'
    kl = train_posterior(net, train_loader, posterior_folder, args, device)

    print('Computing certificate...')
    certificate_folder = f'results/{args.name}_{args.name_data}_{args.model}-{args.layers}_sig{args.sigma_prior}_pmin{args.pmin}_{args.prior_dist}_epoch{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}_mom{args.momentum}_drop{args.dropout_prob}/certificate/'

    # compute empirical and population risks
    compute_certificate(net, empirical_loader, population_loader, certificate_folder, kl, args, device)

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
        prior_results_dict = torch.load(f'{folder}/prior_results.pth', weights_only=False, map_location=device)
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
        posterior_results_dict = torch.load(f'{folder}/posterior_results_dict.pth', weights_only=False, map_location=device)
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


def compute_certificate(net, empirical_loader, population_loader, folder, kl, args, device='cuda'):

    os.makedirs(folder, exist_ok=True)

    net.eval()

    # compute Lipschitz constant L_w
    L_w = compute_lipschitz_constant_efficient(net, [empirical_loader, population_loader], args.mc_samples, args.pmin, args.clamping, device)

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

    

    # bound evaluation
    k = torch.sqrt(len(empirical_loader.dataset))
    # sigma-sub-Gaussian is equivalent to bounded in [0, 2*sigma], the loss function here is clamped in [0, log(1/pmin)]
    sigma = math.log(1/args.pmin)/2
    bound_ce = cross_entropy_empirical + k*sigma**2 / (2*len(empirical_loader.dataset)) + 1/k * (kl + torch.log())


    print(f"***Final results***")
    print(f"Dataset: {args.name_data}, Sigma: {args.sigma_prior :.5f}, pmin: {args.pmin :.5f}, Dropout: {args.dropout_prob :.5f}, Perc_train: {args.perc_train :.5f}, Perc_prior: {args.perc_prior :.5f}, L_0: {args.l_0}, Channel: {args.channel_type}, Outage: {args.outage :.5f}, MC samples: {args.mc_samples}, Clamping: {args.clamping}, Empirical error: {error_empirical :.5f}, Empirical CE loss: {cross_entropy_empirical :.5f}, Population error: {error_population :.5f}, Population CE loss: {cross_entropy_population :.5f}, KL: {kl :.5f}")

    results_dict = {
        'args': args,
        'error_empirical': error_empirical,
        'cross_entropy_empirical': cross_entropy_empirical,
        'error_population': error_population,
        'cross_entropy_population': cross_entropy_population,
        'kl': kl
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

def compute_lipschitz_constant_efficient(net, loaders, mc_samples, pmin, clamping, device):
    """
    Computes the Lipschitz constant efficiently using torch.func.vmap.

    Parameters
    ----------
    net : nn.Module
        The trained probabilistic neural network.
    loaders : list of DataLoader
        A list of data loaders to iterate over.
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
            kwargs={'sample': False} # We've already sampled the weights!
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
    vmapped_grad_fn = torch.func.vmap(grad_fn, in_dims=(None, None, 0, 0), chunk_size=16)

    combined_iterator = itertools.chain(*loaders)
    total_batches = sum(len(l) for l in loaders)
    
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

            for data_batch, target_batch in combined_iterator:
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
                
                pbar.update(1)
            
            # Reset the iterator for the next MC sample
            combined_iterator = itertools.chain(*loaders)