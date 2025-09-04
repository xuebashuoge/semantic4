import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from pbb.models import NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, ProbCNNet9lChannel, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble
from pbb.bounds import PBBobj
from pbb import data
from pbb.data import loaddataset, loadbatches

# TODOS: 1. make a train prior function (bbb, erm)
#        2. make train posterior function 
#        3. rename partitions of data (prior_data, posterior_data, eval_data)
#        4. implement early stopping with validation set & speed
#        5. add data augmentation (maria)
#        6. better way of logging

def train_standard(net, train_loader, test_loader, optimizer, epochs, device='cuda', verbose=True):
    epoch_train_loss = torch.zeros(epochs)
    epoch_train_err = torch.zeros(epochs)
    epoch_test_loss = torch.zeros(epochs)
    epoch_test_err = torch.zeros(epochs)

    for epoch in trange(epochs):
        net.train()
        train_loss = 0
        train_err = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            net.zero_grad()
            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output.data.max(1, keepdim=True)[1]

            train_loss += loss.detach().item()
            train_err += pred.ne(target.data.view_as(pred)).sum().item()

        train_loss /= len(train_loader)
        train_err /= len(train_loader) * train_loader.batch_size
        epoch_train_loss[epoch] = train_loss
        epoch_train_err[epoch] = train_err

        if verbose:
            print(f'Train Epoch: {epoch} \tLoss: {train_loss:.6f}\tError: {train_err:.4f}')

        test_loss = 0
        test_err = 0
        net.eval()
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                output = net(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                test_err += pred.ne(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        test_err /= len(test_loader) * test_loader.batch_size

        epoch_test_loss[epoch] = test_loss
        epoch_test_err[epoch] = test_err

        if verbose:
            print(f'Test set: Average loss: {test_loss:.4f}, Error: {test_err:.4f}\n')

    return epoch_train_loss, epoch_train_err, epoch_test_loss, epoch_test_err



def runexp(name_data, objective, prior_type, model, sigma_prior, pmin, learning_rate, momentum, 
learning_rate_prior=0.01, momentum_prior=0.95, delta=0.025, layers=9, delta_test=0.01, mc_samples=1000, 
samples_ensemble=100, kl_penalty=1, initial_lamb=6.0, train_epochs=100, prior_dist='gaussian', 
verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, verbose_test=False, 
perc_prior=0.2, batch_size=250):
    """Run an experiment with PAC-Bayes inspired training objectives

    Parameters
    ----------
    name_data : string
        name of the dataset to use (check data file for more info)

    objective : string
        training objective to use

    prior_type : string
        could be rand or learnt depending on whether the prior 
        is data-free or data-dependent
    
    model : string
        could be cnn or fcn
    
    sigma_prior : float
        scale hyperparameter for the prior
    
    pmin : float
        minimum probability to clamp the output of the cross entropy loss
    
    learning_rate : float
        learning rate hyperparameter used for the optimiser

    momentum : float
        momentum hyperparameter used for the optimiser

    learning_rate_prior : float
        learning rate used in the optimiser for learning the prior (only
        applicable if prior is learnt)

    momentum_prior : float
        momentum used in the optimiser for learning the prior (only
        applicable if prior is learnt)
    
    delta : float
        confidence parameter for the risk certificate
    
    layers : int
        integer indicating the number of layers (applicable for CIFAR-10, 
        to choose between 9, 13 and 15)
    
    delta_test : float
        confidence parameter for chernoff bound

    mc_samples : int
        number of monte carlo samples for estimating the risk certificate
        (set to 1000 by default as it is more computationally efficient, 
        although larger values lead to tighter risk certificates)

    samples_ensemble : int
        number of members for the ensemble predictor

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    initial_lamb : float
        initial value for the lambda variable used in flamb objective
        (scaled later)
    
    train_epochs : int
        numer of training epochs for training

    prior_dist : string
        type of prior and posterior distribution (can be gaussian or laplace)

    verbose : bool
        whether to print metrics during training

    device : string
        device the code will run in (e.g. 'cuda')

    prior_epochs : int
        number of epochs used for learning the prior (not applicable if prior is rand)

    dropout_prob : float
        probability of an element to be zeroed.

    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)
    
    verbose_test : bool
        whether to print test and risk certificate stats during training epochs

    perc_prior : float
        percentage of data to be used to learn the prior

    batch_size : int
        batch size for experiments
    """

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}

    train, test = data.loaddataset(name_data)
    rho_prior = math.log(math.exp(sigma_prior)-1.0)

    if prior_type == 'rand':
        dropout_prob = 0.0

    # initialise model
    if model == 'cnn':
        if name_data == 'cifar10':
            # only cnn models are tested for cifar10, fcns are only used 
            # with mnist
            if layers == 9:
                net0 = CNNet9l(dropout_prob=dropout_prob).to(device)
            elif layers == 13:
                net0 = CNNet13l(dropout_prob=dropout_prob).to(device)
            elif layers == 15:
                net0 = CNNet15l(dropout_prob=dropout_prob).to(device)
            else: 
                raise RuntimeError(f'Wrong number of layers {layers}')
        else:
            net0 = CNNet4l(dropout_prob=dropout_prob).to(device)
    else:
        net0 = NNet4l(dropout_prob=dropout_prob, device=device).to(device)

    folder = f'results/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}/prior/'
    os.makedirs(folder, exist_ok=True)
    
    if prior_type == 'rand':
        train_loader, test_loader, _, val_bound_one_batch, _, val_bound = data.loadbatches(
            train, test, loader_kargs, batch_size, prior=False, perc_train=perc_train, perc_prior=perc_prior)
        errornet0 = testNNet(net0, test_loader, device=device)
    elif prior_type == 'learnt':
        train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = data.loadbatches(
            train, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)
        optimizer = optim.SGD(
            net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
        
        prior_loss_list = []
        prior_err_list = []
        for epoch in trange(prior_epochs):
            train_loss, train_err = trainNNet(net0, optimizer, epoch, valid_loader,
                      device=device, verbose=verbose)
            prior_loss_list.append(train_loss)
            prior_err_list.append(train_err)

        errornet0 = testNNet(net0, test_loader, device=device)
        

        
        plt.figure()
        plt.plot(range(1,prior_epochs+1), prior_loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Prior NLL loss')
        plt.title(f'Prior NLL loss {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr prior {learning_rate_prior}, momentum prior {momentum_prior}, dropout {dropout_prob}')
        plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lrpri{learning_rate_prior}_mompri{momentum_prior}_kl{kl_penalty}_drop{dropout_prob}_prior_loss.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.plot(range(1,prior_epochs+1), prior_err_list)
        plt.xlabel('Epochs')
        plt.ylabel('Prior 0-1 error')
        plt.title(f'Prior 0-1 error {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr prior {learning_rate_prior}, momentum prior {momentum_prior}, dropout {dropout_prob}')
        plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lrpri{learning_rate_prior}_mompri{momentum_prior}_kl{kl_penalty}_drop{dropout_prob}_prior_err.pdf', dpi=300, bbox_inches='tight')
    else:
        raise RuntimeError(f'Wrong prior type {prior_type}')
    
    torch.save(net0.state_dict(), f'{folder}/prior_net.pth')


    posterior_n_size = len(train_loader.dataset)
    bound_n_size = len(val_bound.dataset)

    toolarge = False
    train_size = len(train_loader.dataset)
    classes = len(train_loader.dataset.classes)

    if model == 'cnn':
        toolarge = True
        if name_data == 'cifar10':
            if layers == 9:
                net = ProbCNNet9l(rho_prior, prior_dist=prior_dist,
                                    device=device, init_net=net0).to(device)
            elif layers == 13:
                net = ProbCNNet13l(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=net0).to(device)
            elif layers == 15: 
                net = ProbCNNet15l(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=net0).to(device)
            else: 
                raise RuntimeError(f'Wrong number of layers {layers}')
        else:
            net = ProbCNNet4l(rho_prior, prior_dist=prior_dist,
                          device=device, init_net=net0).to(device)
    elif model == 'fcn':
        if name_data == 'cifar10':
            raise RuntimeError(f'Cifar10 not supported with given architecture {model}')
        elif name_data == 'mnist':
            net = ProbNNet4l(rho_prior, prior_dist=prior_dist,
                        device=device, init_net=net0).to(device)
    else:
        raise RuntimeError(f'Architecture {model} not supported')
    # import ipdb
    # ipdb.set_trace()
    bound = PBBobj(objective, pmin, classes, delta,
                    delta_test, mc_samples, kl_penalty, device, n_posterior = posterior_n_size, n_bound=bound_n_size)

    if objective == 'flamb':
        lambda_var = Lambda_var(initial_lamb, train_size).to(device)
        optimizer_lambda = optim.SGD(lambda_var.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer_lambda = None
        lambda_var = None

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    bound_list = []
    kl_list = []
    err_list = []
    loss_list = []

    for epoch in trange(train_epochs):
        avg_bound, avg_kl, avg_loss, avg_err = trainPNNet(net, optimizer, bound, epoch, train_loader, lambda_var, optimizer_lambda, verbose)
        bound_list.append(avg_bound)
        kl_list.append(avg_kl)
        err_list.append(avg_err)
        loss_list.append(avg_loss)
        if verbose_test and ((epoch+1) % 5 == 0):
            train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge,
            bound, device=device, lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

            stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
            post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
            ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

            print(f"***Checkpoint results***")         
            print(f"Objective, Dataset, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_CE, Risk_01, KL, Train NLL loss, Train 01 error, Stch loss, Stch 01 error, Post mean loss, Post mean 01 error, Ens loss, Ens 01 error, 01 error prior net, perc_train, perc_prior")
            print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")

    train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge, bound, device=device,
    lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

    stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
    post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
    ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

    print(f"***Final results***") 
    print(f"Objective: {objective}, Dataset: {name_data}, Sigma: {sigma_prior :.5f}, pmin: {pmin :.5f}, LR: {learning_rate :.5f}, momentum: {momentum :.5f}, LR_prior: {learning_rate_prior :.5f}, momentum_prior: {momentum_prior :.5f}, kl_penalty: {kl_penalty : .5f}, dropout: {dropout_prob :.5f}, Obj_train: {train_obj :.5f}, Risk_CE: {risk_ce :.5f}, Risk_01: {risk_01 :.5f}, KL: {kl :.5f}, Train NLL loss: {loss_ce_train :.5f}, Train 01 error: {loss_01_train :.5f}, Stch loss: {stch_loss :.5f}, Stch 01 error: {stch_err :.5f}, Post mean loss: {post_loss :.5f}, Post mean 01 error: {post_err :.5f}, Ens loss: {ens_loss :.5f}, Ens 01 error: {ens_err :.5f}, 01 error prior net: {errornet0 :.5f}, perc_train: {perc_train :.5f}, perc_prior: {perc_prior :.5f}")
    # print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")

    folder = f'figures/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}/'
    os.makedirs(folder, exist_ok=True)

    plt.figure()
    plt.plot(range(1,train_epochs+1), bound_list)
    plt.xlabel('Epochs')
    plt.ylabel('Training objective')
    plt.title(f'Training objective {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr {learning_rate}, momentum {momentum}, kl penalty {kl_penalty}, dropout {dropout_prob}')
    plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}_obj.pdf', dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(range(1,train_epochs+1), kl_list)
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence')
    plt.title(f'KL divergence {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr {learning_rate}, momentum {momentum}, kl penalty {kl_penalty}, dropout {dropout_prob}')
    plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}_kl.pdf', dpi=300, bbox_inches='tight')
    
    plt.figure()
    plt.plot(range(1,train_epochs+1), loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Training NLL loss')
    plt.title(f'Training NLL loss {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr {learning_rate}, momentum {momentum}, kl penalty {kl_penalty}, dropout {dropout_prob}')
    plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}_loss.pdf', dpi=300, bbox_inches='tight')
    
    plt.figure()
    plt.plot(range(1,train_epochs+1), err_list)
    plt.xlabel('Epochs')
    plt.ylabel('Training 0-1 error')
    plt.title(f'Training 0-1 error {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr {learning_rate}, momentum {momentum}, kl penalty {kl_penalty}, dropout {dropout_prob}')
    plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}_err.pdf', dpi=300, bbox_inches='tight')
    # plt.close('all')

    torch.save(net.state_dict(), f'{folder}/posterior_net.pth')

def myexp(name_data, objective, prior_type, model, sigma_prior, pmin, learning_rate, momentum, 
learning_rate_prior=0.01, momentum_prior=0.95, delta=0.025, layers=9, delta_test=0.01, mc_samples=1000, 
samples_ensemble=100, kl_penalty=1, initial_lamb=6.0, train_epochs=100, prior_dist='gaussian', 
verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, verbose_test=False, 
perc_prior=0.2, batch_size=250):
    """Run an experiment with PAC-Bayes inspired training objectives

    Parameters
    ----------
    name_data : string
        name of the dataset to use (check data file for more info)

    objective : string
        training objective to use

    prior_type : string
        could be rand or learnt depending on whether the prior 
        is data-free or data-dependent
    
    model : string
        could be cnn or fcn
    
    sigma_prior : float
        scale hyperparameter for the prior
    
    pmin : float
        minimum probability to clamp the output of the cross entropy loss
    
    learning_rate : float
        learning rate hyperparameter used for the optimiser

    momentum : float
        momentum hyperparameter used for the optimiser

    learning_rate_prior : float
        learning rate used in the optimiser for learning the prior (only
        applicable if prior is learnt)

    momentum_prior : float
        momentum used in the optimiser for learning the prior (only
        applicable if prior is learnt)
    
    delta : float
        confidence parameter for the risk certificate
    
    layers : int
        integer indicating the number of layers (applicable for CIFAR-10, 
        to choose between 9, 13 and 15)
    
    delta_test : float
        confidence parameter for chernoff bound

    mc_samples : int
        number of monte carlo samples for estimating the risk certificate
        (set to 1000 by default as it is more computationally efficient, 
        although larger values lead to tighter risk certificates)

    samples_ensemble : int
        number of members for the ensemble predictor

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    initial_lamb : float
        initial value for the lambda variable used in flamb objective
        (scaled later)
    
    train_epochs : int
        numer of training epochs for training

    prior_dist : string
        type of prior and posterior distribution (can be gaussian or laplace)

    verbose : bool
        whether to print metrics during training

    device : string
        device the code will run in (e.g. 'cuda')

    prior_epochs : int
        number of epochs used for learning the prior (not applicable if prior is rand)

    dropout_prob : float
        probability of an element to be zeroed.

    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)
    
    verbose_test : bool
        whether to print test and risk certificate stats during training epochs

    perc_prior : float
        percentage of data to be used to learn the prior

    batch_size : int
        batch size for experiments
    """

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device == 'mps':
        torch.use_deterministic_algorithms(True)

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}

    train, test = data.loaddataset(name_data)
    rho_prior = math.log(math.exp(sigma_prior)-1.0)

    if prior_type == 'rand':
        dropout_prob = 0.0

    # initialise model
    if model == 'cnn':
        if name_data == 'cifar10':
            # only cnn models are tested for cifar10, fcns are only used 
            # with mnist
            if layers == 9:
                net0 = CNNet9l(dropout_prob=dropout_prob).to(device)
            elif layers == 13:
                net0 = CNNet13l(dropout_prob=dropout_prob).to(device)
            elif layers == 15:
                net0 = CNNet15l(dropout_prob=dropout_prob).to(device)
            else: 
                raise RuntimeError(f'Wrong number of layers {layers}')
        else:
            net0 = CNNet4l(dropout_prob=dropout_prob).to(device)
    elif model == 'fcn':
        if name_data == 'cifar10':
            raise RuntimeError(f'Cifar10 not supported with given architecture {model}')
        elif name_data == 'mnist':
            net0 = NNet4l(dropout_prob=dropout_prob, device=device).to(device)
    else:
        raise RuntimeError(f'Architecture {model} not supported')

    folder = f'results/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}/'
    prior_folder = folder + 'prior/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(prior_folder, exist_ok=True)
    
    if prior_type == 'rand':
        train_loader, test_loader, _, val_bound_one_batch, _, val_bound = data.loadbatches(train, test, loader_kargs, batch_size, prior=False, perc_train=perc_train, perc_prior=perc_prior)
        errornet0 = testNNet(net0, test_loader, device=device)
    elif prior_type == 'learnt':
        train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = data.loadbatches(train, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)
        optimizer = optim.SGD(
            net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
        
        prior_train_loss, prior_train_err, prior_test_loss, prior_test_err = train_standard(net0, valid_loader, test_loader, optimizer, prior_epochs, device=device, verbose=verbose)

        errornet0 = testNNet(net0, test_loader, device=device)
        
        plt.figure()
        plt.plot(range(1,prior_epochs+1), prior_train_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Prior NLL loss')
        plt.title(f'Prior NLL loss {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr prior {learning_rate_prior}, momentum prior {momentum_prior}, dropout {dropout_prob}')
        plt.savefig(f'{prior_folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lrpri{learning_rate_prior}_mompri{momentum_prior}_kl{kl_penalty}_drop{dropout_prob}_prior_loss.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        plt.plot(range(1,prior_epochs+1), prior_train_err)
        plt.xlabel('Epochs')
        plt.ylabel('Prior 0-1 error')
        plt.title(f'Prior 0-1 error {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr prior {learning_rate_prior}, momentum prior {momentum_prior}, dropout {dropout_prob}')
        plt.savefig(f'{prior_folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lrpri{learning_rate_prior}_mompri{momentum_prior}_kl{kl_penalty}_drop{dropout_prob}_prior_err.pdf', dpi=300, bbox_inches='tight')
        
        torch.save(optimizer.state_dict(), f'{prior_folder}/prior_optimizer.pth')
        torch.save(prior_train_loss, f'{prior_folder}/prior_train_loss.pth')
        torch.save(prior_train_err, f'{prior_folder}/prior_train_err.pth')
        torch.save(prior_test_loss, f'{prior_folder}/prior_test_loss.pth')
        torch.save(prior_test_err, f'{prior_folder}/prior_test_err.pth')
    else:
        raise RuntimeError(f'Wrong prior type {prior_type}')
    
    torch.save(net0.state_dict(), f'{prior_folder}/prior_net.pth')

    posterior_n_size = len(train_loader.dataset)
    bound_n_size = len(val_bound.dataset)

    toolarge = False
    train_size = len(train_loader.dataset)
    classes = len(train_loader.dataset.classes)

    if model == 'cnn':
        toolarge = True
        if name_data == 'cifar10':
            if layers == 9:
                net = ProbCNNet9l(rho_prior, prior_dist=prior_dist,
                                    device=device, init_net=net0).to(device)
            elif layers == 13:
                net = ProbCNNet13l(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=net0).to(device)
            elif layers == 15: 
                net = ProbCNNet15l(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=net0).to(device)
            else: 
                raise RuntimeError(f'Wrong number of layers {layers}')
        else:
            net = ProbCNNet4l(rho_prior, prior_dist=prior_dist,
                          device=device, init_net=net0).to(device)
    elif model == 'fcn':
        if name_data == 'cifar10':
            raise RuntimeError(f'Cifar10 not supported with given architecture {model}')
        elif name_data == 'mnist':
            net = ProbNNet4l(rho_prior, prior_dist=prior_dist,
                        device=device, init_net=net0).to(device)
    else:
        raise RuntimeError(f'Architecture {model} not supported')
    
    # import ipdb
    # ipdb.set_trace()
    bound = PBBobj(objective, pmin, classes, delta,
                    delta_test, mc_samples, kl_penalty, device, n_posterior = posterior_n_size, n_bound=bound_n_size)

    if objective == 'flamb':
        lambda_var = Lambda_var(initial_lamb, train_size).to(device)
        optimizer_lambda = optim.SGD(lambda_var.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer_lambda = None
        lambda_var = None


    train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge, bound, device=device,
    lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

    stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
    post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
    ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

    print(f"***Final results***") 
    print(f"Objective: {objective}, Dataset: {name_data}, Sigma: {sigma_prior :.5f}, pmin: {pmin :.5f}, LR: {learning_rate :.5f}, momentum: {momentum :.5f}, LR_prior: {learning_rate_prior :.5f}, momentum_prior: {momentum_prior :.5f}, kl_penalty: {kl_penalty : .5f}, dropout: {dropout_prob :.5f}, Obj_train: {train_obj :.5f}, Risk_CE: {risk_ce :.5f}, Risk_01: {risk_01 :.5f}, KL: {kl :.5f}, Train NLL loss: {loss_ce_train :.5f}, Train 01 error: {loss_01_train :.5f}, Stch loss: {stch_loss :.5f}, Stch 01 error: {stch_err :.5f}, Post mean loss: {post_loss :.5f}, Post mean 01 error: {post_err :.5f}, Ens loss: {ens_loss :.5f}, Ens 01 error: {ens_err :.5f}, 01 error prior net: {errornet0 :.5f}, perc_train: {perc_train :.5f}, perc_prior: {perc_prior :.5f}")
    # print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")


    plt.figure()
    plt.plot(range(1,train_epochs+1), bound_list)
    plt.xlabel('Epochs')
    plt.ylabel('Training objective')
    plt.title(f'Training objective {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr {learning_rate}, momentum {momentum}, kl penalty {kl_penalty}, dropout {dropout_prob}')
    plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}_obj.pdf', dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(range(1,train_epochs+1), kl_list)
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence')
    plt.title(f'KL divergence {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr {learning_rate}, momentum {momentum}, kl penalty {kl_penalty}, dropout {dropout_prob}')
    plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}_kl.pdf', dpi=300, bbox_inches='tight')
    
    plt.figure()
    plt.plot(range(1,train_epochs+1), loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Training NLL loss')
    plt.title(f'Training NLL loss {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr {learning_rate}, momentum {momentum}, kl penalty {kl_penalty}, dropout {dropout_prob}')
    plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}_loss.pdf', dpi=300, bbox_inches='tight')
    
    plt.figure()
    plt.plot(range(1,train_epochs+1), err_list)
    plt.xlabel('Epochs')
    plt.ylabel('Training 0-1 error')
    plt.title(f'Training 0-1 error {objective}, {name_data}, {model}, sigma prior {sigma_prior}, pmin {pmin}, lr {learning_rate}, momentum {momentum}, kl penalty {kl_penalty}, dropout {dropout_prob}')
    plt.savefig(f'{folder}/{objective}_{name_data}_{model}_sig{sigma_prior}_pmin{pmin}_lr{learning_rate}_mom{momentum}_kl{kl_penalty}_drop{dropout_prob}_err.pdf', dpi=300, bbox_inches='tight')
    # plt.close('all')

    torch.save(net.state_dict(), f'{folder}/posterior_net.pth')

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_empirical_risk(outputs, targets, pmin, bounded=True):
    # compute negative log likelihood loss and bound it with pmin (if applicable)
    empirical_risk = F.nll_loss(outputs, targets)
    if bounded == True:
        empirical_risk = (1./(np.log(1./pmin))) * empirical_risk
    return empirical_risk

def test_exp(name_data='cifar10', sigma_prior=0.03, dropout_prob=0.2, batch_size=250, perc_train=1.0, perc_prior=0.5, prior_dist='gaussian', l_0=2, channel_type='bec', outage=0.1, mc_samples=100, clamping=True, pmin=1e-5, device='cuda'):
    

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}

    train, test = loaddataset(name_data)
    rho_prior = math.log(math.exp(sigma_prior)-1.0)

    net0 = CNNet9l(dropout_prob=dropout_prob).to(device)

    prior_file = 'results/fclassic_cifar10_cnn_sig0.03_pmin1e-05_lr0.001_mom0.95_kl1_drop0.2/prior/prior_net.pth'

    if os.path.exists(prior_file):
        net0.load_state_dict(torch.load(prior_file, map_location=device))
        print("Loaded prior")
    else:
        raise RuntimeError(f'Prior file {prior_file} not found')
    
    train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = loadbatches(train, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)
    
    posterior_n_size = len(train_loader.dataset)
    bound_n_size = len(val_bound.dataset)

    train_size = len(train_loader.dataset)
    classes = len(train_loader.dataset.classes)

    # load probabilistic model
    toolarge = True
    net = ProbCNNet9lChannel(rho_prior, prior_dist=prior_dist, l_0=l_0, channel_type=channel_type, outage=outage, device=device, init_net=net0).to(device)
    net_wired = ProbCNNet9l(rho_prior, prior_dist=prior_dist, device=device, init_net=net0).to(device)

    net.eval()
    net_wired.eval()

    kl = net.compute_kl()

    # compute empirical risk using mc samples
    correct_empirical = 0.0
    cross_entropy_empirical = 0.0
    total_empirical = 0.0

    correct_empirical_net0 = 0.0
    cross_entropy_empirical_net0 = 0.0
    total_empirical_net0 = 0.0

    for data_batch, target_batch in tqdm(val_bound):
        data_batch, target_batch = data_batch.to(device), target_batch.to(device)

        outputs_net0 = net0(data_batch)
        loss_net0 = compute_empirical_risk(outputs_net0, target_batch, pmin, clamping)
        pred_net0 = outputs_net0.max(1, keepdim=True)[1]
        correct_empirical_net0 += pred_net0.eq(target_batch.view_as(pred_net0)).sum().item()
        total_empirical_net0 += target_batch.size(0)
        cross_entropy_empirical_net0 += loss_net0.item()
        
        for _ in range(mc_samples):
            outputs = net(data_batch, sample=True, wireless=False, clamping=clamping, pmin=pmin)
            loss_ce = compute_empirical_risk(outputs, target_batch, pmin, clamping)
            pred = outputs.max(1, keepdim=True)[1]
            correct_empirical += pred.eq(target_batch.view_as(pred)).sum().item()
            total_empirical += target_batch.size(0)
            cross_entropy_empirical += loss_ce.item()

    cross_entropy_empirical /= (len(val_bound) * mc_samples)
    error_empirical = 1.0 - (correct_empirical / total_empirical)
    cross_entropy_empirical_net0 /= len(val_bound)
    error_empirical_net0 = 1.0 - (correct_empirical_net0 / total_empirical_net0)

    # compute population risk
    correct_population = 0.0
    cross_entropy_population = 0.0
    total_population = 0.0

    correct_population_net0 = 0.0
    cross_entropy_population_net0 = 0.0
    total_population_net0 = 0.0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            outputs_net0 = net0(data)
            loss_net0 = compute_empirical_risk(outputs_net0, target, pmin, clamping)
            pred_net0 = outputs_net0.max(1, keepdim=True)[1]
            correct_population_net0 += pred_net0.eq(target.view_as(pred_net0)).sum().item()
            total_population_net0 += target.size(0)
            cross_entropy_population_net0 += loss_net0.item()

            for _ in range(mc_samples):

                outputs = net(data, sample=True, wireless=True, clamping=clamping, pmin=pmin)

                loss_ce = compute_empirical_risk(outputs, target, pmin, clamping)
                pred = outputs.max(1, keepdim=True)[1]
                correct_population += pred.eq(target.view_as(pred)).sum().item()
                total_population += target.size(0)
                cross_entropy_population += loss_ce.item()

    cross_entropy_population /= (len(test_loader) * mc_samples)
    error_population = 1.0 - (correct_population / total_population)
    cross_entropy_population_net0 /= len(test_loader)
    error_population_net0 = 1.0 - (correct_population_net0 / total_population_net0)

    print(f"***Final results***")
    print(f"Dataset: {name_data}, Sigma: {sigma_prior :.5f}, pmin: {pmin :.5f}, Dropout: {dropout_prob :.5f}, Perc_train: {perc_train :.5f}, Perc_prior: {perc_prior :.5f}, L_0: {l_0}, Channel: {channel_type}, Outage: {outage :.5f}, MC samples: {mc_samples}, Clamping: {clamping}, Prior empirical error: {error_empirical_net0 :.5f}, Prior empirical CE loss: {cross_entropy_empirical_net0 :.5f}, Prior population error: {error_population_net0 :.5f}, Prior population CE loss: {cross_entropy_population_net0 :.5f}, Empirical error: {error_empirical :.5f}, Empirical CE loss: {cross_entropy_empirical :.5f}, Population error: {error_population :.5f}, Population CE loss: {cross_entropy_population :.5f}, KL: {kl :.5f}")

    print('Done!')