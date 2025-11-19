#!/bin/bash

# Script to compute Lipschitz constant for a given model and dataset

python lip_compute_4090.py --name_data mnist --model fcn --layers 4 --init_prior learnt --l_0 2 --channel_type rayleigh --clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

python lip_compute_4090.py --name_data mnist --model fcn --layers 4 --init_prior learnt --l_0 2 --channel_type rayleigh --no-clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

python lip_compute_4090.py --name_data mnist --model fcn --layers 4 --init_prior random --l_0 2 --channel_type rayleigh --clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

python lip_compute_4090.py --name_data mnist --model fcn --layers 4 --init_prior random --l_0 2 --channel_type rayleigh --no-clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

python lip_compute_4090.py --name_data mnist --model cnn --layers 4 --init_prior learnt --l_0 2 --channel_type rayleigh --clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

python lip_compute_4090.py --name_data mnist --model cnn --layers 4 --init_prior learnt --l_0 2 --channel_type rayleigh --no-clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

python lip_compute_4090.py --name_data mnist --model cnn --layers 4 --init_prior random --l_0 2 --channel_type rayleigh --clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

python lip_compute_4090.py --name_data mnist --model cnn --layers 4 --init_prior random --l_0 2 --channel_type rayleigh --no-clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

# python lip_compute_4090.py --name_data cifar10 --model cnn --layers 13 --init_prior learnt --l_0 4 --channel_type bec --clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

# python lip_compute_4090.py --name_data cifar10 --model cnn --layers 13 --init_prior learnt --l_0 4 --channel_type bec --no-clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

# python lip_compute_4090.py --name_data cifar10 --model cnn --layers 13 --init_prior random --l_0 4 --channel_type bec --clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

# python lip_compute_4090.py --name_data cifar10 --model cnn --layers 13 --init_prior random --l_0 4 --channel_type bec --no-clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

# python lip_compute_4090.py --name_data cifar10 --model cnn --layers 15 --init_prior learnt --l_0 4 --channel_type bec --clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

# python lip_compute_4090.py --name_data cifar10 --model cnn --layers 15 --init_prior learnt --l_0 4 --channel_type bec --no-clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

# python lip_compute_4090.py --name_data cifar10 --model cnn --layers 15 --init_prior random --l_0 4 --channel_type bec --clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150

# python lip_compute_4090.py --name_data cifar10 --model cnn --layers 15 --init_prior random --l_0 4 --channel_type bec --no-clamping --lip_name test-set_mc-dist --lip_bs 250 --chunk_size 250 --gpu 2 --mc_samples 150