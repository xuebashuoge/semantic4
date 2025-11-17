#!/bin/bash

# Script to compute Lipschitz constant for a given model and dataset

python lip_compute.py --name_data mnist --model fcn --layers 4 --init_prior learnt --l_0 2 --channel_type bec --clamping --lip_name test-set

python lip_compute.py --name_data mnist --model fcn --layers 4 --init_prior learnt --l_0 2 --channel_type bec --no-clamping --lip_name test-set

python lip_compute.py --name_data cifar10 --model cnn --layers 9 --init_prior learnt --l_0 4 --channel_type bec --clamping --lip_name test-set

python lip_compute.py --name_data cifar10 --model cnn --layers 9 --init_prior learnt --l_0 4 --channel_type bec --no-clamping --lip_name test-set

python lip_compute.py --name_data cifar10 --model cnn --layers 9 --init_prior random --l_0 4 --channel_type bec --clamping --lip_name test-set

python lip_compute.py --name_data cifar10 --model cnn --layers 9 --init_prior random --l_0 4 --channel_type bec --no-clamping --lip_name test-set