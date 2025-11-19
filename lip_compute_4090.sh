#!/bin/bash

# Script to compute Lipschitz constant for a given model and dataset

python lip_compute.py --name_data cifar10 --model cnn --layers 13 --init_prior learnt --l_0 4 --channel_type bec --clamping --lip_name test-set_mc-dist --gpu 1

python lip_compute.py --name_data cifar10 --model cnn --layers 13 --init_prior learnt --l_0 4 --channel_type bec --no-clamping --lip_name test-set_mc-dist --gpu 1

python lip_compute.py --name_data cifar10 --model cnn --layers 13 --init_prior random --l_0 4 --channel_type bec --clamping --lip_name test-set_mc-dist --gpu 1

python lip_compute.py --name_data cifar10 --model cnn --layers 13 --init_prior random --l_0 4 --channel_type bec --no-clamping --lip_name test-set_mc-dist --gpu 1

python lip_compute.py --name_data cifar10 --model cnn --layers 15 --init_prior learnt --l_0 4 --channel_type bec --clamping --lip_name test-set_mc-dist --gpu 1

python lip_compute.py --name_data cifar10 --model cnn --layers 15 --init_prior learnt --l_0 4 --channel_type bec --no-clamping --lip_name test-set_mc-dist --gpu 1

python lip_compute.py --name_data cifar10 --model cnn --layers 15 --init_prior random --l_0 4 --channel_type bec --clamping --lip_name test-set_mc-dist --gpu 1

python lip_compute.py --name_data cifar10 --model cnn --layers 15 --init_prior random --l_0 4 --channel_type bec --no-clamping --lip_name test-set_mc-dist --gpu 1