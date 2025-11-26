#!/bin/bash

python test.py --name_data mnist --model cnn --layers 4 --l_0 2 --perc_prior 0.5

python test.py --name_data mnist --model fcn --layers 4 --l_0 2 --perc_prior 0.5

