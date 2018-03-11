#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from train_emb import run_training

for i in range(10):
    # seed = np.random.randint(0, 10000)
    seed = i
    print('Starting up...')
    run_training(random_seed=seed)
    print('Exiting...')

