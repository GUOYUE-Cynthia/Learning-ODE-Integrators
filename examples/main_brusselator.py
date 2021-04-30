import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf
import pandas as pd
import yaml

from intlib.integrators import RK3Integrator, RKLikeMatrixIntegrator
from intlib.targets import BrusselatorTarget
from intlib.trainers import NoGroundTruthMSETaylorTrainer
from intlib.taylorloss import MSETaylorLossFunc


def run_toy(config):
    dim = config['dim']
    
    # Different targets in our experiments only require some changes here.
    target = BrusselatorTarget(dim=dim)
    
    rk3 = RK3Integrator(target=target, dim=dim)
    
    integrator_stage = config['integrator_stage']
    taylorloss_order = config['taylorloss_order']
    
    inte_seed = config['inte_seed']
    data_seed = config['data_seed']

    tf.random.set_seed(inte_seed)
    rk_nn = RKLikeMatrixIntegrator(target=target, dim=dim, order=integrator_stage)
    
    # Construct the trainer with MSE + Taylorseries-based Regularizer
    MSETaylorloss = MSETaylorLossFunc(order=taylorloss_order).calc_loss

    # Set a relative integrator as a scale reference
    trainer = NoGroundTruthMSETaylorTrainer(integrator=rk_nn, 
                                            refer_integrator=rk3,
                                            target=target,
                                            loss_func=MSETaylorloss,
                                            loss_order=taylorloss_order,
                                            step=config['step'])

    # Generate data 
    tf.random.set_seed(data_seed)
    train_data = trainer.generate_data(train_size=3000)
    
    # Train the model, gamma is the weight of MSE and mu is that of Taylorseries-based Regularizer. 
    # 'ratio' is an observation that shows 'error_rk / error_nn', we want to train is to below 1. 'ratio_lim' is an early_stopping index.
    epochs, current_loss, mse, taylorloss = trainer.train(train_data = train_data,
                                            epochs=config['epochs'],
                                            learning_rate=config['lr'],
                                            gamma_init=config['gamma_init'],
                                            mu_init=config['mu_init'],
                                            ratio_lim=config['ratio_lim'],
                                            coefficient_weight=config['coefficient_weight'],
                                            lr_decay=config['lr_decay'])  

    # Save weigths
    rk_nn.model.save_weights('brusselator_weights.h5')


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml'))
    run_toy(config['brusselator'])