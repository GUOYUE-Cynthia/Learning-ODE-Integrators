
import numpy as np
import tensorflow as tf


class AbstractIntegrator(object):
    """
    Define taylor loss which depends on the fixed order.
    """ 
    def __init__(self, order):
        self.order = order


class MSETaylorLossFunc(AbstractIntegrator):
    """
    Define the loss with MSE + Taylorseries-based regularizer with a fixed loss order.
    Use RK-3 method as the reference to scale MSE.
    """
    def calc_loss (self, yh_true, outputs_pred, rk3_pred, hs, gamma, mu):
        # mse
        yh_pred = outputs_pred[0]
        pred_derivs = outputs_pred[1]
        exact_derivs = outputs_pred[2]

        # scale
        rk_scale = (rk3_pred-yh_true)**2

        mse_init = (yh_pred-yh_true)**2
        mse_init_mean = tf.reduce_mean(tf.reduce_sum(mse_init, axis=-1)) # plot it to see the curve of MSE loss
        mse_scaled = tf.reduce_mean(tf.reduce_sum(mse_init / rk_scale, axis=-1))
        # multiply gamma if we want to set different weights on MSE and regularizer, gamma for MSE
        mse = gamma * mse_scaled 
        ratio = mse_scaled

        # If diff is [a,b], squ_diff is a^2 + b^2 (which means c_{i}^{2})

        # squared taylorseries based regularizer
        squ_diffs = []
        
        for pred_grad, exact_grad in zip(pred_derivs, exact_derivs):
            diff = pred_grad - exact_grad
            squ_diff = tf.reduce_sum(diff**2, axis=-1)
            
            scale = tf.stop_gradient(tf.reduce_sum(pred_grad**2, axis=-1) + tf.reduce_sum(exact_grad**2, axis=-1))
            squ_diff_scaled = tf.reduce_mean(squ_diff / scale)

            squ_diffs.append(squ_diff_scaled)

        squ_taylorloss_init = tf.reduce_sum(squ_diffs)
        # multiply mu if we want to set different weights on MSE and regularizer, mu for regularizer
        squ_taylorloss = mu /2 * squ_taylorloss_init
        
        print('ratio', mse_scaled.numpy())
        print('squ_taylorloss_init', squ_taylorloss_init.numpy())
        print('gamma', gamma)

        loss_value = mse + squ_taylorloss
        return loss_value, ratio, mse_init_mean, squ_taylorloss_init


