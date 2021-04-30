
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import numpy as np
import pandas as pd
from tqdm.keras import TqdmCallback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import backend as K
from intlib.integrators import RK3Integrator


class AbstractTrainer(object):
    """Abstract Trainer class for integrators

    To instantiate, we need
        integrator -> an instance of the AbstractIntegrator (or derived) class
        target -> an instance of the AbstractTarget (or derived) class
    """
    def __init__(self, integrator, refer_integrator, target, loss_func, loss_order=None, step=None):
        self.integrator = integrator
        self.refer_integrator = refer_integrator
        self.target = target
        assert integrator.dim == target.dim, 'Incompatible target and integrator dimensions.'
        if step is None:
            step = 1
        self.step = step
        if loss_order is None:
            loss_order = 1
        self.loss_order = loss_order
        self.loss_func = loss_func

        self.build()

    def build(self):
        self.model = self.integrator.model

    def train(self, train_data, epochs):
        """Trains the integrator against the target family

        Arguments:
            train_data {list} -- ([y0, h, alpha, h0], yh)
                #rows=N for all, y0 has #cols=dim, h has #cols=1, alpha has #cols=alpha_dim
            epochs {positive int} -- number of epochs to train

        Returns
            pandas dataframe -- training history
        """
        raise NotImplementedError


class NoGroundTruthMSETaylorTrainer(AbstractTrainer):
    def sample_hs(self, n_samples):
        return 10.0**tf.random.uniform(minval=-2, maxval=-1, shape=(n_samples, 1), dtype='float64')

    def sample_h0s(self, n_samples):
        return tf.zeros((n_samples, 1), dtype='float64')

    def generate_inputs(self, n_samples):
        y0s = self.target.sample_y0(n_samples=n_samples)
        hs = self.sample_hs(n_samples=n_samples)
        alphas = self.target.sample_alpha(n_samples=n_samples)
        h0s = self.sample_h0s(n_samples=n_samples)
        return y0s, hs, alphas, h0s  

    def generate_fixed_inputs(self, n_samples, y0_fixed_para, alpha_fixed_para):
        y0s = self.target.sample_fixed_y0(n_samples=n_samples, y0_fixed_para=y0_fixed_para)
        hs = self.sample_hs(n_samples=n_samples)
        alphas = self.target.sample_fixed_alpha(n_samples=n_samples, alpha_fixed_para=alpha_fixed_para)
        h0s = self.sample_h0s(n_samples=n_samples)
        return y0s, hs, alphas, h0s

    def generate_data(self, train_size): 
        """Generate training/testing data

        Arguments:
            train_size {positive int} -- # training data

        Returns:
            tuple -- (inputs_train, outputs_train)
                inputs are of the form [y0, h, alpha, h0]
                outputs are of the form yh
        """
        inputs_train = self.generate_inputs(n_samples=train_size)
        outputs_train = self.target.generate_output(inputs_train[0], inputs_train[1], inputs_train[2])
        return inputs_train, outputs_train

    def train(self, train_data, epochs, learning_rate, gamma_init, mu_init, ratio_lim, coefficient_weight, lr_decay, opt=None):
        epochs = range(epochs)
        inputs = (train_data[0][0], train_data[0][1], train_data[0][2])
        
        y0s = train_data[0][0]
        hs = train_data[0][1]
        alphas = train_data[0][2]
        h0s = train_data[0][3]
        yh = train_data[1]

        n_samples = y0s.shape[0] # train_size

        self.lr = learning_rate

        # optimizer
        if opt == None:
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
        else:
            self.optimizer = opt
        
        func_step = self.integrator.step  # Important: keep it outside of loop to avoid tracing tf.function!
        @tf.function
        def calc_pred_soln_derivs(depth, y, h, alpha):
            grads = []
            tapes = []
            
            # Set up tapes
            for _ in range(depth):
                tape = tf.GradientTape()
                tape.__enter__()
                tape.watch(h)
                tapes.append(tape)
                
            # Feed-forward
            outputs = func_step(y=y, h=h, alpha=alpha)
            
            # Differentiate
            for tape in reversed(tapes):
                tape.__exit__(None, None, None)
                grad = tf.reshape(tape.batch_jacobian(outputs, h), shape=y.shape)
                grads.append(grad)
                outputs = grad

            return grads

        func_f = self.target.f
        @tf.function
        def calc_exact_soln_derivs(depth, y, alpha):
            grads = []
            tapes = []
            
            # Set up tapes
            for _ in range(depth-1):
                tape = tf.GradientTape()
                tape.__enter__()
                tape.watch(y)
                tapes.append(tape)
                
            # Feed-forward
            outputs = func_f(y=y, alpha=alpha)
            grads.append(outputs)
            
            # Differentiate
            for tape in reversed(tapes):
                tape.__exit__(None, None, None)
                jacobian_matrix = tf.transpose(tape.batch_jacobian(grads[-1], y), perm=[0, 2, 1])
                grad = tf.einsum('ij,ijk->ik',outputs, jacobian_matrix)
                grads.append(grad)

            return grads

        self.gamma = gamma_init
        self.mu = mu_init

        rk = self.refer_integrator.step

        self.current_loss = []
        self.current_mse = []
        self.current_taylorloss = []
        self.current_ratio = []
                
        for epoch in epochs:
            with tf.GradientTape() as tape_outer:
                tape_outer.watch(self.integrator.model.weights)

                pred_derivs = calc_pred_soln_derivs(depth=self.loss_order, y=y0s, h=h0s, alpha=alphas)
                exact_derivs = calc_exact_soln_derivs(depth=self.loss_order, y=y0s, alpha=alphas)

                # Multi-step
                h_seg = hs / self.step
                output_1 = self.integrator.model([y0s, h_seg, alphas])
                yh_1 = output_1[:, :-1]
                yh_seg = [yh_1]
                for i in range(self.step - 1):
                    output_next = self.integrator.model([yh_seg[-1], h_seg, alphas])
                    yh_next = output_next[:,:-1]
                    yh_seg.append(yh_next)
                yh_pred = yh_seg[-1]


                outputs_pred = [yh_pred, pred_derivs, exact_derivs]
                rk_pred = rk(y0s, hs, alphas)

                # Simple nonlinear target needs these coefficients with gamma_init = 1.0, mu_init = 1e16.
                if coefficient_weight == True:    
                    if self.gamma < 1e5:
                        self.gamma = self.gamma * 1.01

                # if self.mu > 1e6:
                #     self.mu = self.mu /1.001

                loss, ratio, mse_init, squ_taylorloss_init = self.loss_func(yh, outputs_pred, rk_pred, hs, self.gamma, self.mu)
                

                self.current_loss.append(loss)
                self.current_mse.append(mse_init)
                self.current_taylorloss.append(squ_taylorloss_init)
                self.current_ratio.append(ratio)
                print(epoch, loss.numpy()) 
            dw = tape_outer.gradient(loss, self.integrator.model.weights)


            # # When we want to use ratio as the reference to decrease the learning rate, please uncomment the following lines
            # if ratio < 10 and ratio > tf.reduce_mean(self.current_ratio[-500:]):
            #     self.lr = self.lr * 0.1
            #     self.optimizer.lr = self.lr

            # Reduce learning rate
            if epoch % lr_decay == 0 and epoch > 0:
                self.lr = self.lr * 0.1
                self.optimizer.lr = self.lr

            # Make sure weights won't change when the gradients are very small
            for i in range(len(dw)):
                if tf.math.abs(dw[i]) < 1e-15:
                    dw[i] = tf.zeros(shape=dw[i].shape, dtype='float64')

            self.optimizer.apply_gradients(zip(dw, self.integrator.model.weights))

            # # Save weights during the training
            # if epoch % 200 ==0:
            #     self.integrator.model.save_weights('TrainingModels/epoch%dweights.h5'%(epoch))

            # Earlystopping condition
            if ratio < ratio_lim:
                break

            print('lr', self.optimizer.lr.numpy())
        return epochs, self.current_loss, self.current_mse, self.current_taylorloss
