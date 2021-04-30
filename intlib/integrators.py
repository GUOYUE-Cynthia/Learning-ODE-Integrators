
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Multiply, Lambda, Concatenate
from intlib.layers import MultiDense, SoftmaxDense
from numpy.random import seed


# tf.compat.v1.random.set_random_seed(100)

class AbstractIntegrator(object):
    """Base class for integrators

    To instantiate you should pass
        target -> an instance of the AbstractTarget (and derived) classes
        dim -> the dimension of the state, should match target.dim
    """
    def __init__(self, target, dim):
        self.target = target
        self.dim = dim
        self.build()

    def build(self):
        """
        Build the models here, if necessary
        """
        pass

    def step(self, y, h, alpha):
        """Integrator Stepper

        Arguments:
            y {ndarray [N, dim]} -- current state
            h {positive ndarray [N, 1]} -- step sizes
            alpha {ndarray [N, alpha_dim]} -- params

        Returns:
            ndarray [N, dim] -- next state at time +h
        """
        return 0
        # raise NotImplementedError



class RK2Integrator(AbstractIntegrator):
    """The Runge-Kutta (Order 2) Integrator 
    """
    def get_k(self, y, k, a, h, alpha):
        sub_nn = y + a * k
        output_k = self.target.f(sub_nn, alpha) * h
        return output_k

    def step(self, y, h, alpha):
        k1 = self.target.f(y, alpha) * h
        k2 = self.get_k(y, k1, 1.0, h, alpha)
        yh = y + (k1 / 2 + k2 / 2)
        return yh


class RK3Integrator(AbstractIntegrator):
    """The Runge-Kutta (Order 3) Integrator
    """

    def step(self, y, h, alpha):
        k1 = self.target.f(y, alpha) * h
        k2 = self.target.f(y + k1 / 2, alpha) * h
        k3 = self.target.f(y - k1 + 2 * k2, alpha) * h
        yh = y + (k1 / 6 + k2 * 2/3 + k3 / 6)
        return yh


class RK4Integrator(AbstractIntegrator):
    """The Runge-Kutta (Order 4) Integrator 
    """
    def get_k(self, y, k, a, h, alpha):
        sub_nn = y + a * k
        output_k = self.target.f(sub_nn, alpha) * h
        return output_k

    def step(self, y, h, alpha):
        k1 = self.target.f(y, alpha) * h
        k2 = self.get_k(y, k1, 0.5, h, alpha)
        k3 = self.get_k(y, k2, 0.5, h, alpha)
        k4 = self.get_k(y, k3, 1.0, h, alpha)
        yh = y + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        return yh


class RKLikeMatrixIntegrator(AbstractIntegrator):
    """RKLikeMatrixintegrator based on notations in
    (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)

    The coefficients are left unknown, and is trainable.
    The order parameter sets the number of "k" blocks. The default is 3.
    """
    def __init__(self, target, dim, order=3):
        self.order = order
        super().__init__(target, dim)

    def build(self):
        y = Input((self.dim, ))
        h = Input((1, ))
        alpha = Input((self.target.alpha_num, self.target.alpha_dim, self.target.alpha_dim, ))

        f = lambda u: self.target.f(u[0], u[1])  # Wrap f as a lambda
        fy = Lambda(f)([y, alpha])
        self.ks = [Multiply()([fy, h])]
        for j in range(self.order - 1):
            # modify the model similar to the general RK.
            current_ks = []
            for i in range(len(self.ks)):
                current_ks.append(self.ks[i])
            temp = MultiDense(name=f'k_{j}')(current_ks) # this is a fully connected layer, using all the previous k_i calculated before.
            # temp = MultiDense(name=f'k_{j}')([self.ks[-1]]) # only use k_i from last step
            temp = Add()([y, temp])
            temp = Lambda(f)([temp, alpha])
            k_next = Multiply()([temp, h])
            self.ks.append(k_next)
        ks_dense = SoftmaxDense(name='final')(self.ks)
        y_next = Add()([y, ks_dense])
        outputs = Concatenate()([y_next, h])
        self.model = Model(inputs=[y, h, alpha], outputs=outputs)

    def step(self, y, h, alpha):
        outputs = self.model([y, h, alpha])
        return outputs[:, :-1]  # recall, last column is h, so we throw it

    def set_rk4_weights(self):
        """
        Set weight = rk4 weights. Only use if order=4
        """
        assert self.order == 4, f'Only use if order=4. Got order={self.order}.'
        ln2 = np.log(2.0)
        ln4 = np.log(4.0)
  
        rk4_weights = [
             0.5, 0.5, 1.0, ln2, ln4, ln4, ln2
         ]
        self.model.set_weights(
            [np.diag(w * np.ones(1)) for w in rk4_weights])

    def set_rk3_weights(self):
        """
        Set weight = rk3 weights. Only use if order=3
        """
        assert self.order == 3, f'Only use if order=3. Got order={self.order}.'
        ln4 = np.log(4.0)
        rk3_weights = [
             0.5, -1.0, 2.0, 0.0, ln4, 0.0
            ]
        self.model.set_weights(
            [np.diag(w * np.ones(1)) for w in rk3_weights])

    def set_rk2_weights(self):
        """
        Set weight = rk2 weights. Only use if order=2
        """
        assert self.order == 2, f'Only use if order=2. Got order={self.order}.'
        rk2_weights = [
             1.0, 0.5, 0.5
            ]
        self.model.set_weights(
            [np.diag(w * np.ones(1)) for w in rk2_weights])


class TaylorSeriesIntegrator(AbstractIntegrator):
    """
    An integrator based on Taylor series
    """
    # calculate n!
    def fact(self, n):
        if n==1:
            return 1
        return n * self.fact(n - 1)
   
    @tf.function
    def calc_exact_soln_derivs(self, depth, y, alpha):
        grads = []
        tapes = []
        func_f = self.target.f
        
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

    def step(self, y, h,alpha):
        depth=3 # highest order of Taylor expansion
        derivs = self.calc_exact_soln_derivs(depth, y, alpha)

        for i in range(depth+1):
            if i == 0:
                y_approx = y
            else:
                y_order = derivs[i-1]
                y_approx = y_approx + 1/self.fact(i) * y_order * h**i
        return y_approx

