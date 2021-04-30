
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


class AbstractTarget(object):
    """Base class for target dynamical systems
    """
    def __init__(self, dim):
        self.dim = dim

    def f(self, y, alpha):
        """Forcing function

        Arguments:
            y {ndarray [N, dim]} -- state
            alpha {ndarray [N, alpha_dim]} -- params

        Returns:
            ndarray [N, dim] -- forcing f(state, params)
        """
        raise NotImplementedError

    def step(self, y, h, alpha):
        """Solution operator

        Arguments:
            y {ndarray [N, dim]} -- initial conditions (N different ones)
            h {positive ndarray [N, 1]} -- terminal time of integration for each initial cond
            alpha {ndarray [N, dim, dim]} -- params

        Returns:
            ndarray [N, dim] -- Solution y(h) = y(0) + int_{0}^{h} f(y(s), alpha) ds
        """
        raise NotImplementedError

    def sample_alpha(self, n_samples):
        """Sample params

        Arguments:
            n_samples {positive int} -- number of samples

        Returns:
            ndarray [n_samples, alpha_dim] -- each row is a iid sample of alpha
        """
        raise NotImplementedError

    def sample_y0(self, n_samples):
        """Sample initial conditions

        Arguments:
            n_samples {positive int} -- number of samples

        Raises:
            NotImplementedError: [description]

        Returns:
            ndarray [n_samples, dim] -- each row is a idd sample of init cond
        """
        raise NotImplementedError

    def generate_output(self, y, h, alpha):
        """Generate the inputs for NoGroundTruthMSETaylorTrainer
        Arguments:
            Arguments:
            y {ndarray [N, dim]} -- initial conditions (N different ones)
            h {positive ndarray [N, 1]} -- terminal time of integration for each initial cond
            alpha {ndarray [N, dim, dim]} -- params
        
        Returns:
            The labels for the prediction that have the same shape with target.step
            If we have true data, we directly use it (eg. target.step for some simple examples or some data trajectory).
            If we do not have ground truth, we use Taylor expansion as the approximation to replace the ground truth.
        """


class ScalarLinearTarget(AbstractTarget):
    """Linear dynamical system target:
        dy/dt = f(y, alpha) = - alpha * y
        y can be high dimensional but the same scalar alpha is multiplied to y
    The solution operator for this sytem is:
        y(t) = e^{- alpha * t} y(0)
    The sampling schemes are:
        alpha -> Uniform[1, 5]
        y0 -> Uniform[-5, 5]
    The definition of alpha:
        alpha_dim means the dimension of alpha matrix
        alpha_num means the number of alpha parameters
    """
    def __init__(self, dim):
        self.alpha_dim = 1
        self.alpha_num = 1
        super().__init__(dim)

    def f(self, y, alpha):
        alpha = alpha[:,0,:,:][:,0,:]
        return -alpha * y

    def step(self, y, h, alpha):
        alpha = alpha[:,0,:,:][:,0,:]
        return tf.math.exp(-h * alpha) * y

    def sample_alpha(self, n_samples):
        return tf.random.uniform(
            minval=1.0,
            maxval=5.0,
            shape=(n_samples, self.alpha_num, self.alpha_dim, self.alpha_dim),
            dtype='float64'
        )

    def sample_y0(self, n_samples):
        return tf.random.uniform(
            minval=-5.0,
            maxval=5.0,
            shape=(n_samples, self.dim),
            dtype='float64'
        )

    def generate_output(self, y, h, alpha):
        return self.step(y, h, alpha)


class ScalarNonlinearTarget(ScalarLinearTarget):
    """Nonlinear dynamical system target:
        dy/dt = f(y, alpha) = - alpha * y^{2}
        y can be high dimensional but the same scalar alpha is multiplied to y
    The solution operator for this system is:
        y(t) = 1/(alpha * t + (1/y(0))) 
    The sampling schemes are:
        alpha -> Uniform[0.1, 0.5]
        y0 -> Uniform[1, 5] (We need y0 > 0)
    """
    def f(self, y, alpha):
        alpha = alpha[:,0,:,:][:,0,:]
        return -alpha * y **2 

    def step(self, y, h, alpha):
        alpha = alpha[:,0,:,:][:,0,:]
        return 1/(alpha * h + (1 / y))

    def sample_alpha(self, n_samples):
        return tf.random.uniform(
            minval=0.1,
            maxval=0.5,
            shape=(n_samples, self.alpha_num, self.alpha_dim, self.alpha_dim),
            dtype='float64'
        )

    def sample_y0(self, n_samples):
        return tf.random.uniform(
            minval=1.0,
            maxval=3.0,
            shape=(n_samples, self.dim),
            dtype='float64'
        )


class VanderPolTarget(AbstractTarget):
    """
    Van der Pol Oscillator
    The dimension of state y is 2, the equation is shown in
    (https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)
    The sampling schemes are:
        alpha -> Uniform[1, 2]
        y0=(y1, y2): y1 -> Uniform[-4, -3], y2 -> Uniform[0, 2]
    Use Taylor expansion to replace ground truth by function 'taylorseries_step'
    """
    def __init__(self, dim):
        self.alpha_dim = 1
        self.alpha_num = 1
        super().__init__(dim)

    def f(self, y, alpha):
        y1 = y[:,0:1]
        y2 = y[:,-1:]
        f1 = y2
        alpha = alpha[:,0,:,:]
        f2 = tf.einsum('ij,ijk->ik',1-y1**2, alpha)*y2 - y1
        return tf.concat([f1,f2],axis=-1)

    def sample_alpha(self, n_samples):
        return tf.random.uniform(
            minval=1.0,
            maxval=2.0,
            shape=(n_samples, self.alpha_num, self.alpha_dim, self.alpha_dim),
            dtype='float64'
        )

    def sample_fixed_alpha(self, n_samples, alpha_fixed_para):
        return alpha_fixed_para * tf.ones(
            shape=(n_samples, self.alpha_num, self.alpha_dim, self.alpha_dim),
            dtype='float64'
        )

    def sample_y0(self, n_samples):
        y1 = tf.random.uniform(    
            minval=-4.0,
            maxval=-3.0,
            shape=(n_samples, 1),
            dtype='float64'
        )
        
        y2 = tf.random.uniform(
            minval=0.0,
            maxval=2.0,
            shape=(n_samples, 1),
            dtype='float64'
        )
        return tf.concat([y1,y2], axis=-1)

    def sample_fixed_y0(self, n_samples, y0_fixed_para):
        return y0_fixed_para * tf.ones(shape=(n_samples, self.dim), dtype='float64')

    def get_k(self, y, k, a, h, alpha):
        sub_nn = y + a * k
        output_k = self.f(sub_nn, alpha) * h
        return output_k

    def rk4_step(self, y, h, alpha):
        k1 = self.f(y, alpha) * h
        k2 = self.get_k(y, k1, 0.5, h, alpha)
        k3 = self.get_k(y, k2, 0.5, h, alpha)
        k4 = self.get_k(y, k3, 1.0, h, alpha)
        yh = y + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        return yh

    @tf.function
    def step(self, y, h, alpha):
        N = 20
        hs = h/N
        soln = [y]
        for i in range(N):
            y_current = soln[-1]
            y_next = self.rk4_step(y_current,hs,alpha)
            soln.append(y_next)
        return soln[-1]

    # calculate n!
    def fact(self, n):
        if n==1:
            return 1
        return n * self.fact(n - 1)
   
    @tf.function
    def calc_exact_soln_derivs(self,depth, y, alpha):
        grads = []
        tapes = []
        func_f = self.f
        
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

    def taylorseries_step(self, y, h,alpha):
        depth=8
        derivs = self.calc_exact_soln_derivs(depth, y, alpha)

        for i in range(depth+1):
            if i == 0:
                y_approx = y
            else:
                y_order = derivs[i-1]
                y_approx = y_approx + 1/self.fact(i) * y_order * h**i
        return y_approx

    def generate_output(self, y, h, alpha):
        return self.taylorseries_step(y, h, alpha)


class BrusselatorTarget(AbstractTarget):
    """
    The Brusselator
    The dimension of state y is 2, the general equation is shown in
    (https://en.wikipedia.org/wiki/Brusselator)
    We use the form shown in
    (http://www-dimat.unipv.it/~boffi/teaching/download/Brusselator.pdf)
    The sampling schemes are:
        a = 1
        b -> Uniform[0.5, 2]
        y0=(y1, y2): y1 -> Uniform[1.5, 3], y2 -> Uniform[2, 3]
    Use Taylor expansion to replace ground truth by function 'taylorseries_step'
    """
    def __init__(self, dim):
        self.alpha_dim = 1
        self.alpha_num = 2
        super().__init__(dim)

    def f(self, y, alpha):
        y1 = y[:,0:1]
        y2 = y[:,-1:]

        a = alpha[:,0,:,:][:,0,:]
        b = alpha[:,1,:,:][:,0,:]
        
        f1 = 1 - (b+1)*y1 + a * y1**2 * y2
        f2 = b*y1 - a*y1**2*y2
        return tf.concat([f1,f2],axis=-1)

    def sample_alpha(self, n_samples):
        a = tf.ones(
            shape=(n_samples, 1, self.alpha_dim, self.alpha_dim),
            dtype='float64'
        )

        b = tf.random.uniform(
            minval=0.5,
            maxval=2.0,
            shape=(n_samples, 1, self.alpha_dim, self.alpha_dim),
            dtype='float64'
        )

        alphas = tf.concat([a,b],axis=1)
        return alphas

    def sample_fixed_alpha(self, n_samples, b_fixed_para, a_fixed_para):
        a = a_fixed_para * tf.ones(
            shape=(n_samples, 1, self.alpha_dim, self.alpha_dim),
            dtype='float64'
        )
             
        b = b_fixed_para * tf.ones(
            shape=(n_samples, 1, self.alpha_dim, self.alpha_dim),
            dtype='float64'
        )

        alphas = tf.concat([a,b],axis=1)
        return alphas

    def sample_y0(self, n_samples):
        y1 = tf.random.uniform(    
            minval=1.5,
            maxval=3.0,
            shape=(n_samples, 1),
            dtype='float64'
        )
        
        y2 = tf.random.uniform(
            minval=2.0,
            maxval=3.0,
            shape=(n_samples, 1),
            dtype='float64'
        )
        return tf.concat([y1,y2], axis=-1)

    def sample_fixed_y0(self, n_samples, y1_fixed_para, y2_fixed_para):
        y1 = y1_fixed_para*tf.ones(shape=(n_samples,1), dtype='float64')
        y2 = y2_fixed_para*tf.ones(shape=(n_samples,1), dtype='float64')
        return tf.concat([y1,y2],axis=-1)

    def get_k(self, y, k, a, h, alpha):
        sub_nn = y + a * k
        output_k = self.f(sub_nn, alpha) * h
        return output_k

    def rk4_step(self, y, h, alpha):
        k1 = self.f(y, alpha) * h
        k2 = self.get_k(y, k1, 0.5, h, alpha)
        k3 = self.get_k(y, k2, 0.5, h, alpha)
        k4 = self.get_k(y, k3, 1.0, h, alpha)
        yh = y + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        return yh

    # Ground Truth
    @tf.function
    def step(self, y, h, alpha):
        N = 20
        hs = h/N
        soln = [y]
        for i in range(N):
            y_current = soln[-1]
            y_next = self.rk4_step(y_current,hs,alpha)
            soln.append(y_next)
        return soln[-1]

    def fact(self, n):
        if n==1:
            return 1
        return n * self.fact(n - 1)
   
    @tf.function
    def calc_exact_soln_derivs(self,depth, y, alpha):
        grads = []
        tapes = []
        func_f = self.f
        
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

    def taylorseries_step(self, y, h,alpha):
        depth=8
        derivs = self.calc_exact_soln_derivs(depth, y, alpha)

        for i in range(depth+1):
            if i == 0:
                y_approx = y
            else:
                y_order = derivs[i-1]
                y_approx = y_approx + 1/self.fact(i) * y_order * h**i
        return y_approx

    def generate_output(self, y, h, alpha):
        return self.taylorseries_step(y, h, alpha)
