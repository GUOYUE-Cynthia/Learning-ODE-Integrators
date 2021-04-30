
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import numpy as np
from tqdm import tqdm


class AbstractEvaluator(object):
    def evaluate(self, integrator, target):
        """Evaluate the performance of integrator on target
        Arguments:
            integrator {AbastractIntegrator instance} -- integrator
            target {AbstractTarget instance} -- target
        Returns:
            object -- some quantification of the error
        """
        return 0


class OrderEvaluator(object):
    """Evaluate the order of accuracy of integrator
        the evaluate method returns hs, errors, from which one can plot the order by
            plt.loglog(hs, errors)
        whose slope gives the desired order. For example, for rk4 this should be 4.
    """
    # @tf.function
    def compute_soln(self, integrator, y0, alpha, t_grid):
        """Compute the solution of the integrator acting on initial condition y0 on times t_grid
        Arguments:
            integrator {AbstractIntegrator or AbstractTarget instance} -- integrator or soln operator
            y0 {ndarray [N, dim]} -- initial condition
            alpha {ndarray [N, alpha_dim]} -- params
            t_grid {ndarray [M, ]} -- time grid points
        Returns:
            ndarray [N, M, dim] -- integrated solution
                [n, m, :] -> solution of nth sample at time step t_grid[m]
        """
        soln = [y0]
        for h in np.diff(t_grid):
            y_current = soln[-1]
            hs = h * np.ones((y_current.shape[0], 1))
            y_next = integrator.step(y_current, hs, alpha)
            soln.append(y_next)
        soln = np.asarray(soln).transpose(1, 0, 2)
        return soln

    # @tf.function
    def compute_error(self, integrator, target, y0, h, alpha, T):
        """Compute the error between integrator and target solutions
        Arguments:
            integrator {AbstractIntegrator instance} -- integrator
            target {AbstractTarget instance} -- target
            y0 {ndarray [N, dim]} -- initial conditions
            h {ndarray [N, 1]} -- step sizes
            alpha {ndarray [N, alpha_dim]} -- params
            T {positive float} -- terminal time
        Returns:
            positive float -- averaged error
                1/N sum_{i=1}^N max_{j\leq T/h} || y_target[i, j, :] - y_integrator[i, j, :] ||_2
        """
        N = int(T / h)
        t_grid = np.linspace(0, T, N)
        exact_soln = self.compute_soln(target, y0, alpha, t_grid)
        int_soln = self.compute_soln(integrator, y0, alpha, t_grid)
        diff = exact_soln - int_soln
        # calculate the error at each step
        feature_wise_error = np.linalg.norm(diff, ord=2, axis=-1)
        # choose the largest error as the final evalluated error 
        time_wise_error = np.linalg.norm(
            feature_wise_error,
            ord=np.inf,
            axis=-1,
        )
        average_error = np.mean(time_wise_error)
        return average_error

    def evaluate(self,
                 integrator,
                 target,
                 y0=None,
                 alpha=None,
                 n_samples=10,
                 T=2.0,
                 hrange=(-1,-2),
                 n_h=10):
        """evaluate integrator vs target
        Arguments:
            integrator {AbstractIntegrator instance} -- integrator
            target {AbstractTarget instance} -- target
        Keyword Arguments:
            y0 {ndarray [N, d]} -- initial conditions, if None we take random sample (default: {None})
            alpha {ndarray [N, alpha_dim]} -- initial conditions, if None we take random sample (default: {None})
            n_samples {int} -- #samples (default: {10})
            T {float} -- terminal time (default: {2.0})
            hrange {tuple} -- range of logspace for h (default: {(0, -3)})
            n_h {int} -- #grid points for h (default: {10})
        Returns:
            (ndarray [N, ], ndarray [N, ]) -- (hs, errors) both of size [N, ]
        """
        assert T > 10.0**hrange[
            0], f'T={T} must be > than the biggest h={10.0**hrange[0]}'
        hs = np.logspace(*hrange, n_h)  # log-uniform grid for step sizes
        if y0 is None:
            tf.random.set_seed(100) # Fix the random seed to confirm that the state input for each time step is the same.
            y0 = target.sample_y0(n_samples=n_samples)
        if alpha is None:
            tf.random.set_seed(1) # Fix the random seed to confirm that the parameter input for each time step is the same.
            alpha = target.sample_alpha(n_samples=n_samples)
        error_func = lambda h: self.compute_error(
            integrator=integrator, target=target, y0=y0, h=h, alpha=alpha, T=T)
        errors = list(map(error_func, tqdm(hs)))  # tqdm for progress bar
        return hs, errors


class GMeanEvaluator(object):
    """Evaluate the relatice error of integrator
        the evaluate method returns hs, errors, 
        then compare the errors with those from reference Rk method to obtain the relatice error
        Finally we use geometric mean to see the average.
    """
    # @tf.function
    def compute_soln(self, integrator, y0, alpha, t_grid):
        """Compute the solution of the integrator acting on initial condition y0 on times t_grid
        Arguments:
            integrator {AbstractIntegrator or AbstractTarget instance} -- integrator or soln operator
            y0 {ndarray [N, dim]} -- initial condition
            alpha {ndarray [N, alpha_dim]} -- params
            t_grid {ndarray [M, ]} -- time grid points
        Returns:
            ndarray [N, M, dim] -- integrated solution
                [n, m, :] -> solution of nth sample at time step t_grid[m]
        """
        soln = [y0]
        for h in np.diff(t_grid):
            y_current = soln[-1]
            hs = h * np.ones((y_current.shape[0], 1))
            y_next = integrator.step(y_current, hs, alpha)
            soln.append(y_next)
        soln = np.asarray(soln).transpose(1, 0, 2)
        return soln

    # @tf.function
    def compute_error(self, integrator, target, y0, h, alpha, T):
        """Compute the error between integrator and target solutions
        Arguments:
            integrator {AbstractIntegrator instance} -- integrator
            target {AbstractTarget instance} -- target
            y0 {ndarray [N, dim]} -- initial conditions
            h {ndarray [N, 1]} -- step sizes
            alpha {ndarray [N, alpha_dim]} -- params
            T {positive float} -- terminal time
        Returns:
            positive float -- averaged error
                sqrt[n]{prod_{i=1}^N max_{j\leq T/h} || y_target[i, j, :] - y_integrator[i, j, :] ||_2}
        """
        N = int(T / h)
        t_grid = np.linspace(0, T, N)
        exact_soln = self.compute_soln(target, y0, alpha, t_grid)
        int_soln = self.compute_soln(integrator, y0, alpha, t_grid)
        diff = exact_soln - int_soln
        feature_wise_error = np.linalg.norm(diff, ord=2, axis=-1)
        time_wise_error = np.linalg.norm(
            feature_wise_error,
            ord=np.inf,
            axis=-1,
        )
        n = time_wise_error.shape[0]
        multiply = 1
        for i in time_wise_error:
            multiply = (multiply)*(i)
        GMean = (multiply)**(1/n)
        return GMean

    def evaluate(self,
                 integrator,
                 target,
                 y0=None,
                 alpha=None,
                 n_samples=10,
                 T=2.0,
                 hrange=(-1,-2),
                 n_h=10):
        """evaluate integrator vs target
        Arguments:
            integrator {AbstractIntegrator instance} -- integrator
            target {AbstractTarget instance} -- target
        Keyword Arguments:
            y0 {ndarray [N, d]} -- initial conditions, if None we take random sample (default: {None})
            alpha {ndarray [N, alpha_dim]} -- initial conditions, if None we take random sample (default: {None})
            n_samples {int} -- #samples (default: {10})
            T {float} -- terminal time (default: {2.0})
            hrange {tuple} -- range of logspace for h (default: {(0, -3)})
            n_h {int} -- #grid points for h (default: {10})
        Returns:
            (ndarray [N, ], ndarray [N, ]) -- (hs, errors) both of size [N, ]
        """
        assert T > 10.0**hrange[
            0], f'T={T} must be > than the biggest h={10.0**hrange[0]}'
        hs = np.logspace(*hrange, n_h)  # log-uniform grid for step sizes
        if y0 is None:
            tf.random.set_seed(100)
            y0 = target.sample_y0(n_samples=n_samples)
        if alpha is None:
            tf.random.set_seed(1)
            alpha = target.sample_alpha(n_samples=n_samples)
        error_func = lambda h: self.compute_error(
            integrator=integrator, target=target, y0=y0, h=h, alpha=alpha, T=T)
        errors = list(map(error_func, tqdm(hs)))  # tqdm for progress bar
        return hs, errors