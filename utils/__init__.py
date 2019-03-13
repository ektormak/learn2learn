import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def generate_regression_task_dataset(n_tasks):
    kernel = RBF(length_scale=1., length_scale_bounds=(1e-5, 1e5))
    gpr = GaussianProcessRegressor(kernel=kernel)
    X = np.linspace(0, 100, 1000)
    X = np.atleast_2d(X).T
    f_samples = []
    for i in range(n_tasks):
        # sample a function from a zero mean GP
        f = gpr.sample_y(X, random_state=i)
        f_samples.append(f)
    return f_samples
