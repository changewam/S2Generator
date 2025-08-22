# -*- coding: utf-8 -*-
"""
Created on 2025/08/08 11:38:24
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import functools

import numpy as np
from numpy import ndarray
from numpy.random import RandomState
from scipy.stats import special_ortho_group
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)

from typing import Tuple, List, Optional

from S2Generator.old_params import Params


class Incentives(object):
    """"""

    def __init__(self, params: Params):
        self.params = params

        # Model order when generating ARMA sequences
        self.p_min, self.p_max = params.p_min, params.p_max
        self.q_min, self.q_max = params.q_min, params.q_max

        #
        self.max_kernels = params.max_kernels

        self.rotate = params.rotate

    def generate_stats(
        self, rng: RandomState, input_dimension: int, n_centroids: int
    ) -> Tuple[ndarray, ndarray, List[ndarray]]:
        """Generate parameters required for sampling from a mixture distribution"""
        means = rng.randn(
            n_centroids, input_dimension
        )  # Means of the mixture distribution
        covariances = rng.uniform(
            0, 1, size=(n_centroids, input_dimension)
        )  # Variances of the mixture distribution
        if self.rotate:
            rotations = [
                (
                    special_ortho_group.rvs(input_dimension)
                    if input_dimension > 1
                    else np.identity(1)
                )
                for i in range(n_centroids)
            ]
        else:
            rotations = [np.identity(input_dimension) for i in range(n_centroids)]
        return means, covariances, rotations

    def generate_gaussian(
        self,
        rng: RandomState,
        input_dimension: int,
        n_centroids: int,
        n_points_comp: ndarray,
    ) -> ndarray:
        """Generate sequences of specified dimensions and lengths using a Gaussian mixture distribution"""
        means, covariances, rotations = self.generate_stats(
            rng, input_dimension, n_centroids
        )
        return np.vstack(
            [
                rng.multivariate_normal(mean, np.diag(covariance), int(sample))
                @ rotation
                for (mean, covariance, rotation, sample) in zip(
                    means, covariances, rotations, n_points_comp
                )
            ]
        )

    def generate_uniform(
        self,
        rng: RandomState,
        input_dimension: int,
        n_centroids: int,
        n_points_comp: ndarray,
    ) -> ndarray:
        """Generate sequences of specified dimensions and lengths using a uniform mixture distribution"""
        means, covariances, rotations = self.generate_stats(
            rng, input_dimension, n_centroids
        )
        return np.vstack(
            [
                (
                    mean
                    + rng.uniform(-1, 1, size=(sample, input_dimension))
                    * np.sqrt(covariance)
                )
                @ rotation
                for (mean, covariance, rotation, sample) in zip(
                    means, covariances, rotations, n_points_comp
                )
            ]
        )

    def generate_ARMA(
        self, rng: RandomState, n_inputs_points: int, input_dimension: int = 1
    ) -> ndarray:
        """Generate ARMA stationary time series based on the specified input points and dimensions"""
        x = np.zeros(shape=(n_inputs_points, input_dimension))

        # Generate clusters with numerical explosion through a while loop
        d = 0
        while d < input_dimension:
            # Get the number of clusters k through a uniform distribution
            p = rng.randint(low=self.p_min, high=self.p_max)
            q = rng.randint(low=self.q_min, high=self.q_max)

            # Generate AR(p) parameters
            p_last = rng.uniform(-1, 1)
            p_former = rng.uniform(-1, 1, p - 1)
            P = np.append(p_former / np.sum(p_former) * (1 - p_last), p_last)

            # Generate MA(q) parameters
            Q = rng.uniform(-1, 1, q)

            output = arma(rng=rng, ts=x[:, d], P=P, Q=Q)
            if np.max(np.abs(output)) <= 256:
                x[:, d] = output
                d += 1
        return x

    def generate_KernelSynth(
        self, rng: RandomState, n_inputs_points: int, input_dimension: int = 1
    ) -> ndarray:
        """
        Generate a time series from KernelSynth, which comes from Chronos.
        https://github.com/amazon-science/chronos-forecasting.
        Ansari A F, Stella L, Turkmen C, et al. Chronos: Learning the language of time series[J].
        arXiv preprint arXiv:2403.07815, 2024.
        https://arxiv.org/abs/2403.07815.

        :param rng: Random number generator in NumPy.
        :param n_inputs_points: Number of input points, the length of the time series.
        :param input_dimension: Dimension of the time series.
        :return: The excitation time series generated by KernelSynth methods.
        """
        return np.vstack(
            [
                generate_KernelSynth(
                    rng=rng, max_kernels=self.max_kernels, length=n_inputs_points
                )
                for _ in range(input_dimension)
            ]
        ).T

    def generate_forecast_pfn(
        self, rng: RandomState, n_inputs_points: int, input_dimension: int = 1
    ) -> ndarray:
        """
        Generate a time series from forecast, which comes from ForecastPFN.

        :param rng: Random number generator in NumPy.
        :param n_inputs_points: Number of input points, the length of the time series.
        :param input_dimension: Dimension of the time series.
        :return: The excitation time series generated by ForecastPFN methods.
        """


def arma(rng, ts: ndarray, P: ndarray, Q: ndarray) -> ndarray:
    """Generate an ARMA process based on the specified parameters"""
    # TODO: 这里的参数控制需要进一步的调整
    for index in range(len(ts)):
        # Get the previous p AR values
        index_p = max(0, index - len(P))
        p_vector = np.flip(ts[index_p:index])
        # Compute the dot product of p values and model parameters
        p_value = np.dot(p_vector, P[0 : len(p_vector)])
        # Generate q values through a white noise sequence
        q_value = np.dot(rng.randn(len(Q)), Q)
        sum_value = p_value + rng.randn(1) + q_value
        if sum_value > 1024:
            sum_value = q_value
        ts[index] = sum_value
    return ts


def get_kernel_bank(length: Optional[int] = 256):
    """Get all kernel in the bank list"""
    kernel_bank = [
        ExpSineSquared(periodicity=24 / length),  # H
        ExpSineSquared(periodicity=48 / length),  # 0.5H
        ExpSineSquared(periodicity=96 / length),  # 0.25H
        ExpSineSquared(periodicity=24 * 7 / length),  # H
        ExpSineSquared(periodicity=48 * 7 / length),  # 0.5H
        ExpSineSquared(periodicity=96 * 7 / length),  # 0.25H
        ExpSineSquared(periodicity=7 / length),  # D
        ExpSineSquared(periodicity=14 / length),  # 0.5D
        ExpSineSquared(periodicity=30 / length),  # D
        ExpSineSquared(periodicity=60 / length),  # 0.5D
        ExpSineSquared(periodicity=365 / length),  # D
        ExpSineSquared(periodicity=365 * 2 / length),  # 0.5D
        ExpSineSquared(periodicity=4 / length),  # W
        ExpSineSquared(periodicity=26 / length),  # W
        ExpSineSquared(periodicity=52 / length),  # W
        ExpSineSquared(periodicity=4 / length),  # M
        ExpSineSquared(periodicity=6 / length),  # M
        ExpSineSquared(periodicity=12 / length),  # M
        ExpSineSquared(periodicity=4 / length),  # Q
        ExpSineSquared(periodicity=4 * 10 / length),  # Q
        ExpSineSquared(periodicity=10 / length),  # Y
        DotProduct(sigma_0=0.0),
        DotProduct(sigma_0=1.0),
        DotProduct(sigma_0=10.0),
        RBF(length_scale=0.1),
        RBF(length_scale=1.0),
        RBF(length_scale=10.0),
        RationalQuadratic(alpha=0.1),
        RationalQuadratic(alpha=1.0),
        RationalQuadratic(alpha=10.0),
        WhiteKernel(noise_level=0.1),
        WhiteKernel(noise_level=1.0),
        ConstantKernel(),
    ]
    return kernel_bank


def random_binary_map(a: Kernel, b: Kernel) -> ndarray:
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.

    :param a: A GP kernel
    :param b: A GP kernel
    :return: The composite kernel `a + b` or `a * b`.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)


def sample_from_gp_prior(
    kernel: Kernel, X: ndarray, random_seed: Optional[int] = None
) -> ndarray:
    """
    Draw a sample from a GP prior.

    :param kernel: The GP covaraince kernel
    :param X: The input "time" points
    :param random_seed: The random seed for sampling, by default None
    :return: A time series sampled from the GP prior
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2
    gpr = GaussianProcessRegressor(kernel=kernel)
    ts = gpr.sample_y(X, n_samples=1, random_state=random_seed)

    return ts


def sample_from_gp_prior_efficient(
    kernel: Kernel,
    X: ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
) -> ndarray:
    """
    Draw a sample from a GP prior. An efficient version that allows specification
    of the sampling method. The default sampling method used in GaussianProcessRegressor
    is based on SVD which is significantly slower that alternatives such as `eigh` and
    `cholesky`.

    :param kernel: The GP covaraince kernel
    :param X: The input "time" points
    :param random_seed: The random seed for sampling, by default None
    :param method: The sampling method for multivariate_normal, by default `eigh`
    :return: A time series sampled from the GP prior
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2

    cov = kernel(X)
    ts = np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=cov, method=method
    )

    return ts


def generate_KernelSynth(
    rng: RandomState, max_kernels: Optional[int] = 5, length: Optional[int] = 256
) -> ndarray:
    """
    Generate a synthetic time series from KernelSynth.

    :param rng: Random Number Generator
    :param max_kernels: The maximum number of base kernels to use for each time series, by default 5
    :param length: The length of the time series, by default 256
    :return: A time series generated by KernelSynth
    """
    while True:
        X = np.linspace(0, 1, length)

        # Randomly select upto max_kernels kernels from the KERNEL_BANK
        selected_kernels = rng.choice(
            get_kernel_bank(length), rng.randint(1, max_kernels + 1), replace=True
        )

        # Combine the sampled kernels using random binary operators
        kernel = functools.reduce(random_binary_map, selected_kernels)

        # Sample a time series from the GP prior
        try:
            ts = sample_from_gp_prior(kernel=kernel, X=X)
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue

        # The timestamp is arbitrary
        return ts.squeeze()
