# -*- coding: utf-8 -*-
"""
Created on 2025/08/13 21:48:34
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Dict

from S2Generator.incentive.base_incentive import BaseIncentive


def arma_series(
        rng: np.random.RandomState,
        time_series: np.ndarray,
        p_params: np.ndarray,
        q_params: np.ndarray,
) -> np.ndarray:
    """
    Generate an ARMA process based on the specified parameters.

    :param rng: Random number generator of NumPy with fixed seed.
    :param time_series: The zeros time series.
    :param p_params: The parameters of the AR(p) process.
    :param q_params: The parameters of the MA(q) process.
    """
    # TODO: 这里的参数控制需要进一步的调整
    for index in range(len(time_series)):
        # Get the previous p AR values
        index_p = max(0, index - len(p_params))
        p_vector = np.flip(time_series[index_p:index])
        # Compute the dot product of p values and model parameters
        p_value = np.dot(p_vector, p_params[0: len(p_vector)])
        # Generate q values through a white noise sequence
        q_value = np.dot(rng.randn(len(q_params)), q_params)
        sum_value = p_value + rng.randn(1) + q_value
        if sum_value > 1024:
            sum_value = q_value
        time_series[index] = sum_value
    return time_series


class ARMA(BaseIncentive):
    """"""

    def __init__(
            self,
            p_min: Optional[int] = 1,
            p_max: Optional[int] = 3,
            q_min: Optional[int] = 1,
            q_max: Optional[int] = 5,
            upper_bound: float = 512,
            dtype: np.dtype = np.float64,
    ) -> None:
        """
        :param upper_bound: The upper bound number of the ARMA process.
        """
        super().__init__(dtype=dtype)

        self.p_min = p_min
        self.p_max = p_max
        self.q_min = q_min
        self.q_max = q_max

        self.p_order, self.q_order = None, None

        self.p_params, self.q_params = None, None

        self.upper_bound = upper_bound

    @staticmethod
    def create_autoregressive_params(
            rng: np.random.RandomState, p_order: int
    ) -> np.ndarray:
        """
        构建自回归过程的模型参数.
        由于自回归过程会利用时间序列的历史累计信息因此很容易导致生成的激励时间序列发生数值爆炸。
        为保证自回归过程的能够稳定生成平稳时间序列，我们对自回归过程的参数进行了一定的约束。
        1. The absolute value of the last parameter (i.e., φ_p) is less than 1: |φ_p| < 1
        2. The sum of all parameters is less than 1: φ_1 + φ_2 + ... + φ_p < 1

        :param rng: The random number generator of NumPy with fixed seed.
        :param p_order: The order of the autoregressive process.
        :return: The autoregressive parameters.
        """
        # Generate the last params first
        p_last = rng.uniform(low=-0.99, high=0.99)

        # Generate other params
        p_params = np.append(rng.uniform(low=-1.0, high=1.0, size=p_order - 1), p_last)

        # 计算参数的求和
        total = np.sum(p_params)

        # 缩放参数使总和<1（同时保持|φ_p|<1）
        if total >= 1:
            scale_factor = 0.95 / (total + 0.1)  # 确保缩放后总和<1
            p_params *= scale_factor

        return p_params

    @staticmethod
    def create_moving_average_params(
            rng: np.random.RandomState, q_order: int
    ) -> np.ndarray:
        """
        构建滑动平均过程的模型参数。

        :param rng: The random number generator of NumPy with fixed seed.
        :param q_order: The order of the moving average process.
        :return: The moving average parameters.
        """
        return rng.uniform(low=-1.0, high=1.0, size=q_order)

    @property
    def order(self) -> Dict[str, int]:
        """获取ARMA模型中自回归过程和滑动平均过程的阶数"""
        return {"AR(p)": self.p_order, "MA(q)": self.q_order}

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """获取ARMA模型中自回归过程和滑动平均过程的参数"""
        return {"AR(p)": self.p_params, "MA(q)": self.q_params}

    def arma_series(
            self,
            rng: np.random.RandomState,
            time_series: np.ndarray,
            p_params: Optional[np.ndarray] = None,
            q_params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate an ARMA process based on the specified parameters.

        :param rng: Random number generator of NumPy with fixed seed.
        :param time_series: The zeros time series.
        :param p_params: The parameters of the AR(p) process.
        :param q_params: The parameters of the MA(q) process.
        """
        return arma_series(
            rng=rng,
            time_series=time_series,
            p_params=self.p_params if p_params is None else p_params,
            q_params=self.q_params if q_params is None else q_params,
        )

    def create_params(self, rng: np.random.RandomState) -> np.ndarray:
        """构建滑动平均自回归时间序列数据的参数"""
        # 首先随机生成模型的阶数
        self.p_order = rng.randint(low=self.p_min, high=self.p_max)
        self.q_order = rng.randint(low=self.q_min, high=self.q_max)

        # Generate the parameters of AR(p)
        self.p_params = self.create_autoregressive_params(rng=rng, p_order=self.p_order)

        # Generate the parameters of MA(q)
        self.q_params = self.create_moving_average_params(rng=rng, q_order=self.q_order)

    def generate(
            self,
            rng: np.random.RandomState,
            n_inputs_points: int = 512,
            input_dimension: int = 1,
    ) -> np.ndarray:
        """
        Generate ARMA stationary time series based on the specified input points and dimensions.

        :param rng: The random number generator of NumPy with fixed seed.
        :param n_inputs_points: The number of input points.
        :param input_dimension: The dimension of the time series.
        :return: The generated ARMA time series.
        """
        # 生成全零的时间序列数据
        time_series = self.create_zeros(
            n_inputs_points=n_inputs_points, input_dimension=input_dimension
        )

        # Generate clusters with numerical explosion through a while loop
        index = 0
        while index < input_dimension:
            # Generate the AMRA series
            arma = self.arma_series(rng=rng, time_series=time_series, p_params=self.p_params, q_params=self.q_params)

            # Check the upper bound
            if np.max(np.abs(arma)) <= self.upper_bound:
                time_series[:, index] = arma
                # Remove the pointer
                index += 1

        return time_series
