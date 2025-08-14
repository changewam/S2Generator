# -*- coding: utf-8 -*-
"""
Created on 2025/08/14 11:01:12
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from scipy.stats import special_ortho_group
from pysdkit.utils import max_min_normalization

from typing import Optional, Dict, Tuple, List
from S2Generator.incentive.base_incentive import BaseIncentive


class MixedDistribution(BaseIncentive):
    """通过混合分布的方式生成激励时间序列数据"""

    def __init__(
        self,
        min_centroids: Optional[int] = 3,
        max_centroids: Optional[int] = 8,
        rotate: Optional[bool] = False,
        gaussian: Optional[bool] = True,
        uniform: Optional[bool] = True,
        probability_dict: Optional[Dict[str, float]] = None,
        probability_list: Optional[List[float]] = None,
        dtype: np.dtype = np.float64,
    ):
        super().__init__(dtype=dtype)

        # 最小和最大的混合分布数目
        self.min_centroids = min_centroids
        self.max_centroids = max_centroids

        # 是否对采样获得的时间序列数据乘以旋转矩阵
        self.rotate = rotate

        # 是否开启高斯过程和均匀分布过程的采样
        self.gaussian = gaussian
        self.uniform = uniform

        # 激励概率字典和列表
        self.probability_dict = probability_dict
        self.probability_list = probability_list

        # 获取可用的字典和列表
        self._available_dict, self._available_list, self._available_prob = (
            self._get_available(
                probability_dict=probability_dict, probability_list=probability_list
            )
        )

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        return "MixedDistribution"

    @property
    def default_probability_dict(self) -> Dict[str, float]:
        """当用户没有指定概率字典时提供数据生成的默认配置"""
        if self.gaussian is True and self.uniform is True:
            return {"gaussian": 0.5, "uniform": 0.5}
        elif self.gaussian is True and self.uniform is False:
            return {"gaussian": 1.0}
        elif self.gaussian is False and self.uniform is True:
            return {"uniform": 1.0}
        else:
            raise ValueError

    @property
    def available_dict(self) -> Dict[str, float]:
        return self._available_dict

    @property
    def available_list(self) -> List[str]:
        return self._available_list

    @property
    def available_prob(self) -> List[float]:
        return self._available_prob

    def _get_available(
        self,
        probability_dict: Optional[Dict[str, float]] = None,
        probability_list: Optional[list[float]] = None,
    ) -> Tuple[Dict[str, float], List[str], List[float]]:
        """处理用户提供的概率列表和概率字典"""
        if probability_dict is None and probability_list is None:
            # 当字典和列表都没有提供时返回默认配置
            available_dict = self.default_probability_dict

        elif probability_dict is not None:
            # 当用户提供了概率字典时
            if "gaussian" not in probability_dict and "uniform" not in probability_dict:
                # 没有指定的键值则返回默认配置
                available_dict = self.default_probability_dict
            else:
                # 对字典中的内容进行归一化处理
                probability_array = max_min_normalization(
                    np.array(
                        [probability_dict["gaussian"], probability_dict["uniform"]]
                    )
                )
                available_dict = {
                    "gaussian": probability_array[0],
                    "uniform": probability_array[1],
                }

        elif probability_list is not None and probability_dict is None:
            # 当用户提供了概率列表时
            probability_array = max_min_normalization(np.array([probability_list]))
            available_dict = {
                "gaussian": probability_array[0],
                "uniform": probability_array[1],
            }

        else:
            raise ValueError("Something wrong in probability_dict or probability_list!")

        # 获取字典中的内容
        available_list = list(available_dict.keys())
        available_prob = list(available_dict.values())

        return available_dict, available_list, available_prob

    def generate_stats(
        self, rng: np.random.RandomState, input_dimension: int, n_centroids: int
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
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
        rng: np.random.RandomState,
        input_dimension: int,
        n_centroids: int,
        n_points_comp: np.ndarray,
    ) -> np.ndarray:
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
        rng: np.random.RandomState,
        input_dimension: int,
        n_centroids: int,
        n_points_comp: np.ndarray,
    ) -> np.ndarray:
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

    def generate_once(
        self, rng: np.random.RandomState, n_inputs_points: int = 512
    ) -> np.ndarray | None:
        """
        通过混合分布生成单个通道的激励时间序列数据。

        :param rng: The random state generator in NumPy.
        :param n_inputs_points: The number of input points in this sampling.
        :return: The generated time series samples with mixture distribution.
        """
        # 1. Statistical parameters for mixture distribution sampling
        n_centroids = rng.randint(low=self.min_centroids, high=self.max_centroids)

        # 2. Randomly generate the weight values for each distribution
        weights = rng.uniform(0, 1, size=(n_centroids,))
        weights /= np.sum(weights)
        n_points_comp = rng.multinomial(n_inputs_points, weights)

        # 3. 决定使用那种分布进行采样
        dist_list = rng.choice(
            self.available_list, size=n_centroids, p=self.available_prob
        )

        # 4. 遍历混合分布的列表来生成时间序列数据
        for sampling_type in dist_list:
            if sampling_type == "gaussian":
                # Sample using a Gaussian mixture distribution
                return self.generate_gaussian(
                    rng=rng,
                    input_dimension=1,
                    n_centroids=n_centroids,
                    n_points_comp=n_points_comp,
                )
            elif sampling_type == "uniform":
                # Sample using a uniform mixture distribution
                return self.generate_uniform(
                    rng=rng,
                    input_dimension=1,
                    n_centroids=n_centroids,
                    n_points_comp=n_points_comp,
                )
            else:
                raise ValueError("Something wrong in sampling_type!")
        return None

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int = 512,
        input_dimension: int = 1,
    ) -> np.ndarray:
        """
        Generate time series of specified dimensions and lengths using a uniform or gaussian mixture distribution.

        :param rng: The random number generator of NumPy with fixed seed.
        :param n_inputs_points: The number of input points.
        :param input_dimension: The dimension of the time series.
        :return: The generated mixed distribution time series.
        """
        # 遍历多个通道来生成时间序列数据
        time_series = np.hstack(
            [
                self.generate_once(rng=rng, n_inputs_points=n_inputs_points)
                for _ in range(input_dimension)
            ]
        )

        return time_series


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    mixed_distribution = MixedDistribution()

    time = mixed_distribution.generate(
        rng=np.random.RandomState(100), n_inputs_points=512, input_dimension=5
    )

    for i in range(5):
        plt.plot(time[:, i])
        plt.show()
