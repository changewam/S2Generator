# -*- coding: utf-8 -*-
"""
Created on 2025/08/12 13:40:16
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from pysdkit.data import (
    add_noise,
    generate_sin_signal,
    generate_cos_signal,
    generate_am_signal,
    generate_sawtooth_wave,
)
from pysdkit.utils import max_min_normalization

from typing import Optional, Dict, List, Tuple, Any, Callable
from S2Generator.incentive.base_incentive import BaseIncentive

# 所有可用使用的本征模特函数的字典
ALL_IMF_DICT = {
    "generate_sin_signal": generate_sin_signal,
    "generate_cos_signal": generate_cos_signal,
    "generate_am_signal": generate_am_signal,
    "generate_sawtooth_wave": generate_sawtooth_wave,
}


def _check_probability_dict(prob_dict: Dict[str, float]) -> Dict[str, float]:
    """
    检查输入的概率字典是否符合基本要求并对概率进行归一化

    :param prob_dict: 输入的包含各种本征模态函数的概率字典。
    :return:
    """
    prob_list = []

    # 首先检查概率字典的输入键值是否符合要求
    for key, value in prob_dict.items():
        # 检查键值
        if key not in ALL_IMF_DICT.keys():
            raise ValueError(f"Illegal key: {key} in `prob_dict`!")

        # 记录概率的大小
        prob_list.append(value)

    # 当格式检测没问题时对概率值进行归一化操作
    prob_array = max_min_normalization(x=np.array(prob_list))

    return {key: value for key, value in zip(prob_dict.keys(), prob_array)}


def _check_probability_list(prob_list: List[float]) -> Dict[str, float]:
    """
    检查输入的概率列表是否符合基本要求并对概率进行归一化

    :param prob_list:
    :return:
    """
    length = len(prob_list)

    # 进行列表的长度检测
    if length > len(ALL_IMF_DICT) or length <= 0:
        raise ValueError("Error length of `prob_list`")

    prob_array = max_min_normalization(x=np.array(prob_list))

    return {key: value for key, value in zip(ALL_IMF_DICT.keys()[:length], prob_array)}


def _get_energy(signal: np.ndarray) -> float:
    """
    获取信号能量的大小.
    能量的计算方式有很多，这里简单进行绝对值的平均。

    :param signal:
    :return:
    """
    return np.mean(np.abs(signal))


def get_adaptive_sampling_rate(duration: float, length: int) -> float:
    """
    根据所需信号的长度与时间周期自适应获得采样频率。

    :param duration: Length of the signal in seconds.
    :param length: Length of generated signal.
    :return: The adaptive sampling rate.
    """
    return np.ceil(length / duration)


class IMFs(BaseIncentive):
    """用于生成本征模态函数形式的激励时间序列"""

    def __init__(
            self,
            min_base_imfs: int = 1,
            max_base_imfs: int = 3,
            min_choice_imfs: int = 2,
            max_choice_imfs: int = 5,
            probability_dict: Optional[Dict[str, float]] = None,
            probability_list: Optional[List[float]] = None,
            min_duration: float = 0.5,
            max_duration: float = 10.0,
            min_amplitude: float = 0.01,
            max_amplitude: float = 10.0,
            min_frequency: float = 0.01,
            max_frequency: float = 8.0,
            noise_level: float = 0.01,
            dtype: np.dtype = np.float64,
    ) -> None:
        """"""
        super().__init__(dtype=dtype)

        # 记录本征模态函数基底的数目
        self.min_base_imfs = min_base_imfs
        self.max_base_imfs = max_base_imfs

        # 基函数的形式
        self.base_imfs = [generate_sin_signal, generate_cos_signal]

        # 记录通过采样选择的不同本征模态函数数目
        self.min_choice_imfs = min_choice_imfs
        self.max_choice_imfs = max_choice_imfs

        # 获得可用的本征模态函数概率字典
        self.available_dict, self.available_list, self.available_probability = (
            self._processing_probability(
                probability_dict=probability_dict, probability_list=probability_list
            )
        )

        # 时间戳长度
        self.min_duration = min_duration
        self.max_duration = max_duration

        # 获取最小和最大的振幅范围
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

        # 获取最小和最大的频率范围
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

        # 添加噪声的大小
        self.noise_level = noise_level

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def all_imfs_dict(self) -> Dict[str, Callable]:
        """获取所有可以使用的本征模特函数的字典"""
        return ALL_IMF_DICT

    @property
    def all_imfs_list(self) -> List[Callable]:
        """获取所有可以使用的本征模态函数的列表"""
        return list(self.all_imfs_dict.values())

    @property
    def default_probability_dict(self) -> Dict[str, float]:
        """当用户为输入指定参数时获得默认的概率字典"""
        return {
            "generate_sin_signal": 0.30,
            "generate_cos_signal": 0.30,
            "generate_am_signal": 0.20,
            "generate_sawtooth_wave": 0.20,
        }

    def _processing_probability(
            self,
            probability_dict: Dict[str, float] = None,
            probability_list: List[float] = None,
    ) -> Tuple[Dict[str, float], List[str], List[float]]:
        """
        处理输入的概率字典和列表.

        :param probability_dict:
        :param probability_list:
        :return:
        """

        if probability_dict is None and probability_list is None:
            # 如果输入均为None则使用默认参数
            available_dict = self.default_probability_dict

        elif probability_dict is not None and probability_list is None:
            # 如果输入的字典为非None，而输入的列表为None

            available_dict = _check_probability_dict(prob_dict=probability_dict)

        elif probability_dict is None and probability_list is not None:
            # 如果输入的字典为None，而输入的列表为None
            available_dict = _check_probability_list(prob_list=probability_list)

        elif probability_dict is not None and probability_list is not None:
            # 如果均有输入则以字典为主
            available_dict = _check_probability_dict(prob_dict=probability_dict)

        else:
            raise ValueError(
                "`probability_dict` and `probability_list` must be specified!"
            )

        # 从字典中进一步获取字典和概率
        available_list = list(available_dict.keys())
        available_probability = list(available_dict.values())

        return available_dict, available_list, available_probability

    def _add_noise(self, imfs: np.ndarray, n_inputs_points: int) -> np.ndarray:
        """
        为构建的本征模态函数添加一定`噪声的函数`.
        其中噪声数值的大小在一定程度上取决于构建本征模态函数的`能量`大小。

        :param imfs: 构建的本征模态函数形式的激励时间序列。
        :param n_inputs_points: 采样序列的时间戳数目。
        :return: The noise time series.
        """
        return add_noise(
            N=n_inputs_points, Mean=0, STD=self.noise_level * _get_energy(signal=imfs)
        )

    def get_random_duration(self, rng: np.random.RandomState, number) -> np.ndarray:
        """
        随机获取每个本征模态函数中的时间范围。

        :param rng: The random state generator in NumPy.
        :param number: 本次采样中所有基函数的数目。
        :return: NumPy ndarray, 存放随机时间范围的数组。
        """
        return rng.uniform(low=self.min_duration, high=self.max_duration, size=number)

    def get_random_amplitude(
            self, rng: np.random.RandomState, number: int
    ) -> np.ndarray:
        """
        随机获取每个本征模态函数中的振幅大小。

        :param rng: The random state generator in NumPy.
        :param number: 本次采样中所有基函数的数目。
        :return: NumPy ndarray, 存放随机振幅的数组。
        """
        return rng.uniform(low=self.min_amplitude, high=self.max_amplitude, size=number)

    def get_random_frequency(self, rng: np.random.RandomState, number: int) -> np.ndarray:
        """
        随机获取每个本征模态函数中的周期大小。

        :param rng: The random state generator in NumPy.
        :param number: 本次采样中所有基函数的数目。
        :return: NumPy ndarray, 存放随机周期的数组。
        """
        return rng.uniform(low=self.min_frequency, high=self.max_frequency, size=number)

    def get_base_imfs(
            self, imfs: np.ndarray, rng: np.random.RandomState, n_inputs_points: int
    ) -> np.ndarray:

        # 获得基函数的数目
        base_number = rng.randint(low=self.min_base_imfs, high=self.max_base_imfs + 1)

        # 随机获取基函数的振幅和周期
        for (base_function, amplitude, frequency, duration) in zip(
                rng.choice(self.base_imfs, size=base_number, p=np.array([0.5, 0.5])),
                self.get_random_amplitude(rng=rng, number=base_number),
                self.get_random_frequency(rng=rng, number=base_number),
                self.get_random_duration(rng=rng, number=base_number),
        ):
            # 逐步添加基函数
            imfs += amplitude * base_function(duration=duration,
                                              sampling_rate=get_adaptive_sampling_rate(duration=duration,
                                                                                       length=n_inputs_points),
                                              frequency=frequency,
                                              noise_level=0.0)[1][: n_inputs_points]

        return imfs

    def get_choice_imfs(
            self, imfs: np.ndarray, rng: np.random.RandomState, n_inputs_points: int
    ) -> np.ndarray:

        # 获得随机本征模态函数的数目
        choice_number = rng.randint(low=self.min_choice_imfs, high=self.max_choice_imfs)

        # 添加其他的随机本征模态函数
        for (choice_function, amplitude, frequency, duration) in zip(
                rng.choice(self.available_list, size=choice_number, p=self.available_probability),
                self.get_random_amplitude(rng=rng, number=choice_number),
                self.get_random_frequency(rng=rng, number=choice_number),
                self.get_random_duration(rng=rng, number=choice_number),
        ):
            choice_function = ALL_IMF_DICT[choice_function]
            if choice_function == generate_am_signal:
                imfs += amplitude * generate_am_signal(duration=duration,
                                                       sampling_rate=get_adaptive_sampling_rate(duration=duration,
                                                                                                length=n_inputs_points),
                                                       mod_index=rng.randint(1, 4),
                                                       carrier_freq=rng.randint(low=50, high=150),
                                                       modulating_freq=rng.randint(low=1, high=16),
                                                       noise_level=0.0)[1][: n_inputs_points]
            else:
                print(1)
                imfs += amplitude * choice_function(duration=duration,
                                                    sampling_rate=get_adaptive_sampling_rate(duration=duration,
                                                                                             length=n_inputs_points),
                                                    frequency=frequency,
                                                    noise_level=0.0)[1][: n_inputs_points]

        return imfs

    def generate(
            self, rng: np.random.RandomState, n_inputs_points: int = 512, input_dimension=1
    ) -> np.ndarray:
        """
        通过多种本征模态函数组合的形式获得生数据
        """
        # 初始化一个全零的本征模态函数
        imfs = np.zeros(shape=(n_inputs_points, input_dimension), dtype=self.dtype)

        for i in range(input_dimension):
            # 1. 添加基函数
            imfs[:, i] = self.get_base_imfs(imfs=imfs[:, i], rng=rng, n_inputs_points=n_inputs_points)

            # 2. 添加随机选择的其他基函数
            imfs[:, i] = self.get_choice_imfs(imfs=imfs[:, i], rng=rng, n_inputs_points=n_inputs_points)

            # 3. 为构建的本征模态函数添加随机的噪声
            imfs[:, i] += self._add_noise(imfs=imfs[:, i], n_inputs_points=n_inputs_points)

        return imfs


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    rng = np.random.RandomState(10)
    imfs_generator = IMFs()

    plt.plot(imfs_generator.generate(rng, n_inputs_points=512, input_dimension=1))
    plt.show()
