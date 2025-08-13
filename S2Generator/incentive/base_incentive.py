# -*- coding: utf-8 -*-
"""
Created on 2025/08/11 09:34:54
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseIncentive(ABC):
    """用于生成激励时间序列数据的基类"""

    def __init__(self, dtype: np.dtype):
        self.data_type = dtype

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def dtype(self) -> np.dtype:
        return self.data_type

    @abstractmethod
    def generate(
        self, rng: np.random.RandomState, n_inputs_points: int = 512, input_dimension=1
    ) -> np.ndarray:
        """生成激励时间序列数据"""
