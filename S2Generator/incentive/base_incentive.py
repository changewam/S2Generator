# -*- coding: utf-8 -*-
"""
Created on 2025/08/11 09:34:54
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional
from abc import ABC, abstractmethod


class BaseIncentive(ABC):
    """用于生成激励时间序列数据的基类"""

    def __init__(self, dtype: np.dtype):
        self.dtype = dtype

    @abstractmethod
    def generate(self, rng: np.random.RandomState, ) -> np.ndarray: