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

from typing import Optional, Dict, List, Tuple, Any, Callable
from S2Generator.incentive.base_incentive import BaseIncentive

# 所有可用使用的本征模特函数的字典
ALL_IMF_DICT = {
    "generate_sin_signal": generate_sin_signal,
    "generate_cos_signal": generate_cos_signal,
    "generate_am_signal": generate_am_signal,
    "generate_sawtooth_wave": generate_sawtooth_wave,
}


class IMFs(BaseIncentive):
    """用于生成本征模态函数形式的激励时间序列"""

    def __init__(self,
                 probability_dict: Dict[str, float] = None,
                 probability_list: List[float] = None,

                 dtype: np.dtype = np.float64) -> None:
        """"""
        super().__init__(dtype=dtype)

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
        return {""}

    def _processing_probability(self, probability_dict: Dict[str, float] = None, probability_list: List[float] = None) -> float:
        """处理输入的概率字典和列表"""
        if probability_dict is None and probability_list is None:
            # 如果输入均为None则使用默认参数
