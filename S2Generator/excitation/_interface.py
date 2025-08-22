# -*- coding: utf-8 -*-
"""
This module is used to build a unified interface for generating time series using various different incentives.
It also manages the allocation of specific parameters for various data generation mechanisms.

Created on 2025/08/18 23:31:37
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import numpy as np

from S2Generator.params import SeriesParams
from S2Generator.excitation import (
    MixedDistribution,
    AutoregressiveMovingAverage,
    ForecastPFN,
    KernelSynth,
    IntrinsicModeFunction,
)


class Excitation(object):

    def __init__(self, series_params: SeriesParams) -> None:
        # 实例化
        self.series_params = series_params

    def generate(
        self,
    ) -> np.ndarray:
        pass


if __name__ == "__main__":
    md = MixedDistribution()
    print(md)
