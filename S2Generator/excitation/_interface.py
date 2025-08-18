# -*- coding: utf-8 -*-
"""
该模块用于构建多种不同激励时间序列生成的统一接口。
并为多种不同的数据生成机制分配其特定的参数进行管理。

Created on 2025/08/18 23:31:37"
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import numpy as np


from S2Generator.excitation import MixedDistribution, AutoregressiveMovingAverage, ForecastPFN, KernelSynth, IntrinsicModeFunction

class Excitation(object):

    def __init__(self, series_params) -> None:
        pass


if __name__ == '__main__':
    md = MixedDistribution()
    print(md)
