# -*- coding: utf-8 -*-
"""
Created on 2025/08/13 23:47:59
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
# 通过混合分布采样的方式生成激励时间序列
from .mixed_distribution import MixedDistribution

# 通过构建自回归滑动平均过程生成激励时间序列
from .autoregressive_moving_average import AutoregressiveMovingAverage

# 通过模拟时间序列数据的趋势、周期和噪声分量来生成激励时间序列
from .forecast_pfn import ForecastPFN

# 通过多种核运算的Kernel Synth的方式来生成激励时间序列
from .kernel_synth import KernelSynth

# 通过信号分解的观点构造周期性极强的本征模态函数来生成激励时间序列
from .intrinsic_mode_functions import IntrinsicModeFunction
