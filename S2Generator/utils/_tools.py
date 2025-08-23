# -*- coding: utf-8 -*-
"""
Created on 2025/08/23 17:09:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import numpy as np


def z_score_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform Z-score normalization on the input time series.

    :param x: Input two-dimensional time series with [n_points, n_dims] in NumPy.
    :return: Normalized time series with origin shape.
    """
    return (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)


def max_min_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform min-max normalization on the input time series.

    :param x: Input two-dimensional time series with [n_points, n_dims] in NumPy.
    :return: Normalized time series with origin shape.
    """
    return (x - np.min(x, axis=0, keepdims=True)) / (
        np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True)
    )
