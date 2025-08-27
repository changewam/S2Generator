# -*- coding: utf-8 -*-
"""
Created on 2025/08/3 00:01:56
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
__all__ = [
    "symbol_to_markdown",
    "z_score_normalization",
    "max_min_normalization",
    "wasserstein_distance",
    "wasserstein_distance_matrix",
    "plot_wasserstein_heatmap",
]

# # Visualization the time series data in S2
# from .visualization import plot_series
#
# # Visualization the Symbol data in S2
# from .visualization import plot_symbol

# Transform the symbol from string to latex
from .print_symbol import symbol_to_markdown

# The z-score standardization
from ._tools import z_score_normalization

# The min-max normalization
from ._tools import max_min_normalization

# The Wasserstein distance used to measure the similarity between two datasets
from ._wasserstein_distance import wasserstein_distance

# Calculate the distance matrix between multiple time series data sets using the Wasserstein distance formula
from ._wasserstein_distance import wasserstein_distance_matrix

# Visualization the Wasserstein distance though the heatmap
from ._wasserstein_distance import plot_wasserstein_heatmap
