# -*- coding: utf-8 -*-
"""
Created on 2025/01/23 17:46:11
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
from typing import Optional


class Params(object):
    """Parameter Control in (Series-Symbol) S2 Data Generation"""

    def __init__(
        self,
        min_input_dimension: Optional[int] = 1,
        max_input_dimension: Optional[int] = 6,
        min_output_dimension: Optional[int] = 1,
        max_output_dimension: Optional[int] = 12,
        n_points: Optional[int] = 256,
        max_trials: Optional[int] = 128,
        max_int: Optional[int] = 10,
        prob_const: Optional[float] = 0.0,
        prob_rand: Optional[float] = 0.0,
        min_binary_ops_per_dim: Optional[int] = 0,
        max_binary_ops_per_dim: Optional[int] = 1,
        max_binary_ops_offset: Optional[int] = 4,
        min_unary_ops: Optional[int] = 0,
        max_unary_ops: Optional[int] = 5,
        float_precision: Optional[int] = 3,
        mantissa_len: Optional[int] = 1,
        max_exponent: Optional[int] = 3,
        max_exponent_prefactor: Optional[int] = 1,
        use_abs: Optional[bool] = True,
        operators_to_downsample: Optional[str] = None,
        max_unary_depth: Optional[int] = 6,
        required_operators: Optional[str] = "",
        extra_unary_operators: Optional[str] = "",
        extra_binary_operators: Optional[str] = "",
        extra_constants: Optional[str] = "",
        use_sympy: Optional[bool] = False,
        reduce_num_constants: Optional[bool] = True,
        decimals: Optional[int] = 6,
        max_centroids: Optional[int] = 10,
        max_kernels: Optional[int] = 7,
        p_min: Optional[int] = 1,
        p_max: Optional[int] = 3,
        q_min: Optional[int] = 1,
        q_max: Optional[int] = 5,
        rotate: Optional[bool] = False,
        gaussian: Optional[bool] = True,
        uniform: Optional[bool] = True,
        arma: Optional[bool] = True,
        kernel_synth: Optional[bool] = True,
    ) -> None:
        """
        Specific parameter control of data generation
        :param min_input_dimension: Minimum input dimension (minimum number of variables) for symbolic expressions
        :param max_input_dimension: Maximum input dimension of symbolic expressions (maximum number of variables)
        :param min_output_dimension: Minimum output dimension of multivariate symbolic expressions
        :param max_output_dimension: Maximum output dimension of multivariate symbolic expressions
        :param n_points: Construct sampling series length
        :param max_int: Maximal integer in symbolic expressions
        :param prob_const: Probability to generate integer in leafs
        :param prob_rand: Probability to generate n in leafs
        :param max_trials: How many trials we have for a given function
        :param min_binary_ops_per_dim: Min number of binary operators per input dimension
        :param max_binary_ops_per_dim: Max number of binary operators per input dimension
        :param max_binary_ops_offset: Offset for max number of binary operators
        :param min_unary_ops: Min number of unary operators
        :param max_unary_ops: Max number of unary operators
        :param float_precision: Number of digits in the mantissa
        :param mantissa_len: Number of tokens for the mantissa (must be a divisor or float_precision+1)
        :param max_exponent: Maximal order of magnitude
        :param max_exponent_prefactor: Maximal order of magnitude in prefactors
        :param use_abs: Whether to replace log and sqrt by log(abs) and sqrt(abs)
        :param operators_to_downsample: Which operator to remove
        :param max_unary_depth: Max number of operators inside unary
        :param required_operators: Which operator to remove
        :param extra_unary_operators: Extra unary operator to add to data generation
        :param extra_binary_operators: Extra binary operator to add to data generation
        :param extra_constants: Additional int constants floats instead of ints
        :param use_sympy: Whether to use sympy parsing (basic simplification)
        :param reduce_num_constants: Use minimal amount of constants in eqs
        :param decimals: Number of digits reserved for floating-point numbers in symbolic expressions
        :param max_centroids: Max number of centroids for the input distribution
        :param max_kernels: Max number of kernels for the input distribution in KernelSynth methods
        :param p_min: Minimal order for AR(p) in ARMA(p, q)
        :param p_max: Maximal order in ARMA(p, q)
        :param q_min: Minimal order in ARMA(q, p)
        :param q_max: Maximal order in ARMA(q, p)
        :param rotate: Whether to use the selection vector to increase the diversity of the sampling series
        :param gaussian: Whether to use Gaussian mixture distribution for series sampling
        :param uniform: Whether to use uniform distribution for series sampling
        :param arma: Whether to use the ARMA model for series sampling
        :param kernel_synth: Whether to use kernel synthesis for series sampling
        """
        self.min_input_dimension, self.max_input_dimension = (
            min_input_dimension,
            max_input_dimension,
        )
        self.min_output_dimension, self.max_output_dimension = (
            min_output_dimension,
            max_output_dimension,
        )
        self.n_points = n_points
        self.max_trials = max_trials
        self.max_int = max_int
        self.prob_const, self.prob_rand = prob_const, prob_rand
        self.min_binary_ops_per_dim, self.max_binary_ops_per_dim = (
            min_binary_ops_per_dim,
            max_binary_ops_per_dim,
        )
        self.max_binary_ops_offset = max_binary_ops_offset
        self.min_unary_ops, self.max_unary_ops = min_unary_ops, max_unary_ops
        self.float_precision = float_precision
        self.mantissa_len = mantissa_len
        self.max_exponent = max_exponent
        self.max_exponent_prefactor = max_exponent_prefactor
        self.use_abs = use_abs
        self.operators_to_downsample = operators_to_downsample
        self.max_unary_depth = max_unary_depth
        self.required_operators = required_operators
        self.decimals = decimals
        self.extra_unary_operators, self.extra_binary_operators = (
            extra_unary_operators,
            extra_binary_operators,
        )
        self.extra_constants = extra_constants
        self.use_sympy = use_sympy
        self.reduce_num_constants = reduce_num_constants
        self.max_centroids = max_centroids
        self.max_kernels = max_kernels
        self.p_min, self.p_max, self.q_min, self.q_max = p_min, p_max, q_min, q_max
        self.rotate = rotate
        self.gaussian, self.uniform, self.arma = gaussian, uniform, arma

        # Add various sampling methods
        self.sampling_type = []
        if gaussian is True:
            self.sampling_type.append("gaussian")
        if uniform is True:
            self.sampling_type.append("uniform")
        if arma is True:
            self.sampling_type.append("ARMA")
        if kernel_synth is True:
            self.sampling_type.append("kernel_synth")
        if len(self.sampling_type) == 0:
            raise ValueError(
                "sampling_type is empty! please specify one sampling_type in (gaussian, uniform, ARMA) at least"
            )
        self.operators_to_downsample = (
            "div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3"
            if operators_to_downsample is None
            else operators_to_downsample
        )
