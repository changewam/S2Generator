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

from typing import Optional, List, Tuple, Dict, Any

from S2Generator.params import SeriesParams
from S2Generator.excitation import (
    MixedDistribution,
    AutoregressiveMovingAverage,
    ForecastPFN,
    KernelSynth,
    IntrinsicModeFunction,
)
from S2Generator.utils import z_score_normalization, max_min_normalization


class Excitation(object):

    def __init__(self, series_params: Optional[SeriesParams] = None) -> None:
        """ """
        # Select hyperparameters set by user input
        self._series_params = (
            series_params if series_params is not None else SeriesParams()
        )

        # Create an algorithm instance object based on the hyperparameters entered by the user
        self._sampling_dict = self.create_sampling_dict(
            series_params=self._series_params
        )

    def create_sampling_dict(
        self, series_params: Optional[SeriesParams] = None
    ) -> Dict[
        str,
        MixedDistribution
        | AutoregressiveMovingAverage
        | ForecastPFN
        | KernelSynth
        | IntrinsicModeFunction,
    ]:
        series_params = self.series_params if series_params is None else series_params

        sampling_dict = {
            name: method
            for name, method in zip(
                self.sampling_methods,
                [
                    self.create_mixed_distribution(series_params=series_params),
                    self.create_autoregressive_moving_average(
                        series_params=series_params
                    ),
                    self.create_forecast_pfn(series_params=series_params),
                    self.create_kernel_synth(series_params=series_params),
                    self.create_intrinsic_mode_function(series_params=series_params),
                ],
            )
        }

        return sampling_dict

    def choice(
        self, rng: np.random.RandomState, input_dimension: int = 1
    ) -> np.ndarray:
        return rng.choice(
            self.sampling_methods, size=input_dimension, p=self.prob_array
        )

    def create_mixed_distribution(
        self, series_params: Optional[SeriesParams] = None
    ) -> MixedDistribution:
        """
        Constructing excitation time series generation for sampling from mixture distributions.

        :param series_params: The parameters for generating management incentive time series data.
        :return:
        """
        series_params = self.series_params if series_params is None else series_params
        return MixedDistribution(
            min_centroids=series_params.min_centroids,
            max_centroids=series_params.max_centroids,
            rotate=series_params.rotate,
            gaussian=series_params.gaussian,
            uniform=series_params.uniform,
            probability_dict=series_params.mixed_distribution_dict,
            probability_list=series_params.mixed_distribution_list,
            dtype=self.dtype,
        )

    def create_autoregressive_moving_average(
        self, series_params: Optional[SeriesParams] = None
    ) -> AutoregressiveMovingAverage:
        """
        Constructing excitation time series generation for sampling from autoregressive moving averages process.

        :param series_params: The parameters for generating management incentive time series data.
        :return:
        """
        series_params = self.series_params if series_params is None else series_params
        return AutoregressiveMovingAverage(
            p_min=series_params.p_min,
            p_max=series_params.p_max,
            q_min=series_params.q_min,
            q_max=series_params.q_max,
            upper_bound=series_params.upper_bound,
            dtype=self.dtype,
        )

    def create_forecast_pfn(
        self, series_params: Optional[SeriesParams] = None
    ) -> ForecastPFN:
        """
        Constructing excitation time series generation for sampling from ForecastPFN.

        :param series_params: The parameters for generating management incentive time series data.
        :return:
        """
        series_params = self.series_params if series_params is None else series_params
        return ForecastPFN(
            is_sub_day=series_params.is_sub_day,
            transition=series_params.transition,
            start_time=series_params.start_time,
            end_time=series_params.end_time,
            random_walk=series_params.random_walk,
            dtype=self.dtype,
        )

    def create_kernel_synth(
        self, series_params: Optional[SeriesParams] = None
    ) -> KernelSynth:
        """
        Constructing excitation time series generation for sampling from KernelSynth.

        :param series_params: The parameters for generating management incentive time series data.
        :return:
        """
        series_params = self.series_params if series_params is None else series_params
        return KernelSynth(
            min_kernels=series_params.min_kernels,
            max_kernels=series_params.max_kernels,
            exp_sine_squared=series_params.exp_sine_squared,
            dot_product=series_params.dot_product,
            rbf=series_params.rbf,
            rational_quadratic=series_params.rational_quadratic,
            white_kernel=series_params.white_kernel,
            constant_kernel=series_params.constant_kernel,
            dtype=self.dtype,
        )

    def create_intrinsic_mode_function(
        self, series_params: Optional[SeriesParams] = None
    ) -> IntrinsicModeFunction:
        """
        Constructing excitation time series generation for sampling from intrinsic mode function.

        :param series_params: The parameters for generating management incentive time series data.
        :return:
        """
        series_params = self.series_params if series_params is None else series_params
        return IntrinsicModeFunction(
            min_base_imfs=series_params.min_base_imfs,
            max_base_imfs=series_params.max_base_imfs,
            min_choice_imfs=series_params.min_choice_imfs,
            max_choice_imfs=series_params.max_choice_imfs,
            probability_dict=series_params.imfs_dict,
            probability_list=series_params.imfs_list,
            min_duration=series_params.min_duration,
            max_duration=series_params.max_duration,
            min_amplitude=series_params.min_amplitude,
            max_amplitude=series_params.max_amplitude,
            min_frequency=series_params.min_frequency,
            max_frequency=series_params.max_frequency,
            dtype=self.dtype,
        )

    @property
    def series_params(self) -> SeriesParams:
        """Get the parameters for generating management incentive time series data."""
        return self._series_params

    @property
    def sampling_methods(self) -> List[str]:
        """Returns a list of various name of the different sampling methods"""
        return self.series_params.sampling_methods

    @property
    def sampling_object(self) -> List[Any]:
        """Returns a list of class objective of various sampling methods"""
        return list(self._sampling_dict.values())

    @property
    def sampling_dict(self) -> Dict[
        str,
        MixedDistribution
        | AutoregressiveMovingAverage
        | ForecastPFN
        | KernelSynth
        | IntrinsicModeFunction,
    ]:
        return self._sampling_dict

    @property
    def prob_array(self) -> np.ndarray:
        """Obtaining the sampling probability of time series data with different incentive methods."""
        return self.series_params.prob_array

    @property
    def dtype(self) -> np.dtype:
        """Obtaining the data type of the time series data with different incentive methods."""
        return self.series_params.dtype

    def generate(
        self,
        rng: np.random.RandomState,
        n_inputs_points: int,
        input_dimension: Optional[int] = 1,
        normalization: Optional[str] = None,
    ) -> np.ndarray:
        """ """
        # 1. Randomly select different sampling methods according to the specified probability
        choice_list = self.choice(rng=rng, input_dimension=input_dimension)

        # 2. Traverse the array to get the specific runnable instantiation object from the sampling dictionary
        time_series = np.hstack(
            [
                self.sampling_dict[name].generate(
                    rng=rng,
                    n_inputs_points=n_inputs_points,
                    input_dimension=1,
                )
                for name in choice_list
            ]
        )

        # 3. Whether to normalize the stimulus time series data
        if normalization is None:
            return time_series
        if normalization == "z-score":
            for dim in range(input_dimension):
                time_series[:, dim] = z_score_normalization(x=time_series[:, dim])
        elif normalization == "max-min":
            for dim in range(input_dimension):
                time_series[:, dim] = max_min_normalization(x=time_series[:, dim])
        else:
            raise ValueError(
                "The normalization option must be 'z-score' or 'max-min' or None!"
            )

        return time_series


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    excitation = Excitation()

    for i in range(20, 60):
        time_series = excitation.generate(
            np.random.RandomState(i), n_inputs_points=100, input_dimension=3
        )
        print(time_series.shape)
        # plt.plot(time_series)
        # plt.show()
