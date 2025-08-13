import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import date

from numpy import ndarray, dtype, floating
from numpy._typing import _32Bit
from pandas.tseries.frequencies import to_offset
from scipy.stats import beta

from dataclasses import dataclass

from typing import Optional, Tuple, List, Any


# ========================series_configs.py
@dataclass
class ComponentScale:
    base: float
    linear: float = None
    exp: float = None
    a: np.ndarray = None
    q: np.ndarray = None
    m: np.ndarray = None
    w: np.ndarray = None
    h: np.ndarray = None
    minute: np.ndarray = None


@dataclass
class ComponentNoise:
    # shape parameter for the weibull distribution
    k: float
    median: float

    # noise will be finally calculated as
    # noise_term = (1 + scale * (noise - E(noise)))
    # no noise can be represented by scale = 0
    scale: float


@dataclass
class SeriesConfig:
    scale: ComponentScale
    offset: ComponentScale
    noise_config: ComponentNoise

    def __str__(self):
        return f"L{1000 * self.scale.linear:+02.0f}E{10000 * (self.scale.exp - 1):+02.0f}A{100 * self.scale.a:02.0f}M{100 * self.scale.m:02.0f}W{100 * self.scale.w:02.0f}"


class Config:
    frequencies = None
    frequency_names = None
    freq_and_index = None
    transition = False

    @classmethod
    def set_freq_variables(cls, is_sub_day):
        if is_sub_day:
            # TODO: 这里的 is_sub_day 是什么含义
            # cls.frequencies = [("min", 1/1440), ("h", 1/24), ("D", 1), ("W", 7), ("MS", 30), ("YE", 12)]
            # cls.frequency_names = ["minute", "hourly", "daily", "weekly", "monthly", "yearly"]
            # cls.freq_and_index = (("minute", 0), ("hourly", 1), ("daily", 2), ("weekly", 3), ("monthly", 4), ("yearly", 5))

            # Whenxuan: we remove the frequencies of year (YE)
            cls.frequencies = [
                ("min", 1 / 1440),
                ("h", 1 / 24),
                ("D", 1),
                ("W", 7),
                ("MS", 30),
            ]
            cls.frequency_names = ["minute", "hourly", "daily", "weekly", "monthly"]
            cls.freq_and_index = (
                ("minute", 0),
                ("hourly", 1),
                ("daily", 2),
                ("weekly", 3),
                ("monthly", 4),
            )
        else:
            cls.frequencies = [("D", 1), ("W", 7), ("MS", 30)]
            cls.frequency_names = ["daily", "weekly", "monthly"]
            cls.freq_and_index = (("daily", 0), ("weekly", 1), ("monthly", 2))

    @classmethod
    def set_transition(cls, transition):
        cls.transition = transition


# ========================utils.py
def weibull_noise(
    k: Optional[float] = 2, length: Optional[int] = 1, median: Optional[int] = 1
) -> np.ndarray:
    """
    Function to generate weibull noise with a fixed median.
    Its main feature is that it achieves a fixed median output by adjusting the scale parameter.
    The probability density function of the Weibull distribution is:
    f(x; λ, k) = (k/λ)(x/λ)^{k-1}e^{-(x/λ)^k} for x ≥ 0
    To achieve a fixed median, the function inversely solves for the scale parameter λ using the median formula:
    lamda = median / (np.log(2) ** (1 / k))

    :param k: Shape parameter, determines the shape of the distribution:
              1. k < 1: Decreasing failure rate;
              2. k = 1: Exponential distribution;
              3. k > 1: Increasing failure rate.
    :param length: Shape parameter, determines the shape of the distribution:
    :param median: Mandatory median (50% quantile).
    :return:
    """
    # we set lambda so that median is a given value
    lamda = median / (np.log(2) ** (1 / k))

    return lamda * np.random.weibull(k, length)


def shift_axis(
    days: pd.DatetimeIndex, shift: Optional[pd.DatetimeIndex] = None
) -> pd.DatetimeIndex:
    """
    Used to adjust the relative position of a time series (or other numerical series),
    specifically to shift the series proportionally to a new reference point.

    :param days: pd.DatetimeIndex containing the time series to shift.
    :param shift: Shift parameter, determines the shape of the distribution:
    :return: pd.DatetimeIndex containing the shifted time series.
    """
    if shift is None:
        return days
    return days - shift * days[-1]


def get_random_walk_series(length: int, movements: Optional[List[int]] = None):
    """
    Function to generate a random walk series with a specified length.
    This is a random process model widely used in finance, physics, statistics and other fields.

    :param length: Shape parameter, determines the shape of the distribution:
    :param movements: Shape parameter, possible step sizes:
                      1. Default: Binary Random Walk (±1);
                      2. Customizable (e.g., [-2, 0, 2]).
    :return: pd.DatetimeIndex containing the random walk series.
    """
    if movements is None:
        movements = [-1, 1]

    random_walk = list()
    random_walk.append(np.random.choice(movements))
    for i in range(1, length):
        movement = np.random.choice(movements)
        value = random_walk[i - 1] + movement
        random_walk.append(value)

    return np.array(random_walk)


def sample_scale() -> np.ndarray:
    """
    Function to sample scale such that it follows 60-30-10 distribution
    i.e. 60% of the times it is very low, 30% of the times it is moderate and
    the rest 10% of the times it is high
    """
    rand = np.random.rand()

    # very low noise
    if rand <= 0.6:
        return np.random.uniform(0, 0.1)
    # moderate noise
    elif rand <= 0.9:
        return np.random.uniform(0.2, 0.4)
    # high noise
    else:
        return np.random.uniform(0.6, 0.8)


def get_transition_coefficients(context_length: int) -> np.ndarray:
    """
    Transition series refers to the linear combination of 2 series
    S1 and S2 such that the series S represents S1 for a period and S2
    for the remaining period. We model S as S = (1 - f) * S1 + f * S2
    Here f = 1 / (1 + e^{-k (x-m)}) where m = (a + b) / 2 and k is chosen
    such that f(a) = 0.1 (and hence f(b) = 0.9). a and b refer to
    0.2 * CONTEXT_LENGTH and 0.8 * CONTEXT_LENGTH

    :param: context_length: The length of time series to be generated.
    :return: np.ndarray of transition coefficients.
    """
    # a and b are chosen with 0.2 and 0.8 parameters
    a, b = 0.2 * context_length, 0.8 * context_length

    # fixed to this value
    f_a = 0.1

    m = (a + b) / 2
    k = 1 / (a - m) * np.log(f_a / (1 - f_a))

    coeff = 1 / (1 + np.exp(-k * (np.arange(1, context_length + 1) - m)))
    return coeff


def make_series_trend(
    series: SeriesConfig, dates: pd.DatetimeIndex
) -> ndarray[Any, dtype[Any]]:
    """
    Function to generate the trend(t) component of synthetic series.

    :param series: series config for generating trend of synthetic series
    :param dates: dates for which data is present
    :return: trend component of synthetic series
    """
    values = np.full_like(dates, series.scale.base, dtype=np.float32)

    days = (dates - dates[0]).days
    if series.scale.linear is not None:
        values += shift_axis(days, series.offset.linear) * series.scale.linear
    if series.scale.exp is not None:
        values *= np.power(series.scale.exp, shift_axis(days, series.offset.exp))

    return values


def get_freq_component(
    dates_feature: pd.Index, n_harmonics: int, n_total: int
) -> np.ndarray | Any:
    """
    Method to get systematic movement of values across time
    :param dates_feature: the component of date to be used for generating
    the seasonal movement is different. For example, for monthly patterns
    in a year we will use months of a date, while for day-wise patterns in
    a month, we will use days as the feature

    :param n_harmonics: number of harmonics to include.
                        For example, for monthly trend, we use 12/2 = 6 harmonics
    :param n_total: total cycle length
    :return: numpy array of shape dates_feature.shape containing
    sinusoidal value for a given point in time
    """
    harmonics = list(range(1, n_harmonics + 1))

    # initialize sin and cosine coefficients with 0
    sin_coef = np.zeros(n_harmonics)
    cos_coef = np.zeros(n_harmonics)

    # choose coefficients inversely proportional to the harmonic
    for idx, harmonic in enumerate(harmonics):
        sin_coef[idx] = np.random.normal(scale=1 / harmonic)
        cos_coef[idx] = np.random.normal(scale=1 / harmonic)

    # normalize the coefficients such that their sum of squares is 1
    coef_sq_sum = np.sqrt(np.sum(np.square(sin_coef)) + np.sum(np.square(cos_coef)))
    sin_coef /= coef_sq_sum
    cos_coef /= coef_sq_sum

    # construct the result for systematic movement which
    # comprises of patterns of varying frequency
    return_val = 0
    for idx, harmonic in enumerate(harmonics):
        return_val += sin_coef[idx] * np.sin(
            2 * np.pi * harmonic * dates_feature / n_total
        )
        return_val += cos_coef[idx] * np.cos(
            2 * np.pi * harmonic * dates_feature / n_total
        )

    return return_val


def make_series_seasonal(series: SeriesConfig, dates: pd.DatetimeIndex) -> Any:
    """
    Function to generate the seasonal(t) component of synthetic series.
    It represents the systematic pattern-based movement over time

    :param series: series config used for generating values
    :param dates: dates on which the data needs to be calculated
    """
    seasonal = 1

    seasonal_components = defaultdict(lambda: 1)
    if series.scale.minute is not None:
        seasonal_components["minute"] = 1 + series.scale.minute * get_freq_component(
            dates.minute, 10, 60
        )
        seasonal *= seasonal_components["minute"]
    if series.scale.h is not None:
        seasonal_components["h"] = 1 + series.scale.h * get_freq_component(
            dates.hour, 10, 24
        )
        seasonal *= seasonal_components["h"]
    if series.scale.a is not None:
        seasonal_components["a"] = 1 + series.scale.a * get_freq_component(
            dates.month, 6, 12
        )
        seasonal *= seasonal_components["a"]
    if series.scale.m is not None:
        seasonal_components["m"] = 1 + series.scale.m * get_freq_component(
            dates.day, 10, 30.5
        )
        seasonal *= seasonal_components["m"]
    if series.scale.w is not None:
        seasonal_components["w"] = 1 + series.scale.w * get_freq_component(
            dates.dayofweek, 4, 7
        )
        seasonal *= seasonal_components["w"]

    seasonal_components["seasonal"] = seasonal
    return seasonal_components


def make_series(
    series: SeriesConfig,
    freq: pd.DateOffset,
    periods: int,
    start: pd.Timestamp,
    options: dict,
    random_walk: bool,
):
    """
    make series of the following form
    series(t) = trend(t) * seasonal(t)
    """
    start = freq.rollback(start)
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    scaled_noise_term = 0

    if random_walk:
        values = get_random_walk_series(len(dates))
    else:
        values_trend = make_series_trend(series, dates)
        values_seasonal = make_series_seasonal(series, dates)

        values = values_trend * values_seasonal["seasonal"]

        weibull_noise_term = weibull_noise(
            k=series.noise_config.k,
            median=series.noise_config.median,
            length=len(values),
        )

        # approximating estimated value from median
        noise_expected_val = series.noise_config.median

        # expected value of this term is 0
        # for no noise, scale is set to 0
        scaled_noise_term = series.noise_config.scale * (
            weibull_noise_term - noise_expected_val
        )

    dataframe_data = {
        **values_seasonal,
        "values": values,
        "noise": 1 + scaled_noise_term,
        "dates": dates,
    }

    return dataframe_data


BASE_START = date.fromisoformat("1885-01-01").toordinal()
BASE_END = date.fromisoformat("2050-12-31").toordinal() + 1

PRODUCT_SCHEMA = {
    "doc": "Timeseries sample",
    "name": "TimeseriesSample",
    "type": "record",
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "ts", "type": {"type": "int", "logicalType": "date"}},
        {"name": "y", "type": ["null", "float"]},
        {"name": "noise", "type": ["float"]},
    ],
}


def __generate(
    n=100,
    freq_index: int = None,
    start: pd.Timestamp = None,
    options: dict = {},
    random_walk: bool = False,
):
    """
    Function to construct synthetic series configs and generate synthetic series.

    :param n: The length of time series to generate.
    :param freq_index: The frequency of time series to generate.
    :param start: The start date of time series to generate.
    :param options: Options dict for generating series.
    :param random_walk: Whether to generate random walk or not.
    """
    if freq_index is None:
        # TODO: 这里完全可以在这个地方就随机指定
        freq_index = np.random.choice(len(Config.frequencies))

    # 获取时间频率
    freq, timescale = Config.frequencies[freq_index]

    # annual, monthly, weekly, hourly and minutely components
    a, m, w, h, minute = 0.0, 0.0, 0.0, 0.0, 0.0
    if freq == "min":
        minute = np.random.uniform(0.0, 1.0)
        h = np.random.uniform(0.0, 0.2)
    elif freq == "h":
        minute = np.random.uniform(0.0, 0.2)
        h = np.random.uniform(0.0, 1)
    elif freq == "D":
        w = np.random.uniform(0.0, 1.0)
        m = np.random.uniform(0.0, 0.2)
    elif freq == "W":
        m = np.random.uniform(0.0, 0.3)
        a = np.random.uniform(0.0, 0.3)
    elif freq == "MS":
        w = np.random.uniform(0.0, 0.1)
        a = np.random.uniform(0.0, 0.5)
    elif freq == "YE":
        w = np.random.uniform(0.0, 0.2)
        a = np.random.uniform(0.0, 1)
    else:
        raise NotImplementedError

    if start is None:
        # start = pd.Timestamp(date.fromordinal(np.random.randint(BASE_START, BASE_END)))
        start = pd.Timestamp(
            date.fromordinal(int((BASE_START - BASE_END) * beta.rvs(5, 1) + BASE_START))
        )

    scale_config = ComponentScale(
        1.0,
        np.random.normal(0, 0.01),
        np.random.normal(1, 0.005 / timescale),
        a=a,
        m=m,
        w=w,
        minute=minute,
        h=h,
    )

    offset_config = ComponentScale(
        0,
        np.random.uniform(-0.1, 0.5),
        np.random.uniform(-0.1, 0.5),
        a=np.random.uniform(0.0, 1.0),
        m=np.random.uniform(0.0, 1.0),
        w=np.random.uniform(0.0, 1.0),
    )

    noise_config = ComponentNoise(
        k=np.random.uniform(1, 5), median=1, scale=sample_scale()
    )

    cfg = SeriesConfig(scale_config, offset_config, noise_config)

    return make_series(cfg, to_offset(freq), n, start, options, random_walk)


def generate(
    n=100,
    freq_index: int = None,
    start: pd.Timestamp = None,
    options: dict = {},
    random_walk: bool = False,
) -> np.ndarray:
    """
    Function to generate a synthetic series for a given config
    """

    series1 = __generate(n, freq_index, start, options, random_walk)
    series2 = __generate(n, freq_index, start, options, random_walk)

    if Config.transition:
        coeff = get_transition_coefficients(CONTEXT_LENGTH)
        values = coeff * series1["values"] + (1 - coeff) * series2["values"]
    else:
        values = series1["values"]

    return values


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    # 这两个参数是可以控制的
    Config.set_freq_variables(True)
    Config.set_transition(True)

    # 这个控制的应该是生成序列的次数
    N = 50
    options = {}
    # 这个参数控制的是生成序列的长度
    size = CONTEXT_LENGTH = 1_000

    for freq, freq_index in Config.freq_and_index:
        print("freq", freq, freq_index)
        start = None

        for i in range(N):
            if i % 1000 == 0:
                print(f"Completed: {i}")

            if i < N * options.get("linear_random_walk_frac", 0):
                sample = generate(
                    size,
                    freq_index=freq_index,
                    start=start,
                    options=options,
                    random_walk=True,
                )
            else:
                sample = generate(
                    size, freq_index=freq_index, start=start, options=options
                )
            break

        plt.plot(sample, color="royalblue")
        plt.show()
        break
