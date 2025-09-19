<img width="100%" align="middle" src=".\images\background.png?raw=true">

<div align="center">

[![PyPI version](https://badge.fury.io/py/PySDKit.svg)](https://pypi.org/project/PySDKit/)  ![License](https://img.shields.io/github/license/wwhenxuan/PySDKit) [![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python)](https://www.python.org/) [![Downloads](https://pepy.tech/badge/pysdkit)](https://pepy.tech/project/pysdkit) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Installation](#Installation) | [Examples](https://github.com/wwhenxuan/S2Generator/tree/main/examples) | [Docs]() | [Acknowledge]() | [Citation](#Citation)

</div>


基于时间序列是复杂动力系统的外在表征这一重要观点，我们提出了一个符号和序列双模态的时间序列数据生成机制。该机制能够无限制构造大量的复杂系统（符号表达式）$f(\cdot)$与激励时间序列$X$，并将激励输入到复杂系统中获得响应时间序列$Y=f(X)$。



## Installation 🚀 <a id="Installation"></a>

We have highly encapsulated the algorithm and uploaded the code to PyPI. Users can download the code through `pip`.

~~~
pip install s2generator
~~~

We only used [`NumPy`](https://numpy.org/), [`Scipy`](https://scipy.org/) and [`matplotlib`](https://matplotlib.org/) when developing the project.

## Usage ✨

We provide two interfaces [`Params`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/params.py) and [`Generator`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/generators.py). [`Params`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/params.py) is used to modify the configuration of data generation. [`Generator`](https://github.com/wwhenxuan/S2Generator/blob/main/S2Generator/generators.py) creates a specific data generation object. We start data generation through the `run` method.

~~~python
import numpy as np

# Importing data generators object
from S2Generator import Generator, SeriesParams, SymbolParams, plot_series

# Creating a random number object
rng = np.random.RandomState(0)

# Create the parameter control modules
series_params = SeriesParams()
symbol_params = SymbolParams()  # specify specific parameters here or use the default parameters

# Create an instance
generator = Generator(series_params=series_params, symbol_params=symbol_params)

# Start generating symbolic expressions, sampling and generating series
symbols, inputs, outputs = generator.run(
    rng, input_dimension=1, output_dimension=1, n_inputs_points=256
)

# Print the expressions
print(symbols)
# Visualize the time series
fig = plot_series(inputs, outputs)
~~~

> (73.5 add (x_0 mul (((9.38 mul cos((-0.092 add (-6.12 mul x_0)))) add (87.1 mul arctan((-0.965 add (0.973 mul rand))))) sub (8.89 mul exp(((4.49 mul log((-29.3 add (-86.2 mul x_0)))) add (-2.57 mul ((51.3 add (-55.6 mul x_0)))**2)))))))

![ID2_OD2](https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/ID1_OD1.jpg)

The input and output dimensions of the multivariate time series and the length of the sampling sequence can be adjusted in the `run` method.

~~~python
rng = np.random.RandomState(512)  # Change the random seed

# Try to generate the multi-channels time series
symbols, inputs, outputs = generator.run(rng, input_dimension=2,
                                         output_dimension=2,
                                         n_inputs_points=336)

print(symbols)
fig = plot_series(inputs, outputs)
~~~

> (-9.45 add ((((0.026 mul rand) sub (-62.7 mul cos((4.79 add (-6.69 mul x_1))))) add (-0.982 mul sqrt((4.2 add (-0.14 mul x_0))))) sub (0.683 mul x_1))) | (67.6 add ((-9.0 mul x_1) add (2.15 mul sqrt((0.867 add (-92.1 mul x_1))))))
>
> Two symbolic expressions are connected by " | ".

![ID2_OD2](https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/ID2_OD2.jpg)

## Algorithm 🎯 <img width="25%" align="right" src="https://github.com/wwhenxuan/S2Generator/blob/master/images/trees.png?raw=true">

The key to this algorithm is to construct complex and diverse symbolic expressions $f(\cdot)$ through a tree structure, so as to generate a series $y$ by forward propagating through a sampling series $x$. Since the symbolic expressions of mathematical operations can be represented by a tree structure, we first construct a binary tree with random binary operators to form the basic framework of the expression, as shown in Figure (a). Then we insert random constants or variables as leaf nodes into the constructed structure to form a full binary tree, as shown in Figure (b). Then we increase the diversity of symbolic expressions by randomly inserting unary operators and radioactive transformations, as shown in Figure (c).

![trees](https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/trees.png)

## Citation 🎖️ <a id="Citation"></a>

~~~latex
@misc{wang2025mitigatingdatascarcitytime,
      title={Mitigating Data Scarcity in Time Series Analysis: A Foundation Model with Series-Symbol Data Generation}, 
      author={Wenxuan Wang and Kai Wu and Yujian Betterest Li and Dan Wang and Xiaoyu Zhang and Jing Liu},
      year={2025},
      eprint={2502.15466},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.15466}, 
}
~~~