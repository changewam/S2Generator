<img width="100%" align="middle" src=".\images\background.png?raw=true">

<div align="center">

---

[![PyPI version](https://badge.fury.io/py/PySDKit.svg)](https://pypi.org/project/PySDKit/)  ![License](https://img.shields.io/github/license/wwhenxuan/PySDKit) [![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python)](https://www.python.org/) [![Downloads](https://pepy.tech/badge/pysdkit)](https://pepy.tech/project/pysdkit) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Installation](#Installation) | [Examples](https://github.com/wwhenxuan/S2Generator/tree/main/examples) | [Docs]() | [Acknowledge]() | [Cite]()

</div>

In recent years, the fondation models of Time Series Analysis `(TSA)` have developed rapidly. However, due to data privacy and collection difficulties, large-scale datasets in TSA currently have data shortages and imbalanced representation distribution. This will cause the foundation models pre-trained on them to have certain performance prediction biases, reducing the generalization ability and scalability of the model. At the same time, the semantic information of time series has never been fully explored, which seriously hinders the development of deep learning models for TSA in the direction of multimodality.

In order to solve the above two problems, we believe that time series is a representation of complex dynamic systems, so time series can form a pairing relationship with the symbolic description of the corresponding complex system. The symbolic expression of modeling complex systems can be regarded as the semantic information of the time series. Based on the this view, our S2Generator provides a series-symbol bimodal data generation algorithm. The algorithm can generate high-quality time series data and its paired symbolic expression without restriction to overcome the problems of data shortage and semantic information loss in the field of time series analysis. The specific data generation method is shown in Figure (a) below. Through this method, we generated a large-scale synthetic dataset and trained a bimodal pre-trained basic model on it as shown in Figure (b) below.

<img src="https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/SymTime.png" alt="SymTime" style="zoom:33%;" />

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
import sys
import os

# Importing data generators object
from S2Generator import Generator

# Creating a random number object
rng = np.random.RandomState(0)

print_hello()
~~~

> (73.5 add (sin((-7.57 add (3.89 mul x_0))) mul (((-0.092 mul exp((-63.4 add (-0.204 mul x_0)))) add (-6.12 mul log((-0.847 add (9.55 mul x_0))))) sub ((4.49 mul inv((-29.3 add (-86.2 mul x_0)))) add (-2.57 mul sqrt((51.3 add (-55.6 mul x_0))))))))

<img width="100%" align="middle" src="![https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/ID1_OD1.jpg?raw=true]()">

The input and output dimensions of the multivariate time series and the length of the sampling sequence can be adjusted in the `run` method.

~~~python
rng = np.random.RandomState(42)  # Change the random seed

# Try to generate the multi-channels time series
trees, x, y = generator.run(rng, input_dimension=2, output_dimension=2, n_points=256)
print(trees)
fig = s2plot(x, y)
~~~

> (7.49 add ((((7.77 mul x_0) add (-89.8 mul sqrt((0.81 add (3.88 mul x_0))))) sub (0.14 mul x_1)) sub (-84.1 mul exp((9.58 add (-81.6 mul x_0)))))) | (-38.6 add ((87.1 mul sin((0.554 add (-57.5 mul x_1)))) sub ((-40.7 mul sin(((-0.86 mul exp((-6.46 add (3.31 mul x_0)))) add (5.57 mul x_0)))) sub (-65.5 mul log(((-0.318 mul x_1) add (-8.19 mul x_0)))))))
>
> Two symbolic expressions are connected by `|`.

![ID2_OD2](https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/ID2_OD2.jpg)

## Algorithm 🎯

The key to this algorithm is to construct complex and diverse symbolic expressions $f(\cdot)$ through a tree structure, so as to generate a series $y$ by forward propagating through a sampling series $x$. Since the symbolic expressions of mathematical operations can be represented by a tree structure, we first construct a binary tree with random binary operators to form the basic framework of the expression, as shown in Figure (a). Then we insert random constants or variables as leaf nodes into the constructed structure to form a full binary tree, as shown in Figure (b). Then we increase the diversity of symbolic expressions by randomly inserting unary operators and radioactive transformations, as shown in Figure (c).

![trees](https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/images/trees.jpg)

## Citation 🎖️ <a id="Citation"></a>

~~~latex
@inproceedings{
SNIP,
title={{SNIP}: Bridging Mathematical Symbolic and Numeric Realms with Unified Pre-training},
author={Kazem Meidani and Parshin Shojaee and Chandan Reddy and Amir Barati Farimani},
booktitle={NeurIPS 2023 AI for Science Workshop},
year={2023},
url={https://openreview.net/forum?id=Nn43zREWvX}
}
~~~