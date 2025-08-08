# -*- coding: utf-8 -*-
"""
Created on 2025/01/25 00:02:43
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

from typing import Optional, Dict, Any, Tuple, List

from S2Generator import Node, NodeList
from S2Generator.utils.print_symbol import symbol_to_markdown


def plot_series(x: np.ndarray, y: np.ndarray) -> plt.Figure:
    """
    Visualize S2 data

    :param x: input sampling series
    :param y: output generated series
    :return: the plot figure of matplotlib
    """

    # Determine the shape and length of the data
    (seq_len, input_dim) = x.shape
    (_, output_dim) = y.shape
    max_dim = max(input_dim, output_dim)

    # Create a matplotlib plotting object
    fig, axes = plt.subplots(
        nrows=max_dim, ncols=2, figsize=(12, 2 * max_dim), sharex=True
    )

    # Plot the input sequence
    for i in range(input_dim):
        if max_dim == 1:
            ax = axes[0]
        else:
            ax = axes[i, 0]
        ax.plot(x[:, i], color="royalblue")
        ax.set_ylabel(f"Input Dim {i + 1}", fontsize=10)
        ax.set_xlim(0, seq_len)

    # Plot the output sequence
    for i in range(output_dim):
        if max_dim == 1:
            ax = axes[1]
        else:
            ax = axes[i, 1]
        ax.plot(y[:, i], color="royalblue")
        ax.set_ylabel(f"Output Dim {i + 1}", fontsize=10)
        ax.set_xlim(0, seq_len)

    # Add titles to the two columns of images
    if max_dim == 1:
        axes[0].set_title("Input Data", fontsize=12)
        axes[1].set_title("Output Data", fontsize=12)
    else:
        axes[0, 0].set_title("Input Data", fontsize=12)
        axes[0, 1].set_title("Output Data", fontsize=12)

    return fig


def which_edges_out(artist: plt.Text | Any, *, padding: Optional[int] = 0) -> Dict[str, bool]:
    """
    判断 artist 在画布的哪几条边之外。
    padding：像素级额外安全边距（可为负值，表示“几乎出去”）。
    返回 dict：{'top','bottom','left','right'} -> True/False
    """
    fig = artist.figure
    if fig is None:
        raise ValueError("artist 尚未添加到任何 figure")

    # 进行对象的渲染
    renderer = fig.canvas.get_renderer()
    bbox = artist.get_window_extent(renderer=renderer)

    # 考虑 padding
    if padding:
        bbox = bbox.expanded(padding / fig.dpi, padding / fig.dpi)

    # 画布像素边界
    w, h = fig.canvas.get_width_height()
    canvas = Bbox([[0, 0], [w, h]])

    return {
        'left': bbox.xmin < canvas.xmin,  # 整盒都在画布左侧之外
        'right': bbox.xmax > canvas.xmax,  # 整盒都在画布右侧之外
        'bottom': bbox.ymin < canvas.ymin,  # 整盒都在画布下方之外
        'top': bbox.ymax > canvas.ymax,  # 整盒都在画布上方之外
    }


def create_symbol_figure(symbol: str | List[str], width: float, height: float, dpi: Optional[int] = 300) -> Tuple[
    plt.Figure, plt.Axes, List[plt.Text]]:
    """


    :param symbol:
    :param width:
    :param height:
    :param dpi:
    :return:
    """
    # Create the Figure for matplotlib
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    # 占满整幅图
    # ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 完全空白，不显示坐标轴
    ax.axis('off')

    if isinstance(symbol, str):
        text = ax.text(0.5, 0.5, symbol,
                    ha='center', va='center',
                    fontsize=14)  # 去掉 usetex=True
        text = [text]
    elif isinstance(symbol, list):
        number = len(symbol)

        position = np.arange(0, number + 2)
        position = (position - position.min()) / (position.max() - position.min())
        position = position[1:-1]
        text = [ax.text(0.5, pos, s, ha='center', va='center', fontsize=14) for (s, pos) in zip(symbol, position[::-1])]

    else:
        raise ValueError("symbol must be str or list")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # 去掉 x 和 y 轴的刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 去掉刻度标签（如果只想隐藏刻度线但保留标签，可以跳过这一步）
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return fig, ax, text


def plot_symbol(symbol: str | Node | NodeList, width: Optional[int] = 20, height: Optional[int] = None, dpi: Optional[int] = 160, return_all: Optional[str] = False) -> plt.Figure | Tuple[plt.Figure, plt.Axes, List[plt.Text]]:
    """
    对符号数据进行可视化的函数。
    由于输入的符号表达式数据各不相同，因此在实际使用时可能需要多次调整width的大小

    :param symbol: 待可视化的符号数据。
    :param width: 绘制图像的宽度大小，可能需要多次调整。
    :param height: 绘制图像的高度大小，为None时则由算法自动指定。
    :param dpi: 可视化图像的分辨率大小。
    :param return_all: 是否要返回全部的可视化信息
    :return: - True: return (Figure, Axis, List[Text]),
             - False: return Figure.
    """
    # Transform the symbol from string to markdown
    symbol_list = symbol_to_markdown(symbol)

    # 为每一个符号添加y和角标
    symbol_list = [f"$ y_{i} = {sym} $" for (i, sym) in enumerate(symbol_list)]

    # 给出初始化的高度和宽度数值
    if height is None:
        height = 0.50 * len(symbol_list)

    # 对符号进行可视化
    fig, ax, text = create_symbol_figure(symbol_list, width, height, dpi=dpi)

    # 是否要返回所有的绘图信息
    if return_all is True:
        return fig, ax, text
    return fig




# # -------------------------------------------------
# # 1. 你的公式
# formula = r"$E = mc^2 + \int_0^\infty e^{-x^2}\,dx + \sum_{k=1}^{n}\frac{a_k}{1+k^2}$"
# # -------------------------------------------------
#
# # 3. 根据尺寸创建真正的空白画布
# fig = plt.figure(figsize=(20, 5))
# ax  = fig.add_axes([0, 0, 1, 1])   # 占满整幅图
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.axis('off')                     # 完全空白，不显示坐标轴
#
# # 4. 把公式放在正中央
# fig.text(0.5, 0.5, formula,
#          ha='center', va='center',
#          fontsize=14)  # 去掉 usetex=True
#
# plt.show()


if __name__ == '__main__':
    import numpy as np
    # Importing data generators, parameter controllers and visualization functions
    from S2Generator import Generator, Params, plot_series

    params = Params()  # Adjust the parameters here
    generator = Generator(params)  # Create an instance

    rng = np.random.RandomState(0)  # Creating a random number object
    # Start generating symbolic expressions, sampling and generating series


    trees, x, y = generator.run(rng, input_dimension=2, output_dimension=10, n_points=20)

    trees_list = str(trees).split(" | ")
    for i, tree in enumerate(trees_list):
        print(i, tree)

    # Print the expressions
    fig = plot_symbol(trees)

    fig.savefig("test.png")
    plt.show()