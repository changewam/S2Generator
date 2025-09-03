# -*- coding: utf-8 -*-
"""
Created on 2025/08/23 17:17:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import unittest
from os import path

import numpy as np

from S2Generator.utils._tools import (
    get_time_now,
    ensure_directory_exists,
    save_s2data,
    save_npy,
    save_npz,
    load_s2data,
    load_npy,
    load_npz,
    is_all_zeros,
    z_score_normalization,
    max_min_normalization,
)


class TestTools(unittest.TestCase):
    """对工具模块的函数进行单元测试"""

    # 构建输入序列
    time_series = np.random.uniform(low=-10, high=10, size=(256, 10))

    # 构建全0时间序列
    zero_series = np.zeros((256, 10))

    # 构建含有无穷大数值的时间序列
    inf_series = np.random.uniform(low=-10, high=10, size=(256, 10))
    inf_series[0, 0] = np.inf
    inf_series[-1, -1] = -np.inf

    # 构建含有NaN值的时间序列
    nan_series = np.random.uniform(low=-10, high=10, size=(256, 10))
    nan_series[0, 0] = np.nan
    nan_series[-1, -1] = np.nan

    # 测试npy数据的地址
    npy_path = "./data/data.npy"

    # 测试npz的地址
    npz_path = "./data/data.npz"

    # 测试S2数据的地址
    s2_npy_path = "./data/s2data.npy"
    s2_npz_path = "./data/s2data.npz"

    # 创建用于保存的样例数据
    data = {"symbol": "hello", "excitation": 256, "response": 256}

    def test_z_score_normalization(self) -> None:
        """测试z-score标准化的函数"""
        # 执行标准化算法
        normalized = z_score_normalization(self.time_series)

        # 检验均值和方差
        self.assertAlmostEqual(
            first=np.mean(normalized),
            second=0,
            delta=0.01,
            msg="z-score normalization failed in mean test!",
        )
        self.assertAlmostEqual(
            first=np.mean(np.std(normalized, axis=0, keepdims=True)),
            second=1,
            delta=0.01,
            msg="z-score normalization failed in std test!",
        )

    def test_z_score_normalization_zero(self) -> None:
        """测试z-score标准化函数对全0时间序列的NaN值结果"""
        # 测试全0时间序列的错误结果
        normalized = z_score_normalization(self.zero_series)

        self.assertEqual(
            first=normalized,
            second=None,
            msg="输入为全零时间序列！",
        )

    def test_z_score_normalization_inf(self) -> None:
        """测试z-score标准化算法对含有无穷大数值的返回结果"""
        # 测试含无穷大数值的时间序列的错误结果
        normalized = z_score_normalization(self.zero_series)

        self.assertEqual(
            first=normalized,
            second=None,
            msg="输入为含有无穷大时间序列！",
        )

    def test_z_score_normalization_nan(self) -> None:
        """测试z-score标准化算法对含有NaN数值的返回结果"""
        # 测试含NaN数值的时间序列的错误结果
        normalized = z_score_normalization(self.zero_series)

        self.assertEqual(
            first=normalized,
            second=None,
            msg="输入为含有NaN时间序列！",
        )

    def test_max_min_normalization(self) -> None:
        """测试max-min标准化算法的函数"""
        # 执行标准化算法
        normalized = max_min_normalization(self.time_series)

        for i in range(normalized.shape[1]):
            # 获取其中的一段时间序列
            time = normalized[:, i]

            # 验证其中的最大值和最小值
            self.assertAlmostEqual(
                first=np.max(time),
                second=1,
                delta=1e-3,
                msg="max-min normalization failed in max-value test!",
            )
            self.assertAlmostEqual(
                first=np.min(time),
                second=0,
                delta=1e-3,
                msg="max-min normalization failed in max-value test!",
            )

    def test_max_min_normalization_zero(self) -> None:
        """测试max_min标准化函数对全0时间序列的NaN值结果"""
        # 测试全0时间序列的错误结果
        normalized = max_min_normalization(self.zero_series)

        self.assertEqual(
            first=normalized,
            second=None,
            msg="输入为全零时间序列！",
        )

    def test_max_min_normalization_inf(self) -> None:
        """测试z-score标准化算法对含有无穷大数值的返回结果"""
        # 测试含无穷大数值的时间序列的错误结果
        normalized = max_min_normalization(self.zero_series)

        self.assertEqual(
            first=normalized,
            second=None,
            msg="输入为含有无穷大时间序列！",
        )

    def test_max_min_normalization_nan(self) -> None:
        """测试z-score标准化算法对含有NaN数值的返回结果"""
        # 测试含NaN数值的时间序列的错误结果
        normalized = max_min_normalization(self.zero_series)

        self.assertEqual(
            first=normalized,
            second=None,
            msg="输入为含有NaN时间序列！",
        )

    def test_is_all_zeros(self) -> None:
        """测试判断一段时间序列是否为全0的函数"""
        # 先测试一段全0序列
        self.assertTrue(
            expr=is_all_zeros(self.zero_series), msg="测试全零时间序列输入错误!"
        )

        # 测试非0序列
        self.assertTrue(
            expr=not is_all_zeros(self.time_series), msg="测试非全零时间序列输入错误!"
        )

    def test_get_time_now(self) -> None:
        """测试获取当前时间信息的函数"""
        # 获取当前的时间
        now = get_time_now()
        # 检验数据类型
        self.assertIsInstance(obj=now, cls=str)

        # 检验数据的格式
        ymd = now.split(" ")[0]
        self.assertEqual(first=len(ymd), second=4 + 2 + 2 + 2)

    def test_ensure_directory_exists(self) -> None:
        """测试确保当前目录存在的函数"""
        # 尝试生成一个目录
        dir_path = "./data"

        if not path.exists(dir_path):
            # 如果这个目录不存在则开始验证
            ensure_directory_exists(dir_path)
            # 判断新创建的目录是存在的
            self.assertTrue(
                expr=path.exists(path=dir_path), msg="如果这个目录不存在则开始验证"
            )

        else:
            # 当这个目录已存在在创建时返回文件名称
            return_path = ensure_directory_exists(dir_path)
            self.assertEqual(
                first=return_path,
                second=path.join(
                    dir_path,
                    "s2data.npz",
                ),
                msg="在创建时返回文件名称",
            )

    def test_save_npy(self) -> None:
        """测试用于将数据保存为npy格式的函数"""
        # 测试正常的保存地址
        status = save_npy(data=self.data, save_path=self.npy_path)
        self.assertTrue(expr=status, msg="加载正确的保存地址")

    def test_load_npy(self) -> None:
        """测试用于加载npy格式的数据的函数"""
        # 测试正确的加载地址
        data = load_npy(data_path=self.npy_path)

        for key in self.data.keys():
            # 判断数据的内容是否一致
            self.assertEqual(
                first=data[key], second=self.data[key], msg="保存字典中的数据不一致"
            )

    def test_save_npz(self) -> None:
        """测试用于将数据保存为npz格式的函数"""
        # 测试正常的保存地址
        status = save_npz(data=self.data, save_path=self.npz_path)
        self.assertTrue(expr=status, msg="加载正确的保存地址")

    def test_load_npz(self) -> None:
        """测试用于加载npz格式的数据的函数"""
        # 测试正确的加载地址
        data = load_npz(data_path=self.npz_path)

        for key in self.data.keys():
            # 判断数据的内容是否一致
            self.assertEqual(
                first=data[key], second=self.data[key], msg="保存字典中的数据不一致"
            )

    def test_save_s2data(self) -> None:
        """测试用于保存S2数据的函数"""
        # 测试将数据保存为npy格式
        status = save_s2data(
            save_path=self.s2_npy_path,
            symbol=self.data["symbol"],
            excitation=self.data["excitation"],
            response=self.data["response"],
        )
        self.assertTrue(expr=status, msg="加载正确的保存地址")

        # 测试将数据保存为npz格式
        status = save_s2data(
            save_path=self.s2_npz_path,
            symbol=self.data["symbol"],
            excitation=self.data["excitation"],
            response=self.data["response"],
        )
        self.assertTrue(expr=status, msg="加载正确的保存地址")

    def test_load_s2data(self) -> None:
        """测试用于加载S2数据的函数"""
        # 测试加载npy格式的数据
        symbol, excitation, response = load_s2data(data_path=self.s2_npy_path)

        # 验证内部数据是否一致
        self.assertEqual(
            first=symbol, second=self.data["symbol"], msg="读取的符号表达式数据不相等!"
        )
        self.assertEqual(
            first=excitation,
            second=self.data["excitation"],
            msg="读取的激励时间序列数据不相等!",
        )
        self.assertEqual(
            first=response,
            second=self.data["response"],
            msg="读取的响应时间序列数据不相等!",
        )


if __name__ == "__main__":
    data = load_npy(data_path="./data/data.npy")
    print(data)

    unittest.main()
