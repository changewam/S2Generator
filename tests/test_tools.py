# -*- coding: utf-8 -*-
"""
Created on 2025/08/23 17:17:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""
import unittest
import os
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
    z_score_normalization,
    max_min_normalization,
)


class TestTools(unittest.TestCase):
    """对工具模块的函数进行单元测试"""

    # 构建输入序列
    time_series = np.random.uniform(low=-10, high=10, size=(256, 10))

    # 构建全0时间序列
    zero_series = np.zeros((256, 10))

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

        self.assertTrue(
            expr=np.isnan(normalized).all(),
            msg="全零时间序列经过z-score标准化后没有出现NaN值!",
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
        """测试全零输入的max-min归一化算法"""
        # 执行标准化算法
        normalized = max_min_normalization(self.zero_series)

        self.assertTrue(
            expr=np.isnan(normalized).all(),
            msg="全零时间序列经过max-min标准化后没有出现NaN值!",
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




if __name__ == "__main__":
    unittest.main()
