# -*- coding: utf-8 -*-
"""
Created on 2025/08/13 23:47:51
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from S2Generator.incentive import ARMA


class TestARMA(unittest.TestCase):
    """测试用于生成激励时间序列数据的ARMA模块"""

    # 用于测试的随机数生成器
    rng = np.random.RandomState(42)

    # 用于测试的实例对象
    arma = ARMA()

    def test_setup(self) -> None:
        """测试模块的创建过程"""
        for p_max in [2, 3, 4, 5]:
            for q_max in [2, 3, 4, 5]:
                for upper_bound in [100, 200, 300, 400]:
                    # 构建激励时间序列生成器
                    arma = ARMA(p_max, q_max)
                    self.assertIsInstance(
                        arma, cls=ARMA, msg="Wrong ARMA type in `test_setup` method"
                    )

    def test_create_autoregressive_params(self) -> None:
        """测试能否正常生成自回归过程的参数"""
        for p_order in [1, 2, 3, 4]:
            # 遍历不同的阶数来生成参数
            p_params = self.arma.create_autoregressive_params(rng=self.rng, p_order=p_order)

            # 检验参数的长度
            self.assertEquals(len(p_params), p_order, msg="自回归过程的参数长度错误!")
            self.assertIsInstance(p_params, np.ndarray, msg="自回归过程的参数类型错误!")

            # 检查自回归过程的参数范围是否符合约束
            self.assertTrue(np.sum(p_params) < 1, msg="自回归过程的参数求和没有小于1!")
            self.assertTrue(np.abs(p_params[-1]) < 1, msg="自回归过程的最后一个参数的绝对值没有小于1!")

    def test_create_moving_average_params(self) -> None:
        """测试能否正常生成滑动平均过程的参数"""
        for q_order in [1, 2, 3, 4, 5]:
            # 遍历不同的阶数来生成参数
            q_params = self.arma.create_autoregressive_params(rng=self.rng, p_order=q_order)

            # 检验参数的长度
            self.assertEquals(len(q_params), q_order, msg="滑动平均过程的参数长度错误!")
            self.assertIsInstance(q_params, np.ndarray, msg="滑动平均过程的参数类型错误!")

    def test_create_params(self) -> None:
        """测试能否正常生成ARAM模型的参数"""
        # 执行创建参数的方法
        self.arma.create_params(rng=self.rng)

        # 通过模型的阶数和参数数组大小进行验证
        p_order = self.arma.p_order
        q_order = self.arma.q_order

        self.assertEquals(first=p_order, second=len(self.arma.p_params), msg="自回归过程的阶数与生成的参数不匹配!")
        self.assertEquals(first=q_order, second=len(self.arma.q_params), msg="滑动平均过程的阶数与生成的参数不匹配!")

    def test_order(self) -> None:
        """测试尝试获取模型阶数的功能"""
        # 执行创建参数的方法
        self.arma.create_params(rng=self.rng)

        # 获取模型的阶数
        order_dict = self.arma.order

        # 测试字典的数据类型
        self.assertIsInstance(obj=order_dict, cls=dict, msg="测试阶数的函数返回了错误的数据类型!")

        # 遍历字典测试数据类型
        for key, value in order_dict.items():
            self.assertIsInstance(obj=key, cls=str, msg="返回内容错误!")
            self.assertIsInstance(obj=value, cls=int, msg="返回内容错误!")

    def test_params(self) -> None:
        """测试尝试获取模型参数的功能"""
        # 执行创建参数的方法
        self.arma.create_params(rng=self.rng)

        # 获取模型的参数
        params_dict = self.arma.params

        # 测试字典的数据类型
        self.assertIsInstance(obj=params_dict, cls=dict, msg="测试参数的函数返回了错误的数据类型!")

        # 遍历字典测试数据类型
        for key, value in params_dict.items():
            self.assertIsInstance(obj=key, cls=str, msg="返回内容错误!")
            self.assertIsInstance(obj=value, cls=np.ndarray, msg="返回内容错误!")

    def test_generate(self) -> None:
        """测试激励时间序列数据能否正确生成"""

    def test_call(self) -> None:
        """测试数据生成类的响应"""

    def test_str(self) -> None:
        """测试获取字符串描述的魔术方法"""
