# -*- coding: utf-8 -*-
"""
用于实时打印输出数据生成状态的模块

Created on 2025/08/23 12:20:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import time
import os
from os import path

import numpy as np
from colorama import Fore, Style
from typing import Optional, List, Tuple

from S2Generator import SeriesParams, SymbolParams, Node, NodeList
from S2Generator.utils import get_time_now


class PrintStatus(object):
    """"""

    def __init__(
        self,
        series_params: SeriesParams,
        symbol_params: SymbolParams,
        logging_path: Optional[str] = None,
    ) -> None:
        """
        Status information for real-time printing of data generation.

        :param series_params: The parameters controlling the generation of the excitation time series.
        :param symbol_params: The parameters controlling the generation of the symbolic expressions.
        :param logging_path: The path of the logging folder or file, if is file please end with `.txt`.
        """
        # The parameters controlling the generation
        self.series_params = series_params
        self.symbol_params = symbol_params

        # Get a variety of different sampling methods and sampling probabilities
        self.sampling_methods = series_params.sampling_methods
        self.prob_array = series_params.prob_array

        # Record the most basic information in data generation
        self.basic_header = "Basic Config of The S2Generator:"
        self.basic_config = (
            f'  {f"{self.sampling_methods[0]}:":<20} {self.prob_array[0]:<20}'
            f'{f"{self.sampling_methods[1]}:":<20} {self.prob_array[1]:<20}\n  '
            f'{f"{self.sampling_methods[2]}:":<20} {self.prob_array[2]:<20}'
            f'{f"{self.sampling_methods[3]}:":<20} {self.prob_array[3]:<20}\n  '
            f'{f"{self.sampling_methods[4]}:":<20} {self.prob_array[4]:<20}\n  '
            f'{f"Min Binary Operator:":<20} {self.symbol_params.min_binary_ops_per_dim:<20}'
            f'{f"Max Binary Operator:":<20} {self.symbol_params.max_binary_ops_per_dim:<20}\n  '
            f'{f"Min Unary Operator:":<20} {self.symbol_params.min_unary_ops:<20}'
            f'{f"Max Unary Operator:":<20} {self.symbol_params.max_unary_ops:<20}\n  '
            f'{f"Max Trials:":<20} {self.symbol_params.max_trials:<20}'
            f'{f"Solve Diff":<20} {self.symbol_params.solve_diff:<20}\n  '
            f'{f"Probability Random:":<20} {self.symbol_params.prob_rand:<20}'
            f'{f"Probability Const:":<20} {self.symbol_params.prob_const:<20}\n'
        )

        # Create a logging txt file and start recording the status information of data generation
        self.logging_path = logging_path
        if self.logging_path is None:
            self.logging = False
        else:
            self.logging = True

            # Determine whether the user has specified a file name
            if self.logging_path.endswith(".txt") or self.logging_path.endswith(".log"):
                self.file_path = self.logging_path
            else:
                # Manually add the name of the registration file
                self.file_path = path.join(self.logging_path, "status.txt")
            self.logging_basis_config()

        # Record the status of data generation and update parameters in real time
        self.generation_header = "Generation Config of The S2Generator:"
        self.generation_config = None

        # Record the status information list in each update
        self.process_header = "The Specific Execution Process of S2Generator:"
        self.status_list = []

        # 用于打印实时更新的表格
        self.header = ["Index", "Target", "Time", "Results"]
        self.max_length = [5, 36, 19, 7]
        self.header_pr = f" {self.header[0]} | {self.header[1]}{' ' * 30} | {self.header[2]}{' ' * 14}  | {self.header[3]}"
        self.sep = ""
        for length in self.max_length:
            self.sep += f"{'-' * (length + 2)}+"
        self.sep = self.sep[:-1] + "-"

        self._index = 0
        self._count = 0

        # 与算法执行时间相关的参数
        self.start_time, self.end_time = None, None

    def reset(self):
        """重置打印状态的相关内容和信息"""
        self._index = 0
        self._count = 0

        self.status_list = []

        self.start_time, self.end_time = None, None

    @property
    def index(self):
        return self._index

    def get_next_index(self):
        self._index += 1
        return self.index

    @property
    def count(self) -> int:
        return self._count

    def get_next_count(self):
        self._count += 1
        return self.count

    def logging_basis_config(self) -> None:
        """实时登记"""
        if self.logging:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(self.basic_header + "\n")
                f.write(self.basic_config + "\n")

    def show_start(
        self,
        n_inputs_points: int,
        input_dimension: int,
        output_dimension: int,
        max_trials: int,
        input_normalize: Optional[str] = "z-score",
        output_normalize: Optional[bool] = "z-score",
        input_max_scale: Optional[float] = 16.0,
        output_max_scale: Optional[float] = 16.0,
        offset: Optional[Tuple[float, float]] = None,
    ) -> None:
        """打印有关数据生成过程中实时参数的基本信息"""
        # Record the time when program execution starts
        self.start_time = time.time()

        print(Style.BRIGHT + Fore.GREEN + self.basic_header + Style.RESET_ALL)
        print(self.basic_config)

        print(Style.BRIGHT + Fore.GREEN + self.generation_header + Style.RESET_ALL)
        self.generation_config = (
            f'  {"Time Series Length:":<20} {n_inputs_points:<20}'
            f'  {"Max Trials:":<20} {str(max_trials):<20}\n'
            f'  {"Input Dimension:":<20} {input_dimension:<20}'
            f'  {"Output Dimension:":<20} {output_dimension:<20}\n'
            f'  {"Input Normalization:":<20} {input_normalize:<20}'
            f'  {"Output Normalization:":<20} {output_normalize:<20}\n'
            f'  {"Input Max Scale:":<20} {input_max_scale:<20}'
            f'  {"Output Max Scale:":<20} {output_max_scale:<20}\n'
            f'  {"Offset":<20} {str(offset):<20}\n'
        )
        print(self.generation_config)

        self.logging_generation_config()

    def logging_generation_config(self) -> None:
        if self.logging:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(self.generation_header + "\n")
                f.write(self.generation_config + "\n")

    def print_new_status(self, status: List[str]) -> None:
        """打印最新上传的状态信息"""
        message = ""
        for s in status:
            message += f" {s} |"
        message = message[:-1]

        print(message)
        self.status_list.append(message)

    def update_symbol(self, status: str) -> None:
        """更新生成符号表达式的状态信息"""
        print(Style.BRIGHT + Fore.GREEN + self.process_header + Style.RESET_ALL)
        self.status_list.append(self.process_header)
        print("-" * len(self.sep))
        print(self.header_pr)
        print(self.sep)

        # Update the information to the list of status
        self.status_list.append("-" * len(self.sep))
        self.status_list.append(self.header_pr)
        self.status_list.append(self.sep)

        self.print_new_status(
            status=self.strip(
                status=[
                    str(self.get_next_index()),
                    "Create the Symbolic Expression",
                    get_time_now(),
                    check_status(status=status),
                ]
            )
        )

    def update_excitation(self, status: str) -> None:
        """更新生成激励时间序列数据的状态信息"""
        self.print_new_status(
            status=self.strip(
                status=[
                    str(self.get_next_index()),
                    f"Generate Excitation Time Series {self.get_next_count()}",
                    get_time_now(),
                    check_status(status=status),
                ]
            )
        )

    def update_response(self, status: str) -> None:
        """更新生成响应时间序列数据的状态信息"""
        self.print_new_status(
            status=self.strip(
                status=[
                    str(self.get_next_index()),
                    f"Generate Response Time Series {self.count}",
                    get_time_now(),
                    check_status(status=status),
                ]
            )
        )

    def strip(self, status: List[str]) -> List[str]:
        """根据输入的字符串对齐内容进行调整"""
        status[0] = status[0] + (self.max_length[0] - len(status[0])) * " "
        status[1] = status[1] + (self.max_length[1] - len(status[1])) * " "

        return status

    def show_end(self, symbol: Node | NodeList | str) -> None:
        """打印算法执行过程中表格的结尾信息"""
        print("-" * len(self.sep) + "\n")

        # 将结尾信息添加到状态列表中
        self.status_list.append("-" * len(self.sep))

        # 打印并上传的符号表达式
        trees = str(symbol).split(" | ")
        print(
            Style.BRIGHT
            + Fore.GREEN
            + "The Generated Symbolic Expression: "
            + Style.RESET_ALL
        )

        # Record the time the program ends
        self.end_time = time.time()

        # 计算程序执行的总时间
        running_time = round(self.end_time - self.start_time, 5)

        if self.logging:
            # 上传数据生成过程中的所有报告
            with open(self.file_path, "a", encoding="utf-8") as f:
                for status in self.status_list:
                    f.write(
                        status.replace("[1m[32m", "")
                        .replace("[31m", "")
                        .replace("[0m", "")
                        + "\n"
                    )

                f.write("\n" + "The Generated Symbolic Expression: \n")
                for tree in trees:
                    f.write(tree + "\n")
                    print(tree)

                # 打印并上传程序执行的具体时间
                print(
                    Style.BRIGHT
                    + Fore.GREEN
                    + "\nRunning Time: \n"
                    + Style.RESET_ALL
                    + str(running_time)
                    + "\n"
                )
                f.write(f"\nRunning Time: \n{running_time}\n")
        else:
            for tree in trees:
                print(tree)

            # 打印程序执行的具体时间
            print(
                Style.BRIGHT
                + Fore.GREEN
                + "\nRunning Time: \n"
                + Style.RESET_ALL
                + str(running_time)
                + "\n"
            )


def check_status(status: str) -> str:
    """
    Check the status feedback information during the data generation process.

    :param status: The finish status for the data generation.
    :return: the feedback information with color in colorama.
    """
    if status == "success":
        return Style.BRIGHT + Fore.GREEN + status + Style.RESET_ALL
    elif status == "failure":
        return Fore.RED + status + Style.RESET_ALL
    else:
        raise ValueError("Invalid status!")


if __name__ == "__main__":
    print_state = PrintStatus(
        logging_path="../../data",
        series_params=SeriesParams(),
        symbol_params=SymbolParams(),
    )
    print_state.show_start(
        n_inputs_points=100,
        input_dimension=2,
        output_dimension=2,
        max_trials=100,
        input_normalize="z-score",
    )
    print(print_state.logging)

    print(get_time_now())

    # 2025-08-28 18:52:34
    print_state.update_symbol("success")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("failure")
    print_state.update_excitation(status="success")
    print_state.update_response("success")

    print_state.show_end("hello")
