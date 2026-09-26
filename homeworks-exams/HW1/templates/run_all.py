#!/usr/bin/env python3
"""Reproduce every number and figure in the paper with one command:

    python run_all.py

Keep this file short: it only calls the steps, in order. Each step writes its
outputs to output/, and the paper and slides read only from there.
Pin package versions in requirements.txt.
"""

import pathlib
import time

OUT = pathlib.Path("output")
OUT.mkdir(exist_ok=True)


def get_data():
    """Download (or read the cached copy of) every data series.
    Record the series IDs and the download date in output/data_vintage.txt."""
    raise NotImplementedError


def data_moments():
    """HP-filtered moments of the data -> output/table_data_moments.tex"""
    raise NotImplementedError


def calibrate():
    """Targets -> parameters, with sources -> output/table_calibration.tex"""
    raise NotImplementedError


def unit_test():
    """The delta = 1 model against its closed form -> output/table_unit_test.tex"""
    raise NotImplementedError


def solve_three_ways():
    """Global, perturbation, projection; accuracy vs. cost -> output/table_methods.tex"""
    raise NotImplementedError


def answers():
    """Model vs. data moments; welfare cost -> output/table_answers.tex, figures"""
    raise NotImplementedError


if __name__ == "__main__":
    for step in (get_data, data_moments, calibrate, unit_test, solve_three_ways, answers):
        t0 = time.perf_counter()
        step()
        print(f"{step.__name__:20s} {time.perf_counter() - t0:8.2f}s")
