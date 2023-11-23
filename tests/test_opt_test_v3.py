"""
unittest file for the optimization function
"""
import os
import inspect
import numpy as np
import pytest
from src.opt_test_v3 import Optimizer


config = {
    'use_path': f'{os.getcwd()}/data/use_data.csv',
    'row': 10,
    'post_cnt_lst': [1, 1, 1, 1],
    'spec_kols_lst': [1, 2, 3, 4],
    'label_path': f'{os.getcwd()}/data/preproc_data_v3.csv',
    'budget': 1000,
    'voice': 5000,
    'lamb': 0.0,
    'alpha': 0.0
}

@pytest.fixture
def optimizer_instance() -> Optimizer:
    """
    Optimizer instance
    """
    return Optimizer(**config)


def test_get_functions(optimizer_instance) -> None:
    """
    Check functions
    """
    func, _ = optimizer_instance.get_functions('plain')
    assert inspect.ismethod(func)


def test_run_opt(optimizer_instance) -> None:
    """
    Check the run optimization function
    """
    _, arr = optimizer_instance.run_opt(
        optimizer_instance.get_functions("plain")[0],
        optimizer_instance.get_functions("plain")[1]
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (int(optimizer_instance.row) * 4, )


def test_get_opt_result(optimizer_instance) -> None:
    """
    Check the get optimization result function
    """
    cost, voice, _ = optimizer_instance.get_opt_result(
        optimizer_instance.run_opt(
            optimizer_instance.get_functions("plain")[0],
            optimizer_instance.get_functions("plain")[1]
        )[0]
    )
    assert isinstance(cost, float)
    assert isinstance(voice, float)
    assert cost == 778.0
    assert int(voice) == 6743


if __name__ == "__main__":
    pytest.main('-v', 'test_opt_test_v3.py')
