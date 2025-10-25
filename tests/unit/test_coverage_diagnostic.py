import os
import sys

from src.execution import order_models
from src.tuning import optimizers


def test_module_paths_are_expected():
    # Ensure pytest + coverage import the same files we inspect in htmlcov
    om_path = os.path.normcase(order_models.__file__)
    opt_path = os.path.normcase(optimizers.__file__)

    assert om_path.endswith(os.path.normcase(os.path.join('src', 'execution', 'order_models.py')))
    assert opt_path.endswith(os.path.normcase(os.path.join('src', 'tuning', 'optimizers.py')))
