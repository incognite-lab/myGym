"""
Module for performing color name / rgba conversions and drawing random colors
"""
from typing import List

import numpy as np

SEED = 0
NUMBER_OF_RGB_CHANNELS = 3
COLOR_DICT = {
    "dark green": (0, 0.4, 0, 0.5),
    "green": (0, 0.8, 0, 1),
    "blue": (0.5, 0.8, 1, 1),
    "gray": (0.2, 0.2, 0.2, 1),
    "transparent black": (0, 0, 0, 0.5),
    "cyan": (0.0, 1.0, 1.0, 1),
    "orange": (1.0, 0.65, 0.0, 1),
    "red": (1, 0, 0, 1),
    "pink": (1, 0.41, 0.7, 1),
}

COLOR_VALUE_LIST = list(COLOR_DICT.values())
REVERSED_COLOR_DICT = {v: k for k, v in COLOR_DICT.items()}
rng = np.random.default_rng(SEED)


def get_all_colors(excluding=None):
    """
    Draw a random color from the color dictionary

    Parameters:
        :param excluding: (list) List of colors to exclude
    Returns:
        :return rgba: (tuple) tuple of rgba channels
    """
    return [v for k, v in COLOR_DICT.items() if k not in excluding]


def get_random_rgba(excluding=None):
    """
    Draw a random color from the color dictionary

    Parameters:
        :param excluding: (list) List of colors to exclude
    Returns:
        :return rgba: (tuple) tuple of rgba channels
    """
    return tuple(rng.choice(COLOR_VALUE_LIST)) if excluding is None else tuple(rng.choice(get_all_colors(excluding)))


def name_to_rgba(name: str) -> List:
    """
    For a given color name find corresponding rgba values

    Parameters:
        :param name: (string) Color name
    Returns:
        :return rgba: (tuple) tuple of rgba channels
    """
    return COLOR_DICT[name]


def rgba_to_name(rgba: List) -> str:
    """
    For given rgba values find the corresponding color name

    Parameters:
        :param rgba: (tuple) tuple of rgba channels
    Returns:
        :return name: (string) Color name
    """
    return REVERSED_COLOR_DICT[rgba]
