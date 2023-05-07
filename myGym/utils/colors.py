"""
Module for performing color name / rgba conversions and drawing random colors
"""
from typing import List

import numpy as np

SEED = 0
TRANSPARENCY_VALUE = 0.4
NUMBER_OF_RGB_CHANNELS = 3
OPAQUE_COLOR_DICT = {
    "green": (0, 0.8, 0, 1),
    "blue": (0.5, 0.8, 1, 1),
    "gray": (0.2, 0.2, 0.2, 1),
    "cyan": (0.0, 1.0, 1.0, 1),
    "orange": (1.0, 0.65, 0.0, 1),
    "red": (1, 0, 0, 1),
    "pink": (1, 0.41, 0.7, 1),
}
TRANSPARENT_COLOR_DICT = {"transparent " + k: (v[0], v[1], v[2], TRANSPARENCY_VALUE) for k, v in OPAQUE_COLOR_DICT.items()}
COLOR_DICT = {**OPAQUE_COLOR_DICT, **TRANSPARENT_COLOR_DICT}
REVERSED_COLOR_DICT = {v: k for k, v in COLOR_DICT.items()}
rng = np.random.default_rng(SEED)


def get_all_colors(transparent=False, excluding=None):
    """
    Form a list of available colors.

    Parameters:
        :param transparent: (list) Whether the colors should be transparent
        :param excluding: (list) List of colors to exclude
    Returns:
        :return rgba: (tuple) tuple of rgba channels
    """
    excluding = [] if excluding is None else excluding
    return [v for k, v in (OPAQUE_COLOR_DICT if not transparent else TRANSPARENT_COLOR_DICT).items() if k not in excluding]


def draw_random_rgba(size=None, replace=False, transparent=False, excluding=None):
    """
    Draw a random color from the color dictionary.

    Parameters:
        :param size: (int) If any other than None, then it defines the resulting list size
        :param replace: (bool) Whether to draw with a replacement. It would raise an exception if the number of colors is not enough
        :param transparent: (bool) Whether the colors should be transparent
        :param excluding: (list) List of colors to exclude
    Returns:
        :return rgba: (tuple or list) tuple(s) of rgba channels
    """
    colors = get_all_colors(transparent, excluding)
    if size is not None and len(colors) < size:
        msg = f"The size argument ({size}) is greater than the actuaal number of available colors ({len(colors)})"
        raise Exception(msg)
    rgba = rng.choice(colors, size=size, replace=replace)
    return tuple(rgba) if size is None else [tuple(r) for r in rgba]


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
