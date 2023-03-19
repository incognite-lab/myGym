import json

import numpy as np
import pkg_resources
import os

import webcolors as wc

CURRENT_DIR = pkg_resources.resource_filename('myGym', 'utils')
COLOR_DICT_FILE_PATH = os.path.join(CURRENT_DIR, 'sources', 'colors.json')
NUMBER_OF_RGB_CHANNELS = 3
MAX_COLOR_INTEGER_INTENSITY = 255


class Colors:
    """
    The class for manipulations with colors

    Parameters:
        :param file_path: (string) Source file which to take colors from, the class has the special method for generating this file
        :param seed: (int) Seed for numpy generator
    """
    def __init__(self, file_path=COLOR_DICT_FILE_PATH, seed=0):
        self.names = None
        self.rgba = None
        self.rng = np.random.default_rng(seed)

        with open(file_path, 'r') as file:
            color_dict = json.loads(file.read())
            self.names = list(color_dict.keys())
            self.rgba = np.array(list(color_dict.values()))

    @staticmethod
    def _add_color(color_dict, color, update_if_exists=False):
        name, rgba = color

        if name not in color_dict or update_if_exists:
            color_dict[name] = rgba
        else:
            msg = 'The color {name} already exists!'
            raise Exception(msg)

    @staticmethod
    def generate_colors(
            only_if_file_doesnt_exist=True,
            file_path=COLOR_DICT_FILE_PATH,
            source='html4',
            use_additional_colors_dict=True
    ) -> None:
        """
        Generate the color source file

        Parameters:
            :param only_if_file_doesnt_exist: (bool) Whether to override the file it already exists
            :param file_path: (string) Path to the file
            :param source: (string) Which source to use to generate the base colors, may be: 'html4', 'css3' or None (use only additional dictionary)
            :param use_additional_colors_dict: (bool) Whether to use additional dictionary defined in the end of this module
        """
        if only_if_file_doesnt_exist and os.path.isfile(file_path):
            return

        if source is None:
            color_dict = {}
        elif source == 'html4':
            color_dict = wc.HTML4_NAMES_TO_HEX
        elif source == 'css3':
            color_dict = wc.CSS3_NAMES_TO_HEX
        else:
            msg = f'Unknown source type: {source}'
            raise Exception(msg)

        color_tuples = list(map(wc.hex_to_rgb, list(color_dict.values())))
        color_rgba = [[t[i] / MAX_COLOR_INTEGER_INTENSITY for i in range(NUMBER_OF_RGB_CHANNELS)] + [1] for t in color_tuples]
        color_dict = dict(zip(color_dict.keys(), color_rgba))

        if use_additional_colors_dict:
            for color in ADDITIONAL_COLORS.items():
                Colors._add_color(color_dict, color, update_if_exists=True)

        with open(file_path, 'w') as file:
            json.dump(color_dict, file)

    def name_to_rgba(self, name: str) -> np.array:
        """
        Convert the color name to a rgba value

        Parameters:
            :param name: (string) Color name
        Returns:
            :return rgba: (list) List of rgba channels
        """
        return self.rgba[self.names.index(name)]

    def rgba_to_name(self, rgba) -> str:
        """
        Find the nearest color based on Euclidean distance

        Parameters:
            :param rgba: (list) List of rgba channels of length 4
        Returns:
            :return name: (string) Color name
        """
        differences = self.rgba - np.expand_dims(np.array(rgba), 0)
        distances = np.linalg.norm(differences, axis=1)
        return self.names[np.argmin(distances)]

    def get_random_rgba(self):
        """
        Return a random rgba value

        Returns:
            :return rgba: (list) List of rgba channels of length 4
        """
        return self.rng.choice(self.rgba)


ADDITIONAL_COLORS = {
    'dark green': [0, 0.4, 0, 0.5],
    'green': [0, 0.8, 0, 1],
    'blue': [0.5, 0.8, 1, 1],
    'gray': [0.2, 0.2, 0.2, 1],
}
