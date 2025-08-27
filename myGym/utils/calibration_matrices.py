from numpy import random

calibration_grid = [[[0.30, -0.20, 0.072],
                     [0.30, -0.15, 0.070],
                     [0.30, -0.10, 0.067],
                     [0.30, -0.05, 0.066],
                     [0.30, 0.00, 0.065],
                     [0.30, 0.05, 0.067],
                     [0.30, 0.10, 0.070],
                     [0.30, 0.15, 0.069],
                     [0.30, 0.20, 0.071]],
                    [[0.35, -0.20, 0.077],
                     [0.35, -0.15, 0.074],
                     [0.35, -0.10, 0.073],
                     [0.35, -0.05, 0.073],
                     [0.35, 0.00, 0.071],
                     [0.35, 0.05, 0.072],
                     [0.35, 0.10, 0.073],
                     [0.35, 0.15, 0.073],
                     [0.35, 0.20, 0.077]],
                    [[0.40, -0.20, 0.083],
                     [0.40, -0.15, 0.080],
                     [0.40, -0.10, 0.076],
                     [0.40, -0.05, 0.075],
                     [0.40, 0.00, 0.075],
                     [0.40, 0.05, 0.074],
                     [0.40, 0.10, 0.077],
                     [0.40, 0.15, 0.080],
                     [0.40, 0.20, 0.083]],
                    [[0.45, -0.20, 0.091],
                     [0.45, -0.15, 0.087],
                     [0.45, -0.10, 0.085],
                     [0.45, -0.05, 0.084],
                     [0.45, 0.00, 0.084],
                     [0.45, 0.05, 0.085],
                     [0.45, 0.10, 0.085],
                     [0.45, 0.15, 0.087],
                     [0.45, 0.20, 0.095]]]

grasp_calibration_grid_right = [[[0.30, -0.20, 0.109],
                                 [0.30, -0.15, 0.107],
                                 [0.30, -0.10, 0.105],
                                 [0.30, -0.05, 0.104],
                                 [0.30, 0.00, 0.104],
                                 [0.30, 0.05, 0.104],
                                 [0.30, 0.10, 0.107],
                                 [0.30, 0.15, 0.11],   #
                                 [0.30, 0.20, 0.105]],
                                [[0.35, -0.20, 0.077],
                                 [0.35, -0.15, 0.074],
                                 [0.35, -0.10, 0.073],
                                 [0.35, -0.05, 0.073],
                                 [0.35, 0.00, 0.071],
                                 [0.35, 0.05, 0.072],
                                 [0.35, 0.10, 0.073],
                                 [0.35, 0.15, 0.073],
                                 [0.35, 0.20, 0.077]],
                                [[0.40, -0.20, 0.083],
                                 [0.40, -0.15, 0.080],
                                 [0.40, -0.10, 0.076],
                                 [0.40, -0.05, 0.075],
                                 [0.40, 0.00, 0.075],
                                 [0.40, 0.05, 0.074],
                                 [0.40, 0.10, 0.077],
                                 [0.40, 0.15, 0.080],
                                 [0.40, 0.20, 0.083]],
                                [[0.45, -0.20, 0.124],
                                 [0.45, -0.15, 0.087],
                                 [0.45, -0.10, 0.085],
                                 [0.45, -0.05, 0.084],
                                 [0.45, 0.00, 0.084],
                                 [0.45, 0.05, 0.085],
                                 [0.45, 0.10, 0.085],
                                 [0.45, 0.15, 0.087],
                                 [0.45, 0.20, 0.095]]]

class TargetGrid:
    def __init__(self):
        self.x_start, self.x_end, self.x_step = 0.37, 0.51, 0.02
        self.y_start, self.y_end, self.y_step = -0.2, 0.2, 0.05
        self.z_start, self.z_end, self.z_step = 0.02, 0.02, 0.02
        
        self.x = self.x_start - self.x_step  # Start before the first value
        self.y = self.y_start
        self.z = self.z_start
        self.first_pass = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.x < self.x_end:
            self.x += self.x_step
        elif self.y < self.y_end:
            self.y += self.y_step
            self.x = self.x_start
        elif self.z < self.z_end:
            self.z += self.z_step
            self.y = self.y_start
            self.x = self.x_start
        else:
            if not self.first_pass:
                raise StopIteration
            self.first_pass = False

        return round(self.x, 2), round(self.y, 2), round(self.z, 3)

class TargetGridTiagoTable():
    def __init__(self, x_start, x_end, x_step, y_start, y_end, y_step, z_start, z_end, z_step):
        self.x_start = x_start
        self.x_end = x_end
        self.x_step = x_step
        self.y_start = y_start
        self.y_end = y_end
        self.y_step = y_step
        self.z_start = z_start
        self.z_end = z_end
        self.z_step = z_step

        self.x = self.x_start - self.x_step  # Start before the first value
        self.y = self.y_start
        self.z = self.z_start
        self.first_pass = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.x < self.x_end:
            self.x += self.x_step
        elif self.y < self.y_end:
            self.y += self.y_step
            self.x = self.x_start
        elif self.z < self.z_end:
            self.z += self.z_step
            self.y = self.y_start
            self.x = self.x_start
        else:
            if not self.first_pass:
                raise StopIteration
            self.first_pass = False

        return round(self.x, 2), round(self.y, 2), round(self.z, 3)

def target_experiment(index):
    calibration_matrix = [[0.45, -.05, 0.07],
                          [0.42, -0.0, 0.07],
                          [0.50, 0.05, 0.07],
                          [0.65, 0.03, 0.085],
                          [0.58, -.08, 0.09],
                          [0.43, -.14, 0.06],
                          [0.36, -.075, 0.035]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]

def target_old_experiment(index):
    calibration_matrix = [[0.45, -.05, 0.07],
                          [0.42, -0.0, 0.07],
                          [0.50, 0.05, 0.07],
                          [0.65, 0.03, 0.085],
                          [0.58, -.08, 0.09],
                          [0.43, -.14, 0.06],
                          [0.36, -.075, 0.035]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_calibration_grid(index):
    index %= len(calibration_grid) * len(calibration_grid[0])
    i = int(index / len(calibration_grid[0]))
    j = int(index % len(calibration_grid[0]))

    return calibration_grid[i][j]

def target_random():
    target_position = [0.30 + (0.15 * random.rand()), -0.20 + (0.40 * random.rand()),
                       0.08]  # Write your own method for end effector position here
    # return [0.25, -0.2, 0.15]
    return target_position

def target_joints(index):
    calibration_matrix = [[27.502, 90, 34.999, 114.998, 90., -22.5],
                          [16.997, 79.502, 29.449, 125.503, 89.764, -32.994],
                          [10.99, 68.98, 29.452, 136.007, 89.762, -40.419],
                          [10.98, 58.476, 29.439, 138.99, 89.761, -40.423],
                          [10.98, 47.733, 29.438, 138.991, 89.761, -40.423]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]

def target_calibration(index):
    calibration_matrix = [[0.365, -0.260, 0.04],
                          [0.365, -0.230, 0.04],
                          [0.365, -0.210, 0.04],
                          [0.365, -0.180, 0.04],
                          [0.365, -0.150, 0.04],
                          [0.365, -0.120, 0.030],
                          [0.365, -0.090, 0.030],
                          [0.365, -0.06, 0.030],
                          [0.365, -0.03, 0.030],
                          [0.365, 0.0, 0.030],
                          [0.365, 0.03, 0.030],
                          [0.365, 0.07, 0.030]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_point(index):
    calibration_matrix = [[0.300, -0.200, 0.043], [0.375, 0.000, 0.043], [0.450, 0.200, 0.043]]

    index %= len(calibration_matrix)
    return calibration_matrix[index]


def target_rectangle(index):
    calibration_matrix = [[0.365, -0.260, 0.043],
                          [0.425, -0.260, 0.043],
                          [0.485, -0.260, 0.043],
                          [0.485, -0.130, 0.043],
                          [0.485, 0.000, 0.043],
                          [0.425, 0.00, 0.043],
                          [0.365, 0.00, 0.043],
                          [0.365, -0.130, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_line_vertical_1(index):
    calibration_matrix = [[0.300, -0.200, 0.043],
                          [0.315, -0.200, 0.043],
                          [0.330, -0.200, 0.043],
                          [0.345, -0.200, 0.043],
                          [0.360, -0.200, 0.043],
                          [0.375, -0.200, 0.043],
                          [0.390, -0.200, 0.043],
                          [0.405, -0.200, 0.043],
                          [0.420, -0.200, 0.043],
                          [0.435, -0.200, 0.043],
                          [0.450, -0.200, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_line_vertical_2(index):
    calibration_matrix = [[0.300, 0.000, 0.043],
                          [0.315, 0.000, 0.043],
                          [0.330, 0.000, 0.043],
                          [0.345, 0.000, 0.043],
                          [0.360, 0.000, 0.043],
                          [0.375, 0.000, 0.043],
                          [0.390, 0.000, 0.043],
                          [0.405, 0.000, 0.043],
                          [0.420, 0.000, 0.043],
                          [0.435, 0.000, 0.043],
                          [0.450, 0.000, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_line_vertical_3(index):
    calibration_matrix = [[0.300, 0.20, 0.043],
                          [0.315, 0.20, 0.043],
                          [0.330, 0.20, 0.043],
                          [0.345, 0.20, 0.043],
                          [0.360, 0.20, 0.043],
                          [0.375, 0.20, 0.043],
                          [0.390, 0.20, 0.043],
                          [0.405, 0.20, 0.043],
                          [0.420, 0.20, 0.043],
                          [0.435, 0.20, 0.043],
                          [0.450, 0.20, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_line_horizontal_1(index):
    calibration_matrix = [[0.350, -0.200, 0.043],
                          [0.350, -0.175, 0.043],
                          [0.350, -0.150, 0.043],
                          [0.350, -0.125, 0.043],
                          [0.350, -0.100, 0.043],
                          [0.350, -0.075, 0.043],
                          [0.350, -0.050, 0.043],
                          [0.350, -0.025, 0.043],
                          [0.350, -0.000, 0.043],
                          [0.350, 0.025, 0.043],
                          [0.350, 0.050, 0.043],
                          [0.350, 0.075, 0.043],
                          [0.350, 0.100, 0.043],
                          [0.350, 0.125, 0.043],
                          [0.350, 0.150, 0.043],
                          [0.350, 0.175, 0.043],
                          [0.350, 0.200, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_line_horizontal_2(index):
    calibration_matrix = [[0.400, -0.200, 0.043],
                          [0.400, -0.175, 0.043],
                          [0.400, -0.150, 0.043],
                          [0.400, -0.125, 0.043],
                          [0.400, -0.100, 0.043],
                          [0.400, -0.075, 0.043],
                          [0.400, -0.050, 0.043],
                          [0.400, -0.025, 0.043],
                          [0.400, -0.000, 0.043],
                          [0.400, 0.025, 0.043],
                          [0.400, 0.050, 0.043],
                          [0.400, 0.075, 0.043],
                          [0.400, 0.100, 0.043],
                          [0.400, 0.125, 0.043],
                          [0.400, 0.150, 0.043],
                          [0.400, 0.175, 0.043],
                          [0.400, 0.200, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_line_diagonal_1(index):
    calibration_matrix = [[0.300, -0.200, 0.043],
                          [0.315, -0.160, 0.043],
                          [0.330, -0.120, 0.043],
                          [0.345, -0.080, 0.043],
                          [0.360, -0.040, 0.043],
                          [0.375, 0.000, 0.043],
                          [0.390, 0.040, 0.043],
                          [0.405, 0.080, 0.043],
                          [0.420, 0.120, 0.043],
                          [0.435, 0.160, 0.043],
                          [0.450, 0.200, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_line_diagonal_2(index):
    calibration_matrix = [[0.300, 0.200, 0.043],
                          [0.315, 0.160, 0.043],
                          [0.330, 0.120, 0.043],
                          [0.345, 0.080, 0.043],
                          [0.360, 0.040, 0.043],
                          [0.375, 0.000, 0.043],
                          [0.390, -0.040, 0.043],
                          [0.405, -0.080, 0.043],
                          [0.420, -0.120, 0.043],
                          [0.435, -0.160, 0.043],
                          [0.450, -0.200, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]


def target_v(index):
    calibration_matrix = [[0.325, -0.200, 0.043],
                          [0.340, -0.175, 0.043],
                          [0.355, -0.150, 0.043],
                          [0.370, -0.125, 0.043],
                          [0.385, -0.100, 0.043],
                          [0.400, -0.075, 0.043],
                          [0.415, -0.050, 0.043],
                          [0.430, -0.025, 0.043],
                          [0.445, 0.000, 0.043],
                          [0.430, 0.025, 0.043],
                          [0.415, 0.050, 0.043],
                          [0.400, 0.075, 0.043],
                          [0.385, 0.100, 0.043],
                          [0.370, 0.125, 0.043],
                          [0.355, 0.150, 0.043],
                          [0.340, 0.175, 0.043],
                          [0.325, 0.200, 0.043]]

    if index >= len(calibration_matrix):
        index = 1
    return calibration_matrix[index]
