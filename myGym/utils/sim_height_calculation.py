from utils.calibration_matrices import calibration_grid
import random


def get_boundary_indexes(value, coordinate):
    i1, i2 = 0, 0

    if value >= calibration_grid[-1][-1][coordinate]:
        if coordinate == 0:
            i1 = i2 = len(calibration_grid) - 1
        else:
            i1 = i2 = len(calibration_grid[0]) - 1
    elif value > calibration_grid[0][0][coordinate]:
        min_value = calibration_grid[0][0][coordinate]
        step = calibration_grid[1 - coordinate][0 + coordinate][coordinate] - min_value
        i1 = int((value - min_value) / step)
        i2 = i1 + 1

    return i1, i2


def get_middle_z(value_start, value_end, value, z_start, z_end):
    z_gap = z_end - z_start
    value_gap = value_end - value_start
    value_diff_from_start = value - value_start

    if value_gap == 0:
        ratio = 0
    else:
        ratio = value_diff_from_start / value_gap

    return z_start + ratio * z_gap


def calculate_z(x, y):
    i1, i2 = get_boundary_indexes(x, 0)
    j1, j2 = get_boundary_indexes(y, 1)

    x1_z = get_middle_z(calibration_grid[i1][j1][0],
                        calibration_grid[i2][j1][0],
                        x,
                        calibration_grid[i1][j1][2],
                        calibration_grid[i2][j1][2])

    x2_z = get_middle_z(calibration_grid[i1][j2][0],
                        calibration_grid[i2][j2][0],
                        x,
                        calibration_grid[i1][j2][2],
                        calibration_grid[i2][j2][2])

    z = get_middle_z(calibration_grid[i1][j1][1],
                     calibration_grid[i1][j2][1],
                     y,
                     x1_z,
                     x2_z)

    return z


# for i in range(10):
#     x = round(random.uniform(0.3, 0.45), 2)
#     y = round(random.uniform(-0.2, 0.2), 2)
#     print(f'x = {x*100}, y = {y*100}, z = {round(calculate_z(x, y), 3)*1000}')

# print(calculate_z(0.389, 0.123))
# print(calculate_z(-0.5, -0.5))
# print(calculate_z(0.5, -0.5))
# print(calculate_z(-0.5, 0.5))
# print(calculate_z(0.5, 0.5))
# print(calculate_z(-0.5, 0.02))
# print(calculate_z(0.37, -0.5))
# print(calculate_z(-0.5, -0.02))
# print(calculate_z(0.42, 0.5))
# print(calculate_z(0.4, 0.2))
