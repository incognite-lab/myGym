import numpy as np
import open3d as o3d
import skimage as ski

# ski.morphology.skeletonize_3d()

VOXEL_SCALE = 0.001


def voxels_to_matrix(grid):
    bmin, bmax = grid.get_min_bound(), grid.get_max_bound()
    grid_size = ((bmax - bmin) / grid.voxel_size).astype(np.int64)
    mat = np.zeros(grid_size, dtype=np.uint8)
    for vox in grid.get_voxels():
        mat[*vox.grid_index] = 1
    return mat


def matrix_to_voxels(mat, voxel_size=VOXEL_SCALE, color=None):
    grid = o3d.geometry.VoxelGrid()
    grid.voxel_size = voxel_size

    set_items = np.nonzero(mat)
    for x, y, z in zip(*set_items):
        if color is None:
            v = o3d.geometry.Voxel(np.array([x, y, z]))
        else:
            v = o3d.geometry.Voxel(np.array([x, y, z]), color=color)
        grid.add_voxel(v)

    return grid


def morph_volume(mat):
    # returns the image of the object with voxels covering its entire volume
    return ski.morphology.diameter_closing(mat, max(mat.shape)).astype(bool).astype(np.uint8)


def morph_dilate(mat, radius):
    mat = np.pad(mat, radius, 'constant', constant_values=0)
    return ski.morphology.dilation(mat, ski.morphology.ball(radius)).astype(bool).astype(np.uint8)


def morph_dilate_outwards(mat, radius):
    mat = np.pad(mat, radius, 'constant', constant_values=0).astype(bool).astype(np.uint8)
    dilated_mat = ski.morphology.dilation(mat, ski.morphology.ball(radius)).astype(bool).astype(np.uint8)
    full_mat = morph_volume(mat)
    diff = dilated_mat - full_mat
    diff[diff == 255] = 0
    return diff


def morph_dilate_outwards_axis(mat, radius, axis):
    kernel = np.moveaxis(np.pad(ski.morphology.disk(3)[np.newaxis, ...], ((radius, radius), (0, 0), (0, 0)), 'constant', constant_values=0), 0, axis)
    mat = np.pad(mat, radius, 'constant', constant_values=0).astype(bool).astype(np.uint8)
    dilated_mat = ski.morphology.dilation(mat, kernel).astype(bool).astype(np.uint8)
    full_mat = morph_volume(mat)
    diff = dilated_mat - full_mat
    diff[diff == 255] = 0
    return diff


def morph_erode(mat, radius):
    return ski.morphology.erosion(mat, ski.morphology.ball(radius)).astype(bool).astype(np.uint8)


def morph_surface(mat):
    eroded_mat = morph_erode(mat, 1)
    diff = mat.astype(bool).astype(np.uint8) - eroded_mat
    diff[diff == 255] = 0
    return diff


def morph_slice(mat, axis, slice_index):
    mat_t = np.moveaxis(mat, axis, 0)
    mat_t = mat_t[slice_index, ...][np.newaxis, ...]
    mat = np.moveaxis(mat_t, 0, axis)
    return mat


# obj = o3d.io.read_file_geometry_type("bowl.obj")
# obj = o3d.t..read_file_geometry_type("bowl.obj")
dataset = o3d.data.AvocadoModel()
# obj = o3d.io.read_triangle_mesh(dataset.path)
obj = o3d.io.read_triangle_model(dataset.path)
mesh = obj.meshes[0].mesh

g_orig = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=VOXEL_SCALE)
g_2 = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh.translate(np.r_[0.02, 0, 0]), voxel_size=VOXEL_SCALE)
go = g_2.origin
g_2 = matrix_to_voxels(voxels_to_matrix(g_2), voxel_size=VOXEL_SCALE, color=np.array([1, 0, 0]))
g_2.origin = go

mat_orig = voxels_to_matrix(g_orig)

# mat_skeleton = ski.morphology.area_opening(mat_orig, 128000000000)
# mat_skeleton = ski.morphology.skeletonize_3d(mat_orig)
# mat_skeleton = ski.morphology.diameter_closing(mat_orig, max(mat_orig.shape))
# mat_skeleton = morph_dilate(morph_surface(morph_dilate(mat_orig, 9)), 1)
# mat_skeleton = morph_volume(np.pad(mat_orig, 3, 'constant', constant_values=0))
# mat_skeleton = morph_surface(mat_orig)

# mat_volume = morph_slice(morph_volume(mat_orig), 1, 32)
# mat_skeleton = morph_slice(morph_dilate(mat_orig, 1), 1, 32)
# mat_skeleton = morph_slice(morph_dilate_outwards(mat_orig, 3), 1, 32)
# mat_skeleton = morph_dilate_outwards_axis(mat_orig, 10, 1)
# radius = 3
# mat = mat_orig
# mat = np.pad(mat, radius, 'constant', constant_values=0).astype(bool).astype(np.uint8)
# dilated_mat = ski.morphology.dilation(mat, ski.morphology.ball(radius)).astype(bool).astype(np.uint8)
# full_mat = morph_volume(mat)

# full_mat = morph_slice(full_mat, 1, 32)
# dilated_mat = morph_slice(dilated_mat, 1, 32)

# g_volume = matrix_to_voxels(full_mat, color=np.r_[1, 0, 0])
# g_skeleton = matrix_to_voxels(dilated_mat)

# diff = full_mat - dilated_mat
# import cv2

# cv2.imshow("a", np.squeeze(full_mat) * 255)
# cv2.imshow("b", np.squeeze(dilated_mat) * 255)
# cv2.imshow("c", np.squeeze(diff) * 255)
# cv2.waitKey()
# diff[diff > 0] = 0
# g_volume = matrix_to_voxels(diff, color=np.r_[1, 0, 0])
# g_volume = matrix_to_voxels(mat_volume, color=np.r_[1, 0, 0])
# g_skeleton = matrix_to_voxels(mat_skeleton)
# paint_grid(g_volume, np.r_[1, 0, 0])
# g_skeleton.origin = np.r_[0, 0, 0.03]

# o3d.visualization.draw_geometries([g_skeleton])
# o3d.visualization.draw_geometries([g_volume])
# o3d.visualization.draw_geometries([g_volume, g_skeleton])
o3d.visualization.draw_geometries([g_orig, g_2])
# o3d.visualization.draw_geometries([g_volume, g_skeleton, g_orig])
