import functools
from typing import Any, ClassVar, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import skimage.morphology as morph

RNG = np.random.default_rng(0)


class VolumePredicate():
    VOXEL_SCALE: ClassVar[float] = 0.005

    def __init__(self):
        self.__predicates = {}

    def __setitem__(self, __name: str, __value: 'VolumeMesh') -> None:
        proxy_predicate = self._proxy_predicate(__value)
        self.__predicates[__name] = proxy_predicate
        setattr(self, __name, proxy_predicate)

    def __getitem__(self, __name: str) -> 'VolumeMesh':
        return self.__predicates[__name]

    def _sample_volume(self, volume, *args, **kwargs):
        return volume.sample(*args, **kwargs)

    def _proxy_predicate(self, predicate_volume):
        return lambda *args, **kwargs: self._sample_volume(volume=predicate_volume, *args, **kwargs)


def reset_vm_cache(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self._reset_wm_cache()  # Reset the cache before the method call
        result = func(self, *args, **kwargs)
        return result
    return wrapper


def update_origin_vm_cache(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._update_origin_vm_cache(self._origin)  # Reset the cache before the method call
        return result
    return wrapper


class VolumeMesh:

    def __init__(self, mesh=None, *, matrix: Optional[np.ndarray] = None, origin: Optional[np.ndarray] = None, scale: float = VolumePredicate.VOXEL_SCALE):
        """
        Initializes a VolumeMesh object. Either from a mesh or from a custom matrix (e.g., when duplicating).

        Parameters:
            mesh (optional): The mesh to voxelize and store in the VolumeMesh object. If None, a matrix must be provided.
            matrix (optional): The matrix representation of the volume. If None, a mesh must be provided.
            origin (optional): The origin of the volume. Only used if a mesh is not provided.
            scale (optional): The scale of the volume. Defaults to Volume.VOXEL_SCALE.

        Raises:
            ValueError: If no mesh is provided and no matrix is provided.
        """
        self._mesh = mesh
        self._scale = scale
        self._color = None
        self._origin: np.ndarray
        self.__cache = None
        self._non_empty_cells = None

        if self._mesh is not None:
            # voxelize the mesh
            voxelgrid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(self._mesh, self._scale)
            # compute matrix from voxelgrid
            self._matrix = VolumeMesh._voxelgrid_to_matrix(voxelgrid)
            self._origin = voxelgrid.origin.copy()
        else:
            if matrix is None:
                raise ValueError("If no mesh is provided, a matrix must be provided!")
            self._matrix = matrix
            if origin is None:
                self._origin = np.zeros(3)
            else:
                self._origin = origin

    @property
    def origin(self) -> np.ndarray:
        """
        Returns the origin of the voxel grid.
        """
        return self._origin

    @property
    def voxelgrid(self) -> o3d.geometry.VoxelGrid:
        """
        Generates the voxel grid from the matrix.

        Returns:
            o3d.geometry.VoxelGrid: voxel grid corresponding to the current state of the volume.
        """
        if self.__cache is None:
            self.__cache = VolumeMesh._matrix_to_voxelgrid(self._matrix, self._scale, self._origin, self._color)
        return self.__cache

    @property
    def matrix(self) -> np.ndarray:
        """
        Returns the matrix representation of the volume.
        """
        return self._matrix.copy()

    def _reset_wm_cache(self):
        self.__cache = None
        self._non_empty_cells = None

    @property
    def non_empty_cells(self):
        if self._non_empty_cells is None:
            self._non_empty_cells = np.argwhere(self._matrix)
        return self._non_empty_cells

    def _update_origin_vm_cache(self, origin):
        if self.__cache is None:
            return
        self.__cache.origin = origin

    def __getitem__(self, key):
        if type(key) is slice:
            raise NotImplementedError("slicing not implemented for VolumeMesh")
        elif type(key) is int:
            raise NotImplementedError("indexing not implemented for VolumeMesh")
        elif type(key) is tuple and len(key) == 2 and type(key[0]) is int and (type(key[1]) is slice or type(key[1]) is int):
            return self.slice(key[0], key[1])
        else:
            raise NotImplementedError(f"Using '{type(key)}' for indexing is not implemented for VolumeMesh.")

    def paint(self, color: Optional[np.ndarray]) -> None:
        """
        Set grid color. This will show when exported to VoxelGrid.

        Args:
            color (Optional[np.ndarray]): color array in the shape [R, G, B]. Each component must a float in [0, 1].
        """
        self._color = color

    def _inflate(self, radius: int) -> None:
        """
        Inflates (pads) the matrix by the given radius. The inflation/padding happens on all sides of the matrix.

        Args:
            radius (int): The radius to inflate the matrix by.
        """
        self._matrix = np.pad(self._matrix, radius, 'constant', constant_values=0)
        self._origin -= radius * self._scale

    @update_origin_vm_cache
    def inflate(self, radius: int) -> 'VolumeMesh':
        """
        Inflates the current matrix representation of the volume by the given radius.
        Inflation or rather padding is done on all sides of the matrix. That is,
        all 3 dimensions of the matrix will expand by `radius * 2`.

        Parameters:
            radius (int): The radius of inflation.

        Returns:
            VolumeMesh: The inflated VolumeMesh object.

        """
        self._inflate(radius)
        return self

    @reset_vm_cache
    def fill(self) -> 'VolumeMesh':
        """
        Fills the occupied space so that there is no empty space "inside" the object.

        Returns:
            VolumeMesh: A 'VolumeMesh' object with the matrix filled.
        """
        # returns the image of the object with voxels covering its entire volume
        self._matrix = morph.diameter_closing(self._matrix, max(self._matrix.shape)).astype(bool).astype(np.uint8)
        return self

    @reset_vm_cache
    def erode(self, radius: int) -> 'VolumeMesh':
        """
        Erodes the volume mesh by applying a morphological erosion operation.

        Args:
            radius (int): The radius of the erosion operation.

        Returns:
            VolumeMesh: The eroded volume mesh.
        """
        self._matrix = morph.erosion(self._matrix, morph.ball(radius)).astype(bool).astype(np.uint8)
        return self

    @reset_vm_cache
    def dilate(self, radius: int) -> 'VolumeMesh':
        """
        Dilates the volume mesh by expanding the regions in the mesh.

        Parameters:
            radius (int): The radius of the dilation. Determines the extent of expansion.

        Returns:
            VolumeMesh: The dilated volume mesh.
        """
        self._inflate(radius)
        self._matrix = morph.dilation(self._matrix, morph.ball(radius)).astype(bool).astype(np.uint8)
        return self

    @reset_vm_cache
    def surface(self) -> 'VolumeMesh':
        """
        Reduces the volume to just the surface voxels.

        Returns:
            VolumeMesh: The surface volume mesh.
        """
        eroded_mat = morph.erosion(self._matrix, morph.ball(1)).astype(bool).astype(np.uint8)
        self._matrix -= eroded_mat
        self._matrix[self._matrix == 255] = 0
        return self

    @reset_vm_cache
    def expand(self, radius: int) -> 'VolumeMesh':
        """
        Expand the volume mesh by the given radius. This is done by applying a morphological dilation operation.
        Afterwards, the "inside" of the volume is carved out by the currently occupied volume (current object).
        That is, the new volume is the difference between the current volume/object and the dilated volume.

        Args:
            radius (int): The radius to expand the volume mesh by.

        Returns:
            VolumeMesh: The expanded volume mesh.
        """
        self._inflate(radius)
        dilated_mat = morph.dilation(self._matrix, morph.ball(radius)).astype(bool).astype(np.uint8)
        full_mat = morph.diameter_closing(self._matrix, max(self._matrix.shape)).astype(bool).astype(np.uint8)
        self._matrix = dilated_mat - full_mat
        self._matrix[self._matrix == 255] = 0
        return self

    @staticmethod
    def _dilate_along_axis(matrix: np.ndarray, radius: int, axis: int, polarity: int = 0):
        line = np.ones((radius * 2 + 1, 1, 1))
        if polarity > 0:
            line[:radius + 1, 0, 0] = 0
        elif polarity < 0:
            line[radius:, 0, 0] = 0

        # padding = [(radius, radius), (0, 0), (0, 0)]
        # kernel = np.moveaxis(np.pad(line, padding, 'constant', constant_values=0), 0, axis)
        kernel = np.moveaxis(line, 0, axis)
        return morph.dilation(matrix, kernel).astype(bool).astype(np.uint8)

    @reset_vm_cache
    def expand_along_axis(self, radius: int, axis: int, polarity: int = 0) -> 'VolumeMesh':
        """
        Expand the volume mesh along a specified axis by the given radius.
        This is the same as expand() but only along one axis.

        Parameters:
            radius (int): The radius by which to expand the volume mesh.
            axis (int): The axis along which to expand the volume mesh.

        Returns:
            VolumeMesh: The expanded volume mesh.
        """
        self._inflate(radius)
        dilated_mat = self._dilate_along_axis(self._matrix, radius, axis, polarity)
        full_mat = morph.diameter_closing(self._matrix, max(self._matrix.shape)).astype(bool).astype(np.uint8)
        self._matrix = dilated_mat - full_mat
        self._matrix[self._matrix == 255] = 0
        return self

    @reset_vm_cache
    def expand_along_plane(self, radius: int, up_axis: int) -> 'VolumeMesh':
        """
        Expand the volume mesh along a plane. This is the same as expand() but only along the given plane.
        The plane is defined by up_axis which is the axis perpendicular to the plane.

        Args:
            radius (int): The radius of the disk used for dilation.
            up_axis (int): The axis perpendicular to the expansion plane.

        Returns:
            VolumeMesh: The expanded volume mesh.
        """
        self._inflate(radius)
        disk = np.expand_dims(morph.disk(radius), axis=0)
        padding = [(radius, radius), (0, 0), (0, 0)]
        kernel = np.pad(
            disk,
            padding,
            'constant',
            constant_values=0
        )
        kernel = np.moveaxis(kernel, 0, up_axis)
        dilated_mat = morph.dilation(self._matrix, kernel).astype(bool).astype(np.uint8)
        full_mat = morph.diameter_closing(self._matrix, max(self._matrix.shape)).astype(bool).astype(np.uint8)
        self._matrix = dilated_mat - full_mat
        self._matrix[self._matrix == 255] = 0
        return self

    @staticmethod
    def _get_bounds(matrix) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Calculates the bounds of an object within the given matrix. That is, returns the min & max positions
        for each dimension where occupied voxels exist.

        Parameters:
            matrix (np.ndarray): The matrix to calculate the bounds for.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]: A tuple containing three tuples representing the bounds of the matrix in each dimension.
        """
        x, y, z = np.nonzero(matrix)
        return (x.min().tolist(), x.max().tolist()), (y.min().tolist(), y.max().tolist()), (z.min().tolist(), z.max().tolist())

    @staticmethod
    def _extrude_along_plane(matrix, up_axis: int, a_min: int = 0, a_max: int = 2**32, b_min: int = 0, b_max: int = 2**32) -> np.ndarray:
        """
        Extrudes a matrix along a specified plane within a given range. Extrusion means that along each extruded dimension,
        the space will be fully occupied if there is any occupied voxel in the corresponding column.

        Args:
            matrix (np.ndarray): The input matrix to be extruded.
            axis (int): Axis perpendicular to the extrusion plane.
            a_min (int, optional): The minimum index along the extrusion axis. Defaults to 0.
            a_max (int, optional): The maximum index along the extrusion axis. Defaults to 2**32.
            b_min (int, optional): The minimum index along the other axis. Defaults to 0.
            b_max (int, optional): The maximum index along the other axis. Defaults to 2**32.

        Returns:
            np.ndarray: The extruded matrix.
        """
        tmp = np.ascontiguousarray(np.moveaxis(matrix, up_axis, 0))
        x, _, _ = np.nonzero(tmp)
        x = np.unique(x)
        tmp[x, a_min:a_max, b_min:b_max] = 1
        return np.moveaxis(np.ascontiguousarray(tmp), 0, up_axis)

    @reset_vm_cache
    def extrude_along_plane(self, up_axis: int, *, self_bounded: bool = True, custom_bounds: Optional[Tuple[int, int, int, int]] = None) -> 'VolumeMesh':
        """
        Extrudes the volume mesh along a specific plane, perpendicular to the given axis.

        Args:
            axis (int): The axis perpendicular to extrusion plane.

        Keyword Args:
            self_bounded (bool): Whether to use the current bounds of the volume mesh or not.
                If True, the function will use the current bounds of the volume mesh to determine the extrusion bounds.
                If False, the function will use a default set of bounds.
                Defaults to True.

            custom_bounds (Optional[Tuple[int, int, int, int]]): Custom bounds to use for the extrusion.
                If provided, the function will use these bounds instead of the default or the current bounds.
                Defaults to None.

        Returns:
            VolumeMesh: The extruded volume mesh.

        Raises:
            None.
        """
        if self_bounded:
            bounds = self._get_bounds(self._matrix)
            bounds = tuple(sum(bounds[:up_axis] + bounds[up_axis + 1:], ()))
        else:
            bounds = (0, 2**32) * 2
        if custom_bounds is not None:
            bounds = custom_bounds
        self._matrix = self._extrude_along_plane(self._matrix, up_axis, *bounds)
        return self

    @reset_vm_cache
    def expand_outwards_along_axis(self, radius: int, axis: int, polarity: int = 0) -> 'VolumeMesh':
        """
        Expand the matrix outwards along a specified axis.

        Args:
            radius (int): The radius of the expansion (dilation).
            axis (int): The axis along which to expand the matrix.

        Returns:
            self: The updated object after the expansion.
        """
        self._inflate(radius)
        dilated_mat = self._dilate_along_axis(self._matrix, radius, axis, polarity)
        full_mat = morph.diameter_closing(self._matrix, max(self._matrix.shape)).astype(bool).astype(np.uint8)
        full_mat = self._extrude_along_plane(full_mat, axis)
        self._matrix = dilated_mat - full_mat
        self._matrix[self._matrix == 255] = 0
        return self

    @staticmethod
    def _extrude_along_axis(matrix, axis: int, min_limit=0, max_limit=2**32) -> np.ndarray:
        """
        Extrudes the given matrix along the specified axis.

        Args:
            matrix (ndarray): The input matrix.
            axis (int): The axis along which to extrude the matrix.
            min_limit (int, optional): The minimum limit for extrusion. Defaults to 0.
            max_limit (int, optional): The maximum limit for extrusion. Defaults to 2**32.

        Returns:
            ndarray: The extruded matrix.
        """
        tmp = np.asfortranarray(np.moveaxis(matrix, axis, 0))
        _, y, z = np.nonzero(tmp)
        tmp[min_limit:max_limit, y, z] = 1
        return np.moveaxis(np.ascontiguousarray(tmp), 0, axis)

    @reset_vm_cache
    def extrude_along_axis(self, axis: int, *, self_bounded: bool = True, custom_bounds: Optional[Tuple[int, int]] = None) -> 'VolumeMesh':
        """
        Extrudes the volume mesh along a specified axis.

        Parameters:
            axis (int): The axis along which the volume mesh should be extruded.
            self_bounded (bool, optional): Whether the extrusion should be limited to the bounds of the current volume mesh. Defaults to True.
            custom_bounds (Optional[Tuple[int, int]], optional): Custom bounds for the extrusion. If specified, the extrusion will be limited to these bounds. Defaults to None.

        Returns:
            VolumeMesh: The extruded volume mesh.

        """
        if self_bounded:
            min_limit, max_limit = self._get_bounds(self._matrix)[axis]
        else:
            min_limit, max_limit = 0, 2**32
        if custom_bounds is not None:
            min_limit, max_limit = custom_bounds
        self._matrix = self._extrude_along_axis(self._matrix, axis, min_limit, max_limit)
        return self

    @reset_vm_cache
    def expand_outwards_along_plane(self, radius: int, up_axis: int) -> 'VolumeMesh':
        """
        Expands the volume mesh outwards along a plane.

        Args:
            radius (int): The radius of the expansion.
            up_axis (int): The axis along which to expand.

        Returns:
            VolumeMesh: The expanded volume mesh.
        """
        self._inflate(radius)
        disk = np.expand_dims(morph.disk(radius), axis=0)
        padding = [(radius, radius), (0, 0), (0, 0)]
        kernel = np.pad(
            disk,
            padding,
            'constant',
            constant_values=0
        )
        kernel = np.moveaxis(kernel, 0, up_axis)
        dilated_mat = morph.dilation(self._matrix, kernel).astype(bool).astype(np.uint8)
        full_mat = morph.diameter_closing(self._matrix, max(self._matrix.shape)).astype(bool).astype(np.uint8)
        full_mat = self._extrude_along_axis(full_mat, up_axis)
        # self._matrix = full_mat
        self._matrix = dilated_mat - full_mat
        self._matrix[self._matrix == 255] = 0
        return self

    @reset_vm_cache
    def slice(self, axis: int, slice_index: Union[int, slice]) -> 'VolumeMesh':
        """
        Slice the volume mesh along a given axis at a specific slice index.
        That is, returns a single slice (single plane) of voxels.

        Parameters:
            axis (int): The axis along which to slice the volume mesh.
            slice_index (int): The index within the axis of the slice to extract.

        Returns:
            VolumeMesh: The sliced volume mesh.

        """
        mat_t = np.moveaxis(self._matrix, axis, 0)
        mat_t = mat_t[slice_index, ...]
        if type(slice_index) is int:
            mat_t = mat_t[np.newaxis, ...]
        self._matrix = np.moveaxis(mat_t, 0, axis)
        return self

    @property
    def size(self) -> Tuple[int, int, int]:
        """
        Return the size of the matrix/grid in voxels.

        Returns:
            Tuple[int, int, int]: A tuple containing the dimensions of the matrix in the form (rows, columns, depth).
        """
        return self._matrix.shape

    @property
    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Calculates the bounds of an object within the given matrix.

        Parameters:
            matrix (np.ndarray): The matrix to calculate the bounds for.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]: A tuple containing three tuples representing the bounds of the matrix in each dimension.
        """
        return self._get_bounds(self._matrix)

    @property
    def aabb(self) -> o3d.geometry.AxisAlignedBoundingBox:
        """
        Calculates the axis-aligned bounding box of the voxel grid.

        Returns:
            o3d.geometry.AxisAlignedBoundingBox: The axis-aligned bounding box of the object.
        """
        return self.voxelgrid.get_axis_aligned_bounding_box()

    def compactify(self) -> None:
        """
        Compacts the volume matrix by removing empty leading and trailing voxels
        for each dimension.

        Returns:
            VolumeMesh: The compactified volume mesh.
        """
        n0, n1, n2 = np.nonzero(self._matrix)
        n0 = np.unique(n0)
        n1 = np.unique(n1)
        n2 = np.unique(n2)
        w0_start, w0_end = n0[0], n0[-1] + 1
        w1_start, w1_end = n1[0], n1[-1] + 1
        w2_start, w2_end = n2[0], n2[-1] + 1
        self._origin = np.r_[self._origin[0] + w0_start * self._scale,
                             self._origin[1] + w1_start * self._scale,
                             self._origin[2] + w2_start * self._scale
                             ]
        self._matrix = self._matrix[w0_start:w0_end, w1_start:w1_end, w2_start:w2_end]

    def duplicate(self) -> 'VolumeMesh':
        """
        Creates a duplicate of the VolumeMesh object. The duplicate is create from the current state of the object.
        That is, not from the original mesh.

        Returns:
            VolumeMesh: A new VolumeMesh object that is a duplicate of the original.
        """
        return VolumeMesh(self._mesh, matrix=self._matrix, origin=self._origin, scale=self._scale)

    @update_origin_vm_cache
    def translate(self, translation: np.ndarray) -> None:
        [VolumeMesh.convert_meters_to_voxels(t) for t in translation]
        self._origin += translation

    @staticmethod
    def _matrix_to_voxelgrid(matrix: np.ndarray, voxel_size: float, origin: Optional[np.ndarray] = None, color: Optional[np.ndarray] = None) -> o3d.geometry.VoxelGrid:
        grid = o3d.geometry.VoxelGrid()
        grid.voxel_size = voxel_size
        if origin is not None:
            grid.origin = origin

        set_items = np.nonzero(matrix)
        for x, y, z in zip(*set_items):
            if color is None:
                v = o3d.geometry.Voxel(np.array([x, y, z]))
            else:
                v = o3d.geometry.Voxel(np.array([x, y, z]), color=color)
            grid.add_voxel(v)

        return grid

    @staticmethod
    def _voxelgrid_to_matrix(voxelgrid: o3d.geometry.VoxelGrid) -> np.ndarray:
        bmin, bmax = voxelgrid.get_min_bound(), voxelgrid.get_max_bound()
        grid_size = np.ceil((bmax - bmin) / voxelgrid.voxel_size).astype(np.int64)
        matrix = np.zeros(grid_size, dtype=np.uint8)

        for vox in voxelgrid.get_voxels():
            matrix[tuple(vox.grid_index)] = 1

        return matrix

    @staticmethod
    def _shrink_matrix(matrix):
        non_empty_rows = np.any(matrix, axis=(1, 2))
        non_empty_columns = np.any(matrix, axis=(0, 2))
        non_empty_depths = np.any(matrix, axis=(0, 1))

        shrunk_matrix = matrix[non_empty_rows][:, non_empty_columns][:, :, non_empty_depths]
        return shrunk_matrix

    @staticmethod
    def convert_meters_to_voxels(distance_in_meters: float) -> int:
        return np.ceil(distance_in_meters / VolumePredicate.VOXEL_SCALE).astype(int)

    @staticmethod
    def convert_voxels_to_meters(distance_in_voxels: int) -> float:
        return distance_in_voxels * VolumePredicate.VOXEL_SCALE

    def __add__(self, other: 'VolumeMesh') -> 'VolumeMesh':
        # get aabbs of both vms
        aabb = self.aabb
        aabb2 = other.aabb
        min_bounds = np.minimum(aabb.get_min_bound(), aabb2.get_min_bound())
        max_bounds = np.maximum(aabb.get_max_bound(), aabb2.get_max_bound())
        # get grid size
        total_size = (max_bounds - min_bounds) / VolumePredicate.VOXEL_SCALE
        combined_matrix = np.zeros(np.ceil(total_size).astype(int), dtype=np.uint8)
        pos_self = ((aabb.get_min_bound() - min_bounds) / VolumePredicate.VOXEL_SCALE).astype(int)
        pos_other = ((aabb2.get_min_bound() - min_bounds) / VolumePredicate.VOXEL_SCALE).astype(int)

        mat_self = self._shrink_matrix(self.matrix)
        mat_other = self._shrink_matrix(other.matrix)

        combined_matrix[pos_self[0]:pos_self[0] + mat_self.shape[0], pos_self[1]:pos_self[1] + mat_self.shape[1], pos_self[2]:pos_self[2] + mat_self.shape[2]] += mat_self
        combined_matrix[pos_other[0]:pos_other[0] + mat_other.shape[0], pos_other[1]:pos_other[1] + mat_other.shape[1], pos_other[2]:pos_other[2] + mat_other.shape[2]] += mat_other

        vm = VolumeMesh(None, matrix=combined_matrix, origin=min_bounds, scale=VolumePredicate.VOXEL_SCALE)

        return vm

    def __merge_with_other_volume(self, other: 'VolumeMesh', factor: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get aabbs of both vms
        aabbs = [self.aabb, other.aabb]
        local_min_bounds = np.r_[[b.get_min_bound() for b in aabbs]]
        local_max_bounds = np.r_[[b.get_max_bound() for b in aabbs]]
        arg_min_bounds = np.argmin(local_min_bounds, axis=0)
        min_bounds = np.r_[[lmb[amb] for lmb, amb in zip(local_min_bounds.T, arg_min_bounds)]]
        arg_max_bounds = np.argmax(local_max_bounds, axis=0)
        max_bounds = np.r_[[lmb[amb] for lmb, amb in zip(local_max_bounds.T, arg_max_bounds)]]

        if not np.all(np.r_[[lmb[amb - 1] for lmb, amb in zip(local_min_bounds.T, arg_min_bounds)]] < np.r_[[lmb[amb - 1] for lmb, amb in zip(local_max_bounds.T, arg_max_bounds)]]):
            raise ValueError('The volumes do not overlap!')

        # get grid size
        total_size = np.ceil((max_bounds - min_bounds) / VolumePredicate.VOXEL_SCALE).astype(int)
        combined_matrix = np.zeros(total_size, dtype=np.uint8)
        pos_self = ((local_min_bounds[0, :] - min_bounds) / VolumePredicate.VOXEL_SCALE).astype(int)
        pos_other = ((local_min_bounds[1, :] - min_bounds) / VolumePredicate.VOXEL_SCALE).astype(int)

        mat_self = self._shrink_matrix(self.matrix)
        mat_other = self._shrink_matrix(other.matrix)

        combined_matrix[pos_self[0]:pos_self[0] + mat_self.shape[0], pos_self[1]:pos_self[1] + mat_self.shape[1], pos_self[2]:pos_self[2] + mat_self.shape[2]] += mat_self
        if factor > 1:
            mat_other *= factor
        combined_matrix[pos_other[0]:pos_other[0] + mat_other.shape[0], pos_other[1]:pos_other[1] + mat_other.shape[1], pos_other[2]:pos_other[2] + mat_other.shape[2]] += mat_other

        return combined_matrix, min_bounds, max_bounds

    def __sub__(self, other: 'VolumeMesh') -> 'VolumeMesh':
        combined_matrix, min_bounds, _ = self.__merge_with_other_volume(other, factor=2)

        combined_matrix[combined_matrix > 1] = 0

        vm = VolumeMesh(None, matrix=combined_matrix, origin=min_bounds, scale=VolumePredicate.VOXEL_SCALE)

        return vm

    def __mul__(self, other: 'VolumeMesh') -> 'VolumeMesh':
        combined_matrix, min_bounds, _ = self.__merge_with_other_volume(other)

        overlap_indices = combined_matrix > 1  # wherever == 2 -> overlap
        combined_matrix[np.logical_not(overlap_indices)] = 0
        combined_matrix[overlap_indices] = 1

        vm = VolumeMesh(None, matrix=combined_matrix, origin=min_bounds, scale=VolumePredicate.VOXEL_SCALE)

        return vm

    @staticmethod
    def _get_cell_corners(cell_idx, origin, scale) -> np.ndarray:
        cell_corner = cell_idx * scale + origin
        cell_corners = np.array([
            [cell_corner[0], cell_corner[1], cell_corner[2]],
            [cell_corner[0] + scale, cell_corner[1], cell_corner[2]],
            [cell_corner[0], cell_corner[1] + scale, cell_corner[2]],
            [cell_corner[0] + scale, cell_corner[1] + scale, cell_corner[2]],
            [cell_corner[0], cell_corner[1], cell_corner[2] + scale],
            [cell_corner[0] + scale, cell_corner[1], cell_corner[2] + scale],
            [cell_corner[0], cell_corner[1] + scale, cell_corner[2] + scale],
            [cell_corner[0] + scale, cell_corner[1] + scale, cell_corner[2] + scale]
        ])
        return cell_corners

    @staticmethod
    def random_point_in_bounding_box(corners, rng=RNG) -> np.ndarray:
        min_corner = np.min(corners, axis=0)
        max_corner = np.max(corners, axis=0)
        return rng.uniform(min_corner, max_corner)

    def sample(self, *args, **kwargs) -> np.ndarray:
        if "n_samples" in kwargs:
            n_samples = kwargs["n_samples"]
            del kwargs["n_samples"]
        else:
            n_samples = 1
        if "rng" in kwargs:
            rng = kwargs["rng"]
            del kwargs["rng"]
        else:
            rng = RNG

        cell_idx = rng.choice(len(self.non_empty_cells), n_samples)
        sample_cells = self.non_empty_cells[cell_idx, :]
        if n_samples == 1:
            corners = self._get_cell_corners(sample_cells[0], self.origin, self._scale)
            return self.random_point_in_bounding_box(corners)
        else:
            corners_multi = np.apply_along_axis(self._get_cell_corners, 1, sample_cells, self.origin, self._scale)
            return np.apply_along_axis(self.random_point_in_bounding_box, 1, corners_multi, rng=rng)


def test_read_avocado():
    dataset = o3d.data.AvocadoModel()
    obj = o3d.io.read_triangle_model(dataset.path)
    mesh = obj.meshes[0].mesh
    return mesh


def test_read_pipe():
    dataset = o3d.data.FlightHelmetModel()
    obj = o3d.io.read_triangle_model(dataset.path)
    mesh = obj.meshes[0].mesh
    return mesh


def test_basics():
    v1 = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_avocado, VolumePredicate.VOXEL_SCALE)
    mesh_avocado.translate(np.r_[0.005, 0.0, 0.0])
    v2 = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_avocado, VolumePredicate.VOXEL_SCALE)
    for vxl in v2.get_voxels():
        vxl.color = np.array([0, 1, 0])

    # geometries = [v1, v2]

    # geometries.extend([axis_o])
    # o3d.visualization.draw_geometries(geometries)
    # exit(0)

    # mesh_path = '/home/syxtreme/Documents/repositories/myGym/myGym/envs/objects/geometric/obj/cube.obj'
    # obj = o3d.io.read_triangle_model(mesh_path)
    # mesh = obj.meshes[0].mesh.scale(0.01, np.r_[0, 0, 0])
    # print(mesh)

    # o3d.visualization.draw_geometries([mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])

    vm = VolumeMesh(mesh_avocado)
    orig = vm.duplicate()
    orig.paint(np.array([0, 1, 0]))

    # Define the radius and axis variables
    radius = VolumeMesh.convert_meters_to_voxels(0.05)
    # radius = 3
    axis = 0

    # orig.inflate(radius)
    print(vm.origin)
    print(vm.voxelgrid.origin)
    dist = 0.06
    vm.translate(np.array([dist, 0, 0]))
    print(vm.origin)
    print(vm.voxelgrid.origin)

    # vm2 = vm.duplicate()
    # vm2.set_color(np.array([1, 0, 0]))
    # vm2.translate(np.array([0.05, 0, 0]))
    # vm2.expand_along_axis(radius * 3, axis, polarity=1)

    # vm.expand_along_axis(radius, axis, polarity=-1)

    voxel_grid = vm.voxelgrid
    # voxel_grid2 = vm2.voxelgrid

    # # octree = o3d.geometry.Octree(max_depth=4)
    # # octree.create_from_voxel_grid(voxel_grid)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis.scale(0.05, np.r_[0, 0, 0])
    axis.translate(vm.origin)

    axis_o = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis_o.scale(0.01, np.r_[0, 0, 0])

    # vm3 = vm + vm2
    # vm3 = vm - vm2
    # vm3 = vm2 - vm
    # vm3 = vm * vm2
    # vm3.set_color(np.array([0, 0, 1]))

    # v = VolumePredicate()
    # v['onTop'] = vm
    # v['onBottom'] = vm2

    geometries = [voxel_grid, orig.voxelgrid]
    # geometries = [voxel_grid, voxel_grid2]
    # geometries = [voxel_grid, voxel_grid2]
    # geometries = [vm3.voxelgrid]
    geometries.extend([axis, axis_o])
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    mesh_avocado = test_read_avocado()
    mesh_pipe = test_read_pipe()

    vm_avocado = VolumeMesh(mesh_avocado)
    vm_pipe = VolumeMesh(mesh_pipe)

    # mesh = mesh_pipe
    mesh = mesh_avocado
    # vm = vm_pipe
    vm = vm_avocado

    vm.expand_outwards_along_plane(radius=5, up_axis=1)
    vm[1, :10]
    vm.compactify()
    vm.extrude_along_axis(axis=1, custom_bounds=(0, 10), self_bounded=False)
    vm[1, 0]

    axis_o = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis_o.scale(0.01, np.r_[0, 0, 0])

    vp = VolumePredicate()
    vp['nextTo'] = vm
    s = vp.nextTo(n_samples=1000)

    sample_indicator_points = o3d.geometry.PointCloud()
    sample_indicator_points.points = o3d.utility.Vector3dVector(s)
    o3d.visualization.draw_geometries([axis_o, mesh, vm.voxelgrid, sample_indicator_points])