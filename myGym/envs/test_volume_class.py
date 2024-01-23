from typing import ClassVar, Optional, Tuple

import numpy as np
import open3d as o3d
import skimage.morphology as morph

# ski.morphology.skeletonize_3d()


class Volume():
    VOXEL_SCALE: ClassVar[float] = 0.01

    def __init__(self):
        pass


class VolumeMesh:

    def __init__(self, mesh=None, *, matrix: Optional[np.ndarray] = None, origin: Optional[np.ndarray] = None, scale: float = Volume.VOXEL_SCALE):
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
        return VolumeMesh._matrix_to_voxelgrid(self._matrix, self._scale, self._origin, self._color)

    def set_color(self, color: Optional[np.ndarray]) -> None:
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

    def fill(self) -> 'VolumeMesh':
        """
        Fills the occupied space so that there is no empty space "inside" the object.

        Returns:
            VolumeMesh: A 'VolumeMesh' object with the matrix filled.
        """
        # returns the image of the object with voxels covering its entire volume
        self._matrix = morph.diameter_closing(self._matrix, max(self._matrix.shape)).astype(bool).astype(np.uint8)
        return self

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

    def expand_along_axis(self, radius: int, axis: int) -> 'VolumeMesh':
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
        line = np.ones((radius * 2 + 1, 1, 1))
        padding = [(radius, radius), (0, 0), (0, 0)]
        kernel = np.moveaxis(np.pad(line, padding, 'constant', constant_values=0), 0, axis)
        dilated_mat = morph.dilation(self._matrix, kernel).astype(bool).astype(np.uint8)
        full_mat = morph.diameter_closing(self._matrix, max(self._matrix.shape)).astype(bool).astype(np.uint8)
        self._matrix = dilated_mat - full_mat
        self._matrix[self._matrix == 255] = 0
        return self

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

    def expand_outwards_along_axis(self, radius: int, axis: int) -> 'VolumeMesh':
        """
        Expand the matrix outwards along a specified axis.

        Args:
            radius (int): The radius of the expansion (dilation).
            axis (int): The axis along which to expand the matrix.

        Returns:
            self: The updated object after the expansion.
        """
        self._inflate(radius)
        line = np.ones((radius * 2 + 1, 1, 1))
        padding = [(radius, radius), (0, 0), (0, 0)]
        kernel = np.moveaxis(np.pad(line, padding, 'constant', constant_values=0), 0, axis)
        dilated_mat = morph.dilation(self._matrix, kernel).astype(bool).astype(np.uint8)
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

    def slice(self, axis, slice_index) -> 'VolumeMesh':
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
        mat_t = mat_t[slice_index, ...][np.newaxis, ...]
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

    def duplicate(self) -> 'VolumeMesh':
        """
        Creates a duplicate of the VolumeMesh object. The duplicate is create from the current state of the object.
        That is, not from the original mesh.

        Returns:
            VolumeMesh: A new VolumeMesh object that is a duplicate of the original.
        """
        return VolumeMesh(self._mesh, matrix=self._matrix, origin=self._origin, scale=self._scale)

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
        grid_size = ((bmax - bmin) / voxelgrid.voxel_size).astype(np.int64)
        matrix = np.zeros(grid_size, dtype=np.uint8)

        for vox in voxelgrid.get_voxels():
            matrix[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1

        return matrix


if __name__ == "__main__":
    dataset = o3d.data.AvocadoModel()
    obj = o3d.io.read_triangle_model(dataset.path)
    mesh = obj.meshes[0].mesh

    vm = VolumeMesh(mesh)
    orig = vm.duplicate()
    orig.set_color(np.array([0, 1, 0]))

    # Define the radius and axis variables
    radius = 15
    axis = 0

    # # show full slice
    # vm.fill().slice(1, 32)

    # # show erosion
    # vm.fill().erode(radius)

    # # show expansion
    # vm.expand(radius)

    # # show axis expansion
    # vm.expand_along_axis(radius, axis)

    # # show plane expansion
    # vm.expand_along_plane(radius, axis)

    # # show axis expansion
    # vm.expand_outwards_along_axis(radius, axis)

    # # show plane expansion
    # vm.expand_outwards_along_plane(radius, axis)

    # print(vm.bounds)
    # vm.expand_outwards_along_plane(radius, axis).extrude_along_plane(axis)

    orig.inflate(radius)
    # vm.extrude_along_plane(axis, custom_bounds=orig.bounds[axis])
    # vm.expand_outwards_along_plane(radius, axis).extrude_along_plane(axis, custom_bounds=orig.bounds[axis])
    # vm.extrude_along_axis(axis)
    # vm.inflate(radius).extrude_along_plane(axis, self_bounded=False)
    # vm.expand_outwards_along_plane(radius, axis)#.extrude_along_axis(axis, custom_bounds=orig.bounds[axis])
    vm.expand(radius)#.extrude_along_axis(axis, custom_bounds=orig.bounds[axis])

    voxel_grid = vm.voxelgrid

    o3d.visualization.draw_geometries([voxel_grid, orig.voxelgrid])
