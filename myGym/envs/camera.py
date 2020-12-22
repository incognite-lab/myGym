import numpy as np
import pybullet as p


class Camera():
    """
    Camera class for rendering

    Parameters:
        :param env: (CameraEnv) Environment in which camera is located
        :param position: (list) Eye position in Cartesian world coordinates
        :prarm target_position: (list) Position of the target point
        :param up_vector: (list) Up vector of the camera
        :param up_axis_index: (int) Either 1 for Y or 2 for Z axis up
        :param yaw: (float) Yaw angle in degrees left/right around up-axis
        :param pitch: (float) Pitch in degrees up/down
        :param roll: (float) Roll in degrees around forward vector
        :param distance: (float) Distance from eye to focus point
        :param field_of_view: (float) Field of view
        :param near_plane_distance: (float) Near plane distance
        :param far_plane_distance: (float) Far plane distance
        :param is_absolute_position: (bool) Compute view matrix using poistion or yaw,pitch,roll
    """
    def __init__(self, env=None,
                 position=[0, 0, 0], target_position=[0, 0, 0],
                 up_vector=[0, 0, 1], up_axis_index=2,
                 yaw=180, pitch=-40, roll=0,
                 distance=1.3, field_of_view=60,
                 near_plane_distance=0.1, far_plane_distance=100.0,
                 is_absolute_position=False):

        self.env = env
        self.up_vector = up_vector
        self.up_axis_index = up_axis_index
        self.is_absolute_position = is_absolute_position

        self.set_parameters(position, target_position, yaw, pitch, roll,
                            distance, field_of_view, near_plane_distance,
                            far_plane_distance)

    def set_parameters(self, position=None, target_position=None,
                       yaw=None, pitch=None, roll=None, distance=None,
                       field_of_view=None, near_plane_distance=None,
                       far_plane_distance=None):
        """
        Set camera position and captured image parameters

        Parameters:
            :param position: (list) Eye position in Cartesian world coordinates
            :prarm target_position: (list) Position of the target point
            :param yaw: (float) Yaw angle in degrees left/right around up-axis
            :param pitch: (float) Pitch in degrees up/down
            :param roll: (float) Roll in degrees around forward vector
            :param distance: (float) Distance from eye to focus point
            :param field_of_view: (float) Field of view
            :param near_plane_distance: (float) Near plane distance
            :param far_plane_distance: (float) Far plane distance
        """
        if position is not None: self.position = position
        if target_position is not None: self.target_position = target_position
        if yaw is not None: self.yaw = yaw
        if pitch is not None: self.pitch = pitch
        if roll is not None: self.roll = roll
        if distance is not None: self.distance = distance
        if field_of_view is not None: self.field_of_view = field_of_view
        if near_plane_distance is not None: self.near_plane_distance = near_plane_distance
        if far_plane_distance is not None: self.far_plane_distance = far_plane_distance
        self.recompute_matrixes()

    def recompute_matrixes(self):
        """
        Compute view and projection matrixes needed for rendering
        """
        self.aspect_ratio = float(
            self.env.camera_resolution[0]) / float(self.env.camera_resolution[1])
        if (self.is_absolute_position):
            self.view_matrix = self.env.p.computeViewMatrix(
                self.position, self.target_position, self.up_vector)
        else:
            self.view_matrix = self.env.p.computeViewMatrixFromYawPitchRoll(self.target_position, self.distance,
                                                                   self.yaw, self.pitch, self.roll,
                                                                   self.up_axis_index)
        self.proj_matrix = self.env.p.computeProjectionMatrixFOV(self.field_of_view, self.aspect_ratio,
                                                        self.near_plane_distance, self.far_plane_distance)

        self.view_x_proj = np.matmul(np.reshape(
            self.view_matrix, (4, 4)), np.reshape(self.proj_matrix, (4, 4)))

    def project_point_to_image(self, point):
        """
        Project 3D point in Cartesian world coordinates to 2D point in pixel space

        Parameters:
            :param point: (list) 3D point in Cartesian world coordinates
        Returns:
            :return 2d_point: (list) 2D coordinates of point on image
        """
        # Multiply VMxPMxPoint3D
        xyzw = np.matmul(list(point) + [1], self.view_x_proj)
        # Normalize coord to (-1, 1)
        xyz = xyzw[:3] / xyzw[3]
        # Compute pixel 2D coordinates
        point2d = [(xyz[0] + 1.) * self.env.camera_resolution[0] / 2., (-xyz[1] + 1.)
                   * self.env.camera_resolution[1] / 2.]
        return np.asarray(np.round(point2d))

    def get_opencv_camera_matrix_values(self):
        """
        Get values of OpenCV matrix

        Returns:
            :return values: (dict) fx, fy, cx, cy values
        """
        values = {}
        opengl_matrix = np.reshape(self.proj_matrix, (4, 4))
        values['fx'] = opengl_matrix[0][0] * self.env.camera_resolution[0] / 2.
        values['fy'] = - opengl_matrix[1][1] * \
            self.env.camera_resolution[1] / 2.
        values['cx'] = int((1. - opengl_matrix[0][3])
                           * self.env.camera_resolution[0] / 2.)
        values['cy'] = int((1. + opengl_matrix[1][0])
                           * self.env.camera_resolution[1] / 2.)
        return values

    def render(self):
        """
        Get RGB image,depth image and segmentation mask from the camera

        Returns:
            :return data: (dict) Image, depth, segmentation_mask
        """

        render_parameters = self.env.get_render_parameters()
        width, height, rgb_pixels, depth_pixels, \
        segmentation_mask_buffer = self.env.p.getCameraImage(viewMatrix=self.view_matrix,
                                                    projectionMatrix=self.proj_matrix,
                                                    flags=self.env.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                    **render_parameters)
        rgb_image = np.array(rgb_pixels, dtype=np.uint8)
        rgb_image = np.reshape(rgb_image, (height, width, 4))
        rgb_image = rgb_image[:, :, :3]

        return {
                "image": rgb_image,
                "depth": depth_pixels,
                "segmentation_mask": segmentation_mask_buffer
                }
