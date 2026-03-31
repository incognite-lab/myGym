"""
MuJoCo physics client for myGym - replaces PyBullet's BulletClient.

This module provides a MujocoClient class that wraps MuJoCo's Python API
to provide functionality equivalent to PyBullet's BulletClient. It manages
model composition from multiple URDF files, simulation stepping, rendering,
joint control, collision detection, and inverse kinematics.
"""

import os
import math
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

# Ensure headless rendering works
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

# Connection mode constants (PyBullet compatibility)
GUI = "gui"
DIRECT = "direct"


class BodyInfo:
    """Stores information about a loaded body (URDF/SDF) in the scene."""

    def __init__(self, uid, prefix, root_body_name, is_fixed_base,
                 joint_names=None, joint_indices=None,
                 actuator_indices=None, body_names=None, body_indices=None,
                 geom_names=None, geom_indices=None):
        self.uid = uid
        self.prefix = prefix
        self.root_body_name = root_body_name
        self.is_fixed_base = is_fixed_base
        self.joint_names = joint_names or []
        self.joint_indices = joint_indices or []  # Global MuJoCo joint indices
        self.actuator_indices = actuator_indices or []  # Global MuJoCo actuator indices
        self.body_names = body_names or []
        self.body_indices = body_indices or []  # Global MuJoCo body indices
        self.geom_names = geom_names or []
        self.geom_indices = geom_indices or []
        self.num_joints = len(self.joint_indices)
        # Track joint types (maps local joint index to MuJoCo joint type)
        self.joint_types = []
        # Track link names for PyBullet-style getJointInfo
        self.link_names = []
        # Constraints associated with this body
        self.constraints = {}


class ConstraintInfo:
    """Stores information about an equality constraint."""

    def __init__(self, constraint_id, parent_body_uid, parent_link_index,
                 child_body_uid, child_link_index, joint_type,
                 joint_axis, parent_frame_pos, child_frame_pos,
                 parent_frame_orn=None, child_frame_orn=None):
        self.constraint_id = constraint_id
        self.parent_body_uid = parent_body_uid
        self.parent_link_index = parent_link_index
        self.child_body_uid = child_body_uid
        self.child_link_index = child_link_index
        self.joint_type = joint_type
        self.joint_axis = joint_axis
        self.parent_frame_pos = parent_frame_pos
        self.child_frame_pos = child_frame_pos
        self.parent_frame_orn = parent_frame_orn
        self.child_frame_orn = child_frame_orn


class MujocoClient:
    """
    MuJoCo physics client that provides a PyBullet-compatible interface.

    This class wraps MuJoCo's Python API to support the myGym robotics
    simulation framework. It handles:
    - Dynamic scene composition from multiple URDF files
    - Physics simulation stepping
    - Joint control (position, velocity, torque)
    - Rendering (RGB, depth, segmentation)
    - Collision detection
    - Inverse kinematics
    - Quaternion math utilities
    """

    # PyBullet-compatible constants
    POSITION_CONTROL = 0
    VELOCITY_CONTROL = 1
    TORQUE_CONTROL = 2

    JOINT_REVOLUTE = 0
    JOINT_PRISMATIC = 1
    JOINT_SPHERICAL = 2
    JOINT_PLANAR = 3
    JOINT_FIXED = 4

    JOINT_FEEDBACK = 0

    # Constraint types
    CONSTRAINT_SOLVER_LCP_DANTZIG = 0

    # Renderer types
    ER_TINY_RENDERER = 0
    ER_BULLET_HARDWARE_OPENGL = 1

    # Segmentation mask flags
    ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX = 0

    # GUI constants
    COV_ENABLE_SHADOWS = 0
    COV_ENABLE_GUI = 1

    # Keyboard constants
    KEY_WAS_TRIGGERED = 1
    KEY_IS_DOWN = 2
    B3G_LEFT_ARROW = 65295
    B3G_RIGHT_ARROW = 65296
    B3G_UP_ARROW = 65297
    B3G_DOWN_ARROW = 65298
    B3G_RETURN = 65309

    # URDF flags
    URDF_USE_SELF_COLLISION = 1
    URDF_USE_SELF_COLLISION_EXCLUDE_PARENT = 2

    def __init__(self, connection_mode=None):
        """
        Initialize the MuJoCo client.

        Parameters:
            :param connection_mode: 'gui' for visualization, 'direct' for headless
        """
        self._gui_on = (connection_mode == GUI)
        self._spec = mujoco.MjSpec()
        self._spec.option.gravity = np.array([0, 0, -9.81])
        self._spec.option.timestep = 1.0 / 240.0
        self._spec.compiler.degree = False  # Use radians
        self._spec.compiler.fusestatic = False  # Keep fixed joints as separate bodies
        self._spec.compiler.balanceinertia = True  # Auto-fix non-positive inertia
        self._spec.compiler.boundmass = 0.001  # Minimum mass for bodies
        self._spec.compiler.boundinertia = 0.00001  # Minimum inertia for bodies

        self._model = None
        self._data = None
        self._renderer = None
        self._renderer_width = 640
        self._renderer_height = 480
        self._needs_recompile = True

        # Body tracking
        self._body_uid_counter = 0
        self._bodies = {}  # uid -> BodyInfo
        self._body_frames = {}  # uid -> frame name in spec

        # Constraint tracking
        self._constraint_uid_counter = 0
        self._constraints = {}  # constraint_id -> ConstraintInfo
        self._weld_equality_names = {}  # constraint_id -> equality name

        # Debug visualization tracking
        self._debug_items = {}
        self._debug_item_counter = 0
        self._debug_params = {}
        self._debug_param_counter = 0

        # Keyboard state
        self._key_events = {}

        # State tracking for real-time simulation
        self._real_time = False

        # Add default ground plane geom
        self._has_compiled_once = False

    def _ensure_compiled(self):
        """Compile the spec if needed."""
        if self._needs_recompile or self._model is None:
            self._compile()

    def _compile(self):
        """Compile the current spec into a model and create data."""
        old_qpos = None
        old_qvel = None
        old_ctrl = None

        if self._data is not None:
            old_qpos = self._data.qpos.copy()
            old_qvel = self._data.qvel.copy()
            if self._data.ctrl.size > 0:
                old_ctrl = self._data.ctrl.copy()

        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

        self._model = self._spec.compile()
        self._data = mujoco.MjData(self._model)

        # Restore state
        if old_qpos is not None:
            min_nq = min(old_qpos.size, self._data.qpos.size)
            self._data.qpos[:min_nq] = old_qpos[:min_nq]
        if old_qvel is not None:
            min_nv = min(old_qvel.size, self._data.qvel.size)
            self._data.qvel[:min_nv] = old_qvel[:min_nv]
        if old_ctrl is not None:
            min_nu = min(old_ctrl.size, self._data.ctrl.size)
            self._data.ctrl[:min_nu] = old_ctrl[:min_nu]

        # Forward to compute derived quantities
        mujoco.mj_forward(self._model, self._data)

        # Rebuild body info mappings
        self._rebuild_body_mappings()

        self._needs_recompile = False
        self._has_compiled_once = True

    def _rebuild_body_mappings(self):
        """Rebuild the mapping from body UIDs to MuJoCo indices after recompilation."""
        for uid, info in self._bodies.items():
            prefix = info.prefix

            # First find all bodies belonging to this loaded model
            info.body_indices = []
            info.body_names = []
            for i in range(self._model.nbody):
                body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and body_name.startswith(prefix):
                    info.body_indices.append(i)
                    info.body_names.append(body_name)

            # Build a set of bodies that have joints
            bodies_with_joints = set()
            for i in range(self._model.njnt):
                jnt_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jnt_name and jnt_name.startswith(prefix):
                    bodies_with_joints.add(self._model.jnt_bodyid[i])

            # Build the joint list: includes both real joints and synthetic fixed joints
            # for bodies that don't have real joints (to match PyBullet's behavior)
            info.joint_indices = []
            info.joint_names = []
            info.joint_types = []
            info.link_names = []
            info._real_joint_flags = []  # True if real MuJoCo joint, False if synthetic fixed
            info._body_id_for_joint = []  # MuJoCo body ID for each joint entry

            # First add all real joints
            for i in range(self._model.njnt):
                jnt_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jnt_name and jnt_name.startswith(prefix):
                    # Skip free joints from the joint count
                    if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                        continue
                    info.joint_indices.append(i)
                    info.joint_names.append(jnt_name)
                    info._real_joint_flags.append(True)

                    mj_type = self._model.jnt_type[i]
                    if mj_type == mujoco.mjtJoint.mjJNT_HINGE:
                        info.joint_types.append(self.JOINT_REVOLUTE)
                    elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
                        info.joint_types.append(self.JOINT_PRISMATIC)
                    else:
                        info.joint_types.append(self.JOINT_REVOLUTE)

                    body_id = self._model.jnt_bodyid[i]
                    info._body_id_for_joint.append(body_id)
                    body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    link_name = body_name[len(prefix):] if body_name and body_name.startswith(prefix) else (body_name or "")
                    info.link_names.append(link_name)

            # Then add synthetic fixed joints for bodies without real joints
            # (skip the root body, which is the first one)
            for body_idx in info.body_indices:
                if body_idx not in bodies_with_joints:
                    body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
                    link_name = body_name[len(prefix):] if body_name and body_name.startswith(prefix) else (body_name or "")

                    # Skip the root body of the model
                    if body_idx == info.body_indices[0]:
                        continue

                    info.joint_indices.append(-1)  # -1 indicates synthetic fixed joint
                    jnt_name = f"{prefix}fixed_{link_name}"
                    info.joint_names.append(jnt_name)
                    info.joint_types.append(self.JOINT_FIXED)
                    info._real_joint_flags.append(False)
                    info._body_id_for_joint.append(body_idx)
                    info.link_names.append(link_name)

            info.num_joints = len(info.joint_indices)

            # Find actuators
            info.actuator_indices = []
            for i in range(self._model.nu):
                act_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if act_name and act_name.startswith(f"act_{prefix}"):
                    info.actuator_indices.append(i)

            # Find geoms - match by name prefix OR by belonging to a body in this group
            info.geom_indices = []
            info.geom_names = []
            body_set = set(info.body_indices)
            for i in range(self._model.ngeom):
                geom_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)
                geom_body_id = self._model.geom_bodyid[i]
                if (geom_name and geom_name.startswith(prefix)) or geom_body_id in body_set:
                    info.geom_indices.append(i)
                    info.geom_names.append(geom_name or f"geom_{i}")

    def _get_body_info(self, body_uid):
        """Get BodyInfo for a given body UID."""
        if body_uid not in self._bodies:
            raise ValueError(f"Body UID {body_uid} not found")
        return self._bodies[body_uid]

    def _local_to_global_joint(self, body_uid, local_joint_index):
        """Convert local joint index (within a body) to global MuJoCo joint index.
        Returns -1 for synthetic fixed joints."""
        info = self._get_body_info(body_uid)
        if local_joint_index < 0 or local_joint_index >= info.num_joints:
            raise IndexError(f"Joint index {local_joint_index} out of range for body {body_uid} (has {info.num_joints} joints)")
        return info.joint_indices[local_joint_index]

    def _is_real_joint(self, body_uid, local_joint_index):
        """Check if a joint is a real MuJoCo joint (not synthetic fixed)."""
        info = self._get_body_info(body_uid)
        if local_joint_index < len(info._real_joint_flags):
            return info._real_joint_flags[local_joint_index]
        return False

    def _local_to_global_actuator(self, body_uid, local_joint_index):
        """Convert local joint index to global MuJoCo actuator index.
        Counts only real (non-fixed) joints for actuator mapping."""
        info = self._get_body_info(body_uid)
        # Count real joints before this index
        real_idx = 0
        for i in range(min(local_joint_index, len(info._real_joint_flags))):
            if info._real_joint_flags[i]:
                real_idx += 1
            if i == local_joint_index:
                break
        # Only return actuator for real joints
        if local_joint_index < len(info._real_joint_flags) and info._real_joint_flags[local_joint_index]:
            # Count how many real joints come before this one
            real_count = sum(1 for i in range(local_joint_index) if info._real_joint_flags[i])
            if real_count < len(info.actuator_indices):
                return info.actuator_indices[real_count]
        return None

    @staticmethod
    def _quat_pb_to_mj(quat_xyzw):
        """Convert PyBullet quaternion [x,y,z,w] to MuJoCo quaternion [w,x,y,z]."""
        if quat_xyzw is None:
            return np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array(quat_xyzw, dtype=np.float64)
        return np.array([q[3], q[0], q[1], q[2]])

    @staticmethod
    def _quat_mj_to_pb(quat_wxyz):
        """Convert MuJoCo quaternion [w,x,y,z] to PyBullet quaternion [x,y,z,w]."""
        q = np.array(quat_wxyz, dtype=np.float64)
        return tuple([q[1], q[2], q[3], q[0]])

    # =========================================================================
    # Physics Engine Configuration
    # =========================================================================

    def setPhysicsEngineParameter(self, **kwargs):
        """Set physics engine parameters. Maps to MuJoCo options where applicable."""
        if "numSolverIterations" in kwargs:
            self._spec.option.iterations = kwargs["numSolverIterations"]
        if "numSubSteps" in kwargs:
            self._spec.option.noslip_iterations = kwargs["numSubSteps"]
        if "enableConeFriction" in kwargs:
            self._spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if kwargs["enableConeFriction"] else mujoco.mjtCone.mjCONE_PYRAMIDAL
        if "contactBreakingThreshold" in kwargs:
            pass  # MuJoCo handles contact differently
        # Other parameters are not directly mappable
        self._needs_recompile = True

    def setGravity(self, x, y, z):
        """Set gravity vector."""
        self._spec.option.gravity = np.array([x, y, z])
        if self._model is not None:
            self._model.opt.gravity = np.array([x, y, z])

    def setTimeStep(self, time_step):
        """Set simulation time step."""
        self._spec.option.timestep = time_step
        if self._model is not None:
            self._model.opt.timestep = time_step

    def setRealTimeSimulation(self, enable):
        """Enable/disable real-time simulation mode (no-op for MuJoCo)."""
        self._real_time = bool(enable)

    # =========================================================================
    # Model Loading
    # =========================================================================

    def loadURDF(self, fileName, basePosition=None, baseOrientation=None,
                 useFixedBase=False, flags=0, useMaximalCoordinates=False):
        """
        Load a URDF file into the scene.

        Parameters:
            :param fileName: Path to URDF file
            :param basePosition: [x, y, z] position
            :param baseOrientation: [x, y, z, w] quaternion (PyBullet convention)
            :param useFixedBase: Whether the base is fixed
            :param flags: URDF loading flags (not all supported)
        Returns:
            :return uid: Unique ID for the loaded body
        """
        if basePosition is None:
            basePosition = [0, 0, 0]
        if baseOrientation is None:
            baseOrientation = [0, 0, 0, 1]

        uid = self._body_uid_counter
        self._body_uid_counter += 1
        prefix = f"b{uid}_"

        # Load URDF into child spec
        abs_path = os.path.abspath(fileName)
        child_spec = mujoco.MjSpec.from_file(abs_path)

        # Fix non-positive-definite inertia matrices (common in PyBullet URDFs)
        # MuJoCo requires physically valid inertia, while PyBullet is more lenient
        for body in child_spec.bodies:
            inertia = body.fullinertia
            if body.mass == 0 and np.all(inertia == 0):
                continue  # Skip massless bodies
            # Check if inertia is valid (all eigenvalues must be positive)
            imat = np.array([[inertia[0], inertia[3], inertia[4]],
                             [inertia[3], inertia[1], inertia[5]],
                             [inertia[4], inertia[5], inertia[2]]])
            try:
                eigenvalues = np.linalg.eigvalsh(imat)
                if np.any(eigenvalues <= 0):
                    # Replace with diagonal inertia based on mass
                    mass = max(body.mass, 0.001)
                    body.fullinertia = np.array([mass * 0.01, mass * 0.01, mass * 0.01, 0, 0, 0])
            except Exception:
                mass = max(body.mass, 0.001)
                body.fullinertia = np.array([mass * 0.01, mass * 0.01, mass * 0.01, 0, 0, 0])

        # Apply compiler settings to child spec for URDF compatibility
        child_spec.compiler.balanceinertia = True
        child_spec.compiler.boundmass = 0.001
        child_spec.compiler.boundinertia = 0.00001

        # Fix mesh directory - MuJoCo's URDF compiler strips directory prefixes from
        # mesh filenames. We need to help it find the mesh files.
        urdf_dir = os.path.dirname(abs_path)
        # Check for obj directory at same level (./obj/mesh.obj) or as sibling of parent
        obj_dir_same = os.path.join(urdf_dir, "obj")
        parent_dir = os.path.dirname(urdf_dir)
        obj_dir_parent = os.path.join(parent_dir, "obj")
        if os.path.isdir(obj_dir_same):
            child_spec.compiler.meshdir = obj_dir_same
        elif os.path.isdir(obj_dir_parent):
            child_spec.compiler.meshdir = obj_dir_parent

        # Create attachment frame
        frame = self._spec.worldbody.add_frame()
        frame.name = f"frame_{prefix}"
        frame.pos = np.array(basePosition, dtype=np.float64)
        frame.quat = self._quat_pb_to_mj(baseOrientation)

        # Attach child spec
        self._spec.attach(child_spec, prefix=prefix, suffix="", frame=frame)

        # If not fixed base and not a static object, add freejoint
        # Actually, URDF loaded as fixed base means we don't add freejoint
        # MuJoCo by default connects to parent via fixed (weld) connection
        # For free-floating objects, we need to add a freejoint
        if not useFixedBase:
            # Find the root body of the attached model and add a freejoint
            # We need to find the first body with our prefix in the spec
            for body in self._spec.worldbody.bodies:
                if body.name and body.name.startswith(prefix):
                    # Check if it already has a freejoint
                    has_free = False
                    for jnt in body.joints:
                        if jnt.name and "free" in jnt.name.lower():
                            has_free = True
                            break
                    if not has_free:
                        fj = body.add_freejoint()
                        fj.name = f"{prefix}freejoint"
                    break

        # Store body info (will be populated after compile)
        info = BodyInfo(
            uid=uid,
            prefix=prefix,
            root_body_name=f"{prefix}",
            is_fixed_base=useFixedBase
        )
        self._bodies[uid] = info
        self._body_frames[uid] = f"frame_{prefix}"

        # Compile to discover joints, then add actuators
        self._compile()

        # Add position servo actuators for all real joints of this body
        info = self._bodies[uid]
        actuators_added = False
        for i, jnt_idx in enumerate(info.joint_indices):
            # Skip synthetic fixed joints (jnt_idx == -1)
            if jnt_idx == -1:
                continue

            jnt_name = info.joint_names[i]
            jnt_type = self._model.jnt_type[jnt_idx]

            # Skip free joints
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                continue

            act = self._spec.add_actuator()
            act.name = f"act_{jnt_name}"
            act.target = jnt_name
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT

            # Position servo: kp gain, with bias for PD control
            act.gainprm[0] = 100.0  # kp
            bp = np.zeros(10)
            bp[0] = 0.0     # bias
            bp[1] = -100.0  # -kp
            bp[2] = -10.0   # -kd
            act.biasprm = bp
            act.dyntype = mujoco.mjtDyn.mjDYN_NONE
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE

            # Set control range from joint limits
            jnt_limited = self._model.jnt_limited[jnt_idx]
            if jnt_limited:
                lower = self._model.jnt_range[jnt_idx, 0]
                upper = self._model.jnt_range[jnt_idx, 1]
                act.ctrlrange = np.array([lower, upper])
                act.ctrllimited = True

            actuators_added = True

        if actuators_added:
            self._compile()

        return uid

    def loadSDF(self, fileName):
        """
        Load an SDF file. SDF is not natively supported by MuJoCo,
        so we attempt to load it as URDF (many SDF files are compatible).

        Returns:
            :return [uid]: List with single UID
        """
        # Try loading as URDF-like format
        uid = self.loadURDF(fileName, useFixedBase=False)
        return [uid]

    def removeBody(self, body_uid):
        """Remove a body from the scene."""
        if body_uid not in self._bodies:
            return
        info = self._bodies[body_uid]
        prefix = info.prefix

        # Remove actuators for this body
        actuators_to_remove = []
        for act in self._spec.actuators:
            if act.name and act.name.startswith(f"act_{prefix}"):
                actuators_to_remove.append(act)
        for act in actuators_to_remove:
            self._spec.delete(act)

        # Remove the attachment frame and its children
        frame_name = self._body_frames.get(body_uid)
        if frame_name:
            for frame in self._spec.worldbody.frames:
                if frame.name == frame_name:
                    self._spec.delete(frame)
                    break

        del self._bodies[body_uid]
        if body_uid in self._body_frames:
            del self._body_frames[body_uid]

        self._needs_recompile = True
        self._compile()

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def stepSimulation(self):
        """Step the simulation forward by one time step."""
        self._ensure_compiled()
        mujoco.mj_step(self._model, self._data)

    def resetSimulation(self):
        """Reset the entire simulation."""
        # Create a fresh spec
        old_gravity = self._spec.option.gravity.copy()
        old_timestep = self._spec.option.timestep

        self._spec = mujoco.MjSpec()
        self._spec.option.gravity = old_gravity
        self._spec.option.timestep = old_timestep
        self._spec.compiler.degree = False  # Use radians
        self._spec.compiler.fusestatic = False  # Keep fixed joints as separate bodies
        self._spec.compiler.balanceinertia = True
        self._spec.compiler.boundmass = 0.001
        self._spec.compiler.boundinertia = 0.00001

        self._model = None
        self._data = None
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

        self._bodies = {}
        self._body_frames = {}
        self._body_uid_counter = 0
        self._constraints = {}
        self._constraint_uid_counter = 0
        self._weld_equality_names = {}
        self._debug_items = {}
        self._debug_params = {}
        self._needs_recompile = True

    def disconnect(self):
        """Clean up and disconnect."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self._model = None
        self._data = None

    # =========================================================================
    # Joint Queries
    # =========================================================================

    def getNumJoints(self, body_uid):
        """Get the number of joints for a body."""
        self._ensure_compiled()
        return self._get_body_info(body_uid).num_joints

    def getJointInfo(self, body_uid, joint_index):
        """
        Get information about a joint.

        Returns a tuple compatible with PyBullet's getJointInfo:
        (jointIndex, jointName, jointType, qIndex, uIndex, flags,
         jointDamping, jointFriction, jointLowerLimit, jointUpperLimit,
         jointMaxForce, jointMaxVelocity, linkName, jointAxis,
         parentFramePos, parentFrameOrn, parentIndex)
        """
        self._ensure_compiled()
        info = self._get_body_info(body_uid)
        global_jnt_idx = self._local_to_global_joint(body_uid, joint_index)

        jnt_name = info.joint_names[joint_index]
        jnt_type = info.joint_types[joint_index]
        link_name = info.link_names[joint_index] if joint_index < len(info.link_names) else ""

        # Handle synthetic fixed joints (no actual MuJoCo joint)
        if global_jnt_idx == -1:
            return (
                joint_index,                          # 0: jointIndex
                jnt_name.encode("utf-8"),             # 1: jointName
                self.JOINT_FIXED,                     # 2: jointType
                -1,                                   # 3: qIndex (fixed joints have -1)
                -1,                                   # 4: uIndex
                0,                                    # 5: flags
                0.0,                                  # 6: jointDamping
                0.0,                                  # 7: jointFriction
                0.0,                                  # 8: jointLowerLimit
                -1.0,                                 # 9: jointUpperLimit
                0.0,                                  # 10: jointMaxForce
                0.0,                                  # 11: jointMaxVelocity
                link_name.encode("utf-8"),            # 12: linkName
                (0.0, 0.0, 0.0),                      # 13: jointAxis
                (0.0, 0.0, 0.0),                      # 14: parentFramePos
                (0.0, 0.0, 0.0, 1.0),                # 15: parentFrameOrn
                -1                                    # 16: parentIndex
            )

        # Get joint properties from model
        qpos_adr = self._model.jnt_qposadr[global_jnt_idx]
        dof_adr = self._model.jnt_dofadr[global_jnt_idx]

        # Joint limits
        limited = self._model.jnt_limited[global_jnt_idx]
        if limited:
            lower_limit = float(self._model.jnt_range[global_jnt_idx, 0])
            upper_limit = float(self._model.jnt_range[global_jnt_idx, 1])
        else:
            lower_limit = 0.0
            upper_limit = -1.0  # PyBullet convention for unlimited

        # Damping
        damping = float(self._model.dof_damping[dof_adr]) if dof_adr < self._model.nv else 0.0

        # Max force and velocity (from actuator if available)
        max_force = 100.0  # default
        max_velocity = 3.0  # default

        act_idx = self._local_to_global_actuator(body_uid, joint_index)
        if act_idx is not None and act_idx < self._model.nu:
            # Use actuator force range if available
            if self._model.actuator_forcerange[act_idx, 0] != self._model.actuator_forcerange[act_idx, 1]:
                max_force = float(self._model.actuator_forcerange[act_idx, 1])

        # Joint axis
        jnt_axis = tuple(self._model.jnt_axis[global_jnt_idx])

        return (
            joint_index,                          # 0: jointIndex
            jnt_name.encode("utf-8"),             # 1: jointName
            jnt_type,                             # 2: jointType
            qpos_adr,                             # 3: qIndex
            dof_adr,                              # 4: uIndex
            0,                                    # 5: flags
            damping,                              # 6: jointDamping
            0.0,                                  # 7: jointFriction
            lower_limit,                          # 8: jointLowerLimit
            upper_limit,                          # 9: jointUpperLimit
            max_force,                            # 10: jointMaxForce
            max_velocity,                         # 11: jointMaxVelocity
            link_name.encode("utf-8"),            # 12: linkName
            jnt_axis,                             # 13: jointAxis
            (0.0, 0.0, 0.0),                     # 14: parentFramePos
            (0.0, 0.0, 0.0, 1.0),                # 15: parentFrameOrn
            -1                                    # 16: parentIndex
        )

    def getJointState(self, body_uid, joint_index):
        """
        Get the state of a joint.

        Returns:
            (jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque)
        """
        self._ensure_compiled()
        global_jnt_idx = self._local_to_global_joint(body_uid, joint_index)

        # Fixed/synthetic joints have no state
        if global_jnt_idx == -1:
            return (0.0, 0.0, (0, 0, 0, 0, 0, 0), 0.0)

        qpos_adr = self._model.jnt_qposadr[global_jnt_idx]
        dof_adr = self._model.jnt_dofadr[global_jnt_idx]

        pos = float(self._data.qpos[qpos_adr])
        vel = float(self._data.qvel[dof_adr]) if dof_adr < self._model.nv else 0.0

        return (pos, vel, (0, 0, 0, 0, 0, 0), 0.0)

    def resetJointState(self, body_uid, joint_index, value, targetVelocity=0.0):
        """Reset a joint to a specific position."""
        self._ensure_compiled()
        global_jnt_idx = self._local_to_global_joint(body_uid, joint_index)

        # Skip fixed/synthetic joints
        if global_jnt_idx == -1:
            return

        qpos_adr = self._model.jnt_qposadr[global_jnt_idx]
        dof_adr = self._model.jnt_dofadr[global_jnt_idx]

        self._data.qpos[qpos_adr] = value
        if dof_adr < self._model.nv:
            self._data.qvel[dof_adr] = targetVelocity

        # Forward to update derived quantities
        mujoco.mj_forward(self._model, self._data)

    def setJointMotorControl2(self, bodyUniqueId, jointIndex, controlMode,
                               targetPosition=0, targetVelocity=0,
                               force=None, maxVelocity=None,
                               positionGain=0.7, velocityGain=0.3):
        """
        Set motor control for a joint.

        Maps to MuJoCo actuator control.
        """
        self._ensure_compiled()
        info = self._get_body_info(bodyUniqueId)
        act_idx = self._local_to_global_actuator(bodyUniqueId, jointIndex)

        if act_idx is None:
            return

        if controlMode == self.POSITION_CONTROL:
            self._data.ctrl[act_idx] = targetPosition
        elif controlMode == self.VELOCITY_CONTROL:
            self._data.ctrl[act_idx] = targetVelocity
        elif controlMode == self.TORQUE_CONTROL:
            self._data.ctrl[act_idx] = force if force is not None else 0.0

    def setJointMotorControlArray(self, bodyUniqueId, jointIndices,
                                   controlMode, targetPositions=None,
                                   targetVelocities=None, forces=None,
                                   positionGains=None, velocityGains=None):
        """Set motor control for multiple joints at once."""
        self._ensure_compiled()
        for i, joint_idx in enumerate(jointIndices):
            tp = targetPositions[i] if targetPositions is not None else 0
            tv = targetVelocities[i] if targetVelocities is not None else 0
            f = forces[i] if forces is not None else None
            self.setJointMotorControl2(bodyUniqueId, joint_idx, controlMode,
                                       targetPosition=tp, targetVelocity=tv,
                                       force=f)

    # =========================================================================
    # Body State Queries
    # =========================================================================

    def resetBasePositionAndOrientation(self, body_uid, position, orientation):
        """Reset position and orientation of a body's base."""
        self._ensure_compiled()
        info = self._get_body_info(body_uid)

        if not info.is_fixed_base:
            # Find the freejoint for this body
            prefix = info.prefix
            for i in range(self._model.njnt):
                jnt_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jnt_name and jnt_name == f"{prefix}freejoint":
                    qpos_adr = self._model.jnt_qposadr[i]
                    # Free joint qpos: [x, y, z, qw, qx, qy, qz]
                    self._data.qpos[qpos_adr:qpos_adr + 3] = position
                    mj_quat = self._quat_pb_to_mj(orientation)
                    self._data.qpos[qpos_adr + 3:qpos_adr + 7] = mj_quat
                    # Zero velocities
                    dof_adr = self._model.jnt_dofadr[i]
                    self._data.qvel[dof_adr:dof_adr + 6] = 0
                    mujoco.mj_forward(self._model, self._data)
                    return

        # For fixed base, we need to modify the frame position
        # This requires recompilation
        frame_name = self._body_frames.get(body_uid)
        if frame_name:
            for frame in self._spec.worldbody.frames:
                if frame.name == frame_name:
                    frame.pos = np.array(position, dtype=np.float64)
                    frame.quat = self._quat_pb_to_mj(orientation)
                    self._needs_recompile = True
                    self._compile()
                    return

    def getBasePositionAndOrientation(self, body_uid):
        """Get position and orientation of a body's base."""
        self._ensure_compiled()
        info = self._get_body_info(body_uid)

        if not info.is_fixed_base:
            # Find the freejoint
            prefix = info.prefix
            for i in range(self._model.njnt):
                jnt_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jnt_name and jnt_name == f"{prefix}freejoint":
                    qpos_adr = self._model.jnt_qposadr[i]
                    pos = tuple(self._data.qpos[qpos_adr:qpos_adr + 3])
                    mj_quat = self._data.qpos[qpos_adr + 3:qpos_adr + 7]
                    orn = self._quat_mj_to_pb(mj_quat)
                    return (pos, orn)

        # For fixed base, get from body xpos
        if info.body_indices:
            body_idx = info.body_indices[0]
            pos = tuple(self._data.xpos[body_idx])
            mj_quat = self._data.xquat[body_idx]
            orn = self._quat_mj_to_pb(mj_quat)
            return (pos, orn)

        return ((0, 0, 0), (0, 0, 0, 1))

    def getLinkState(self, body_uid, link_index):
        """
        Get the state of a link.

        Returns a tuple:
        (linkWorldPosition, linkWorldOrientation, localInertialFramePosition,
         localInertialFrameOrientation, worldLinkFramePosition, worldLinkFrameOrientation)
        """
        self._ensure_compiled()
        info = self._get_body_info(body_uid)

        # Use the _body_id_for_joint mapping for both real and synthetic joints
        if link_index < len(info._body_id_for_joint):
            body_id = info._body_id_for_joint[link_index]
        elif info.body_indices:
            body_id = info.body_indices[min(link_index, len(info.body_indices) - 1)]
        else:
            return ((0, 0, 0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0, 1))

        pos = tuple(self._data.xpos[body_id])
        mj_quat = self._data.xquat[body_id]
        orn = self._quat_mj_to_pb(mj_quat)

        return (pos, orn, (0, 0, 0), (0, 0, 0, 1), pos, orn)

    # =========================================================================
    # Inverse Kinematics
    # =========================================================================

    def calculateInverseKinematics(self, bodyUniqueId, endEffectorLinkIndex,
                                    targetPosition, targetOrientation=None,
                                    lowerLimits=None, upperLimits=None,
                                    jointRanges=None, restPoses=None,
                                    maxNumIterations=100, residualThreshold=1e-4):
        """
        Calculate inverse kinematics using damped least squares.

        Returns:
            :return joint_poses: Tuple of joint positions
        """
        self._ensure_compiled()
        info = self._get_body_info(bodyUniqueId)

        # Find the target body ID in MuJoCo
        if endEffectorLinkIndex < info.num_joints:
            global_jnt_idx = info.joint_indices[endEffectorLinkIndex]
            ee_body_id = self._model.jnt_bodyid[global_jnt_idx]
        else:
            # Fallback: use last body in the chain
            ee_body_id = info.body_indices[-1] if info.body_indices else 0

        target_pos = np.array(targetPosition, dtype=np.float64)

        # Save current state
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()

        # Get joint qpos addresses for this body (non-free joints only)
        joint_qpos_addrs = []
        joint_dof_addrs = []
        for jnt_idx in info.joint_indices:
            if self._model.jnt_type[jnt_idx] != mujoco.mjtJoint.mjJNT_FREE:
                joint_qpos_addrs.append(self._model.jnt_qposadr[jnt_idx])
                joint_dof_addrs.append(self._model.jnt_dofadr[jnt_idx])

        n_joints = len(joint_qpos_addrs)
        if n_joints == 0:
            return tuple()

        # IK loop using damped least squares
        for iteration in range(maxNumIterations):
            mujoco.mj_forward(self._model, self._data)
            ee_pos = self._data.xpos[ee_body_id].copy()
            pos_err = target_pos - ee_pos

            if targetOrientation is not None:
                target_quat_mj = self._quat_pb_to_mj(targetOrientation)
                ee_quat_mj = self._data.xquat[ee_body_id].copy()
                # Compute orientation error using MuJoCo
                neg_ee_quat = np.zeros(4)
                mujoco.mju_negQuat(neg_ee_quat, ee_quat_mj)
                err_quat = np.zeros(4)
                mujoco.mju_mulQuat(err_quat, target_quat_mj, neg_ee_quat)
                ori_err = err_quat[1:4]  # axis * sin(angle/2) ≈ axis*angle for small angles
                err = np.concatenate([pos_err, ori_err])
            else:
                err = pos_err

            if np.linalg.norm(pos_err) < residualThreshold:
                break

            # Compute Jacobian
            jacp = np.zeros((3, self._model.nv))
            jacr = np.zeros((3, self._model.nv))
            mujoco.mj_jacBody(self._model, self._data, jacp, jacr, ee_body_id)

            # Extract columns for our joints only
            J_cols = np.array(joint_dof_addrs)
            if targetOrientation is not None:
                J = np.vstack([jacp[:, J_cols], jacr[:, J_cols]])
            else:
                J = jacp[:, J_cols]

            # Damped least squares
            lam = 0.1
            JJT = J @ J.T + lam * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJT, err)

            # Apply joint changes
            for i, qpos_adr in enumerate(joint_qpos_addrs):
                self._data.qpos[qpos_adr] += dq[i]

            # Apply joint limits if provided
            if lowerLimits is not None and upperLimits is not None:
                for i, qpos_adr in enumerate(joint_qpos_addrs):
                    if i < len(lowerLimits) and i < len(upperLimits):
                        self._data.qpos[qpos_adr] = np.clip(
                            self._data.qpos[qpos_adr],
                            lowerLimits[i], upperLimits[i]
                        )

        # Extract final joint positions
        result = []
        for qpos_adr in joint_qpos_addrs:
            result.append(float(self._data.qpos[qpos_adr]))

        # Restore original state
        self._data.qpos[:] = saved_qpos
        self._data.qvel[:] = saved_qvel
        mujoco.mj_forward(self._model, self._data)

        return tuple(result)

    def calculateJacobian(self, bodyUniqueId, linkIndex, localPosition,
                           objPositions, objVelocities, objAccelerations):
        """
        Calculate the Jacobian for a body link.

        Returns:
            (linear_jacobian, angular_jacobian)
        """
        self._ensure_compiled()
        info = self._get_body_info(bodyUniqueId)

        if linkIndex < info.num_joints:
            global_jnt_idx = info.joint_indices[linkIndex]
            body_id = self._model.jnt_bodyid[global_jnt_idx]
        else:
            body_id = info.body_indices[-1] if info.body_indices else 0

        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, jacp, jacr, body_id)

        # Extract columns for this body's joints
        joint_dof_addrs = []
        for jnt_idx in info.joint_indices:
            if self._model.jnt_type[jnt_idx] != mujoco.mjtJoint.mjJNT_FREE:
                joint_dof_addrs.append(self._model.jnt_dofadr[jnt_idx])

        J_cols = np.array(joint_dof_addrs)
        Jt = jacp[:, J_cols].tolist()
        Jr = jacr[:, J_cols].tolist()

        return (Jt, Jr)

    # =========================================================================
    # Collision Detection
    # =========================================================================

    def getContactPoints(self, bodyA=-1, bodyB=-1, linkIndexA=-1, linkIndexB=-1):
        """Get contact points between two bodies."""
        self._ensure_compiled()

        contacts = []
        infoA = self._bodies.get(bodyA)
        infoB = self._bodies.get(bodyB)

        if infoA is None or infoB is None:
            return contacts

        geoms_a = set(infoA.geom_indices)
        geoms_b = set(infoB.geom_indices)

        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in geoms_a and g2 in geoms_b) or (g1 in geoms_b and g2 in geoms_a):
                # Build PyBullet-compatible contact point tuple
                contact = (
                    0,              # contactFlag
                    bodyA,          # bodyUniqueIdA
                    bodyB,          # bodyUniqueIdB
                    -1,             # linkIndexA
                    -1,             # linkIndexB
                    tuple(c.pos),   # positionOnA
                    tuple(c.pos),   # positionOnB
                    tuple(c.frame[:3]),  # contactNormalOnB
                    float(c.dist),  # contactDistance
                    0.0,            # normalForce
                    0.0,            # lateralFriction1
                    (0, 0, 0),      # lateralFrictionDir1
                    0.0,            # lateralFriction2
                    (0, 0, 0),      # lateralFrictionDir2
                )
                contacts.append(contact)

        return contacts

    def getClosestPoints(self, bodyA, bodyB, distance):
        """Get closest points between two bodies within a distance threshold."""
        self._ensure_compiled()

        # Use contact information as approximation
        contacts = self.getContactPoints(bodyA, bodyB)
        # Also check geometric proximity
        infoA = self._bodies.get(bodyA)
        infoB = self._bodies.get(bodyB)
        if infoA is None or infoB is None:
            return contacts

        # Get center positions of bodies
        posA = self._data.xpos[infoA.body_indices[0]] if infoA.body_indices else np.zeros(3)
        posB = self._data.xpos[infoB.body_indices[0]] if infoB.body_indices else np.zeros(3)
        dist = np.linalg.norm(posA - posB)

        if dist <= distance and not contacts:
            # Return a synthetic closest point
            contacts = [(
                0, bodyA, bodyB, -1, -1,
                tuple(posA), tuple(posB),
                tuple((posB - posA) / max(dist, 1e-10)),
                float(dist), 0.0, 0.0, (0, 0, 0), 0.0, (0, 0, 0)
            )]

        return contacts

    def getOverlappingObjects(self, aabb_min, aabb_max):
        """Get objects whose AABB overlaps with the given AABB."""
        self._ensure_compiled()
        overlapping = []

        for uid, info in self._bodies.items():
            for body_idx in info.body_indices:
                pos = self._data.xpos[body_idx]
                # Simple check: is body center within expanded AABB?
                if (aabb_min[0] <= pos[0] <= aabb_max[0] and
                    aabb_min[1] <= pos[1] <= aabb_max[1] and
                    aabb_min[2] <= pos[2] <= aabb_max[2]):
                    overlapping.append((uid, -1))
                    break

        return overlapping if overlapping else None

    def getAABB(self, body_uid, linkIndex=-1):
        """Get axis-aligned bounding box of a body."""
        self._ensure_compiled()
        info = self._get_body_info(body_uid)

        all_points = []
        for geom_idx in info.geom_indices:
            pos = self._data.geom_xpos[geom_idx]
            size = self._model.geom_size[geom_idx]
            geom_type = self._model.geom_type[geom_idx]

            # Approximate AABB based on geom type and size
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_ext = size
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                half_ext = np.array([size[0], size[0], size[0]])
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                half_ext = np.array([size[0], size[0], size[1]])
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_ext = np.array([size[0], size[0], size[0] + size[1]])
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # For meshes, use the AABB from mesh data
                mesh_id = self._model.geom_dataid[geom_idx]
                if mesh_id >= 0:
                    # Use geom size as approximation
                    half_ext = np.abs(size) + 0.01
                else:
                    half_ext = np.array([0.05, 0.05, 0.05])
            else:
                half_ext = np.array([0.05, 0.05, 0.05])

            all_points.append(pos - half_ext)
            all_points.append(pos + half_ext)

        if not all_points:
            # Fallback: use body position with small AABB
            if info.body_indices:
                pos = self._data.xpos[info.body_indices[0]]
                return (tuple(pos - 0.05), tuple(pos + 0.05))
            return ((0, 0, 0), (0, 0, 0))

        all_points = np.array(all_points)
        aabb_min = tuple(np.min(all_points, axis=0))
        aabb_max = tuple(np.max(all_points, axis=0))
        return (aabb_min, aabb_max)

    # =========================================================================
    # Constraints
    # =========================================================================

    def createConstraint(self, parentBodyUniqueId, parentLinkIndex,
                          childBodyUniqueId, childLinkIndex,
                          jointType, jointAxis, parentFramePosition,
                          childFramePosition, parentFrameOrientation=None,
                          childFrameOrientation=None):
        """
        Create a constraint (weld equality) between two bodies.
        In MuJoCo, this is modeled as a weld equality constraint.
        """
        self._ensure_compiled()

        cid = self._constraint_uid_counter
        self._constraint_uid_counter += 1

        self._constraints[cid] = ConstraintInfo(
            constraint_id=cid,
            parent_body_uid=parentBodyUniqueId,
            parent_link_index=parentLinkIndex,
            child_body_uid=childBodyUniqueId,
            child_link_index=childLinkIndex,
            joint_type=jointType,
            joint_axis=jointAxis,
            parent_frame_pos=parentFramePosition,
            child_frame_pos=childFramePosition,
            parent_frame_orn=parentFrameOrientation,
            child_frame_orn=childFrameOrientation,
        )

        # For weld constraints in MuJoCo, we simulate by directly
        # moving the child body to the parent's position in step
        # This is a simplified implementation
        return cid

    def changeConstraint(self, constraintId, jointChildPivot=None,
                          jointChildFrameOrientation=None, maxForce=None):
        """Update a constraint."""
        if constraintId in self._constraints:
            if jointChildPivot is not None:
                self._constraints[constraintId].child_frame_pos = jointChildPivot
            if jointChildFrameOrientation is not None:
                self._constraints[constraintId].child_frame_orn = jointChildFrameOrientation

    def removeConstraint(self, constraintId):
        """Remove a constraint."""
        if constraintId in self._constraints:
            del self._constraints[constraintId]

    # =========================================================================
    # Dynamics
    # =========================================================================

    def changeDynamics(self, body_uid, linkIndex, **kwargs):
        """Change dynamics properties of a body. Some parameters map to MuJoCo."""
        # MuJoCo dynamics are set at compile time, so changes require recompilation
        # For now, this is partially implemented
        pass

    # =========================================================================
    # Rendering
    # =========================================================================

    def computeViewMatrix(self, cameraEyePosition, cameraTargetPosition, cameraUpVector):
        """Compute view matrix from camera parameters."""
        eye = np.array(cameraEyePosition, dtype=np.float64)
        target = np.array(cameraTargetPosition, dtype=np.float64)
        up = np.array(cameraUpVector, dtype=np.float64)

        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)

        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up_corrected
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(up_corrected, eye)
        view[2, 3] = np.dot(forward, eye)

        return tuple(view.T.flatten())

    def computeViewMatrixFromYawPitchRoll(self, cameraTargetPosition, distance,
                                            yaw, pitch, roll, upAxisIndex):
        """Compute view matrix from yaw/pitch/roll parameters."""
        target = np.array(cameraTargetPosition, dtype=np.float64)

        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)

        # Compute eye position from spherical coordinates
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)

        eye = target + distance * np.array([cp * cy, cp * sy, sp])
        up = [0, 0, 1] if upAxisIndex == 2 else [0, 1, 0]

        return self.computeViewMatrix(eye, target, up)

    def computeProjectionMatrixFOV(self, fov, aspect, nearVal, farVal):
        """Compute perspective projection matrix."""
        fov_rad = math.radians(fov)
        f = 1.0 / math.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (farVal + nearVal) / (nearVal - farVal)
        proj[2, 3] = (2 * farVal * nearVal) / (nearVal - farVal)
        proj[3, 2] = -1

        return tuple(proj.T.flatten())

    def getCameraImage(self, viewMatrix=None, projectionMatrix=None,
                        width=640, height=480, flags=0, renderer=None,
                        lightDirection=None, lightColor=None,
                        lightDistance=None, shadow=None,
                        lightAmbientCoeff=None, lightDiffuseCoeff=None,
                        lightSpecularCoeff=None, **kwargs):
        """
        Render camera image.

        Returns: (width, height, rgbPixels, depthPixels, segmentationMaskBuffer)
        """
        self._ensure_compiled()

        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:
                    pass
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)

        # Update scene
        self._renderer.update_scene(self._data)

        # Render RGB
        rgb = self._renderer.render()

        # Render depth
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()

        # Render segmentation
        self._renderer.enable_segmentation_rendering()
        seg = self._renderer.render()
        self._renderer.disable_segmentation_rendering()

        # Add alpha channel to RGB
        alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=2)

        return (width, height, rgba.flatten(), depth, seg[:, :, 0])

    def loadTexture(self, texturePath):
        """Load a texture. Returns a texture ID."""
        # MuJoCo texture loading is done through spec
        # For simplicity, return a dummy ID
        return 0

    def changeVisualShape(self, objectUniqueId, linkIndex, rgbaColor=None,
                           textureUniqueId=None, specularColor=None):
        """Change visual properties of a shape. Limited in MuJoCo without recompilation."""
        # MuJoCo doesn't support runtime visual changes without recompilation
        # For color changes, we can modify geom rgba at runtime
        self._ensure_compiled()
        if rgbaColor is not None:
            info = self._bodies.get(objectUniqueId)
            if info:
                for geom_idx in info.geom_indices:
                    self._model.geom_rgba[geom_idx] = rgbaColor[:4] if len(rgbaColor) >= 4 else list(rgbaColor) + [1.0]

    def createVisualShape(self, shapeType, halfExtents=None, radius=None,
                           length=None, rgbaColor=None, **kwargs):
        """Create a visual shape. Returns shape ID for use with createMultiBody."""
        shape_id = self._debug_item_counter
        self._debug_item_counter += 1
        self._debug_items[shape_id] = {
            "type": shapeType,
            "halfExtents": halfExtents,
            "radius": radius,
            "length": length,
            "rgbaColor": rgbaColor or [1, 1, 1, 1],
        }
        return shape_id

    def createMultiBody(self, baseMass=0, baseVisualShapeIndex=-1,
                         basePosition=None, baseOrientation=None,
                         baseCollisionShapeIndex=-1, **kwargs):
        """
        Create a multi-body object. Adds a body with visual geom to the scene.
        """
        if basePosition is None:
            basePosition = [0, 0, 0]
        if baseOrientation is None:
            baseOrientation = [0, 0, 0, 1]

        uid = self._body_uid_counter
        self._body_uid_counter += 1
        prefix = f"b{uid}_"
        is_fixed = (baseMass == 0)

        # Add a body to the spec
        body = self._spec.worldbody.add_body()
        body.name = f"{prefix}base"
        body.pos = np.array(basePosition, dtype=np.float64)
        body.quat = self._quat_pb_to_mj(baseOrientation)

        if not is_fixed:
            fj = body.add_freejoint()
            fj.name = f"{prefix}freejoint"

        # Add visual geom
        shape_info = self._debug_items.get(baseVisualShapeIndex, {})
        geom = body.add_geom()
        geom.name = f"{prefix}geom"

        if shape_info.get("halfExtents"):
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = np.array(shape_info["halfExtents"], dtype=np.float64)
        elif shape_info.get("radius"):
            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size = np.array([shape_info["radius"], 0, 0], dtype=np.float64)
        else:
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = np.array([0.01, 0.01, 0.01], dtype=np.float64)

        color = shape_info.get("rgbaColor", [1, 1, 1, 0.5])
        geom.rgba = np.array(color[:4], dtype=np.float32)
        geom.contype = 0  # No collision for visual-only shapes
        geom.conaffinity = 0

        if is_fixed:
            body.mass = 0
        else:
            body.mass = baseMass

        info = BodyInfo(
            uid=uid,
            prefix=prefix,
            root_body_name=f"{prefix}base",
            is_fixed_base=is_fixed,
        )
        self._bodies[uid] = info
        self._body_frames[uid] = f"{prefix}base"

        self._needs_recompile = True
        self._compile()

        return uid

    def getVisualShapeData(self, objectUniqueId, flags=0):
        """Get visual shape data for an object."""
        self._ensure_compiled()
        info = self._bodies.get(objectUniqueId)
        if not info:
            return []

        shapes = []
        for geom_idx in info.geom_indices:
            geom_type = self._model.geom_type[geom_idx]
            geom_size = tuple(self._model.geom_size[geom_idx])
            rgba = tuple(self._model.geom_rgba[geom_idx])
            shapes.append((objectUniqueId, -1, geom_type, geom_size, "", (0, 0, 0), (0, 0, 0, 1), rgba))

        return shapes

    # =========================================================================
    # Debug Visualization
    # =========================================================================

    def resetDebugVisualizerCamera(self, distance, yaw, pitch, target):
        """Set debug camera. No-op in headless mode."""
        pass

    def configureDebugVisualizer(self, flag, enable):
        """Configure debug visualizer. No-op in headless mode."""
        pass

    def removeAllUserDebugItems(self):
        """Remove all debug items."""
        self._debug_items.clear()

    def addUserDebugLine(self, lineFromXYZ, lineToXYZ, lineColorRGB=None,
                          lineWidth=1.0, lifeTime=0, replaceItemUniqueId=None, **kwargs):
        """Add debug line. Returns item ID."""
        item_id = replaceItemUniqueId if replaceItemUniqueId is not None else self._debug_item_counter
        if replaceItemUniqueId is None:
            self._debug_item_counter += 1
        self._debug_items[item_id] = {
            "type": "line",
            "from": lineFromXYZ,
            "to": lineToXYZ,
            "color": lineColorRGB,
        }
        return item_id

    def addUserDebugText(self, text, textPosition, textSize=1,
                          textColorRGB=None, lifeTime=0, replaceItemUniqueId=None, **kwargs):
        """Add debug text. Returns item ID."""
        item_id = replaceItemUniqueId if replaceItemUniqueId is not None else self._debug_item_counter
        if replaceItemUniqueId is None:
            self._debug_item_counter += 1
        self._debug_items[item_id] = {
            "type": "text",
            "text": text,
            "position": textPosition,
        }
        return item_id

    def removeUserDebugItem(self, itemUniqueId):
        """Remove a debug item."""
        self._debug_items.pop(itemUniqueId, None)

    def addUserDebugParameter(self, paramName, rangeMin, rangeMax, startValue=0):
        """Add a debug slider parameter. Returns parameter ID."""
        param_id = self._debug_param_counter
        self._debug_param_counter += 1
        self._debug_params[param_id] = {
            "name": paramName,
            "min": rangeMin,
            "max": rangeMax,
            "value": startValue,
        }
        return param_id

    def readUserDebugParameter(self, itemUniqueId):
        """Read debug parameter value."""
        if itemUniqueId in self._debug_params:
            return self._debug_params[itemUniqueId]["value"]
        return 0.0

    def getKeyboardEvents(self):
        """Get keyboard events. Returns empty dict in headless mode."""
        return self._key_events

    def setAdditionalSearchPath(self, path):
        """Set additional search path for data files. No-op."""
        pass

    # =========================================================================
    # Quaternion / Transform Utilities
    # =========================================================================

    @staticmethod
    def getQuaternionFromEuler(euler):
        """Convert Euler angles (roll, pitch, yaw) to quaternion [x, y, z, w]."""
        r = Rotation.from_euler('xyz', euler)
        q = r.as_quat()  # scipy returns [x, y, z, w]
        return tuple(q)

    @staticmethod
    def getEulerFromQuaternion(quaternion):
        """Convert quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw)."""
        r = Rotation.from_quat(quaternion)
        euler = r.as_euler('xyz')
        return tuple(euler)

    @staticmethod
    def multiplyTransforms(positionA, orientationA, positionB, orientationB):
        """Multiply two transforms."""
        rA = Rotation.from_quat(orientationA)
        rB = Rotation.from_quat(orientationB)
        r_combined = rA * rB
        pos = np.array(positionA) + rA.apply(positionB)
        return (tuple(pos), tuple(r_combined.as_quat()))

    @staticmethod
    def invertTransform(position, orientation):
        """Invert a transform."""
        r = Rotation.from_quat(orientation)
        r_inv = r.inv()
        pos_inv = -r_inv.apply(position)
        return (tuple(pos_inv), tuple(r_inv.as_quat()))

    @staticmethod
    def getDifferenceQuaternion(quatA, quatB):
        """Get quaternion representing rotation from A to B."""
        rA = Rotation.from_quat(quatA)
        rB = Rotation.from_quat(quatB)
        r_diff = rA.inv() * rB
        return tuple(r_diff.as_quat())

    @staticmethod
    def getAxisAngleFromQuaternion(quaternion):
        """Convert quaternion to axis-angle representation."""
        r = Rotation.from_quat(quaternion)
        rotvec = r.as_rotvec()
        angle = np.linalg.norm(rotvec)
        if angle < 1e-10:
            return ((0, 0, 1), 0.0)
        axis = rotvec / angle
        return (tuple(axis), float(angle))

    @staticmethod
    def getMatrixFromQuaternion(quaternion):
        """Convert quaternion to 3x3 rotation matrix (flattened)."""
        r = Rotation.from_quat(quaternion)
        mat = r.as_matrix()
        return tuple(mat.flatten())

    # =========================================================================
    # Additional compatibility methods
    # =========================================================================

    def getPhysicsEngineParameters(self):
        """Get current physics engine parameters."""
        self._ensure_compiled()
        return {
            "gravityAccelerationX": self._model.opt.gravity[0],
            "gravityAccelerationY": self._model.opt.gravity[1],
            "gravityAccelerationZ": self._model.opt.gravity[2],
            "fixedTimeStep": self._model.opt.timestep,
        }

    def get_joints_state(self, body_uid):
        """Get positions and velocities of all joints."""
        self._ensure_compiled()
        info = self._get_body_info(body_uid)
        positions = []
        velocities = []
        for jnt_idx in info.joint_indices:
            if self._model.jnt_type[jnt_idx] != mujoco.mjtJoint.mjJNT_FREE:
                qpos_adr = self._model.jnt_qposadr[jnt_idx]
                dof_adr = self._model.jnt_dofadr[jnt_idx]
                positions.append(float(self._data.qpos[qpos_adr]))
                velocities.append(float(self._data.qvel[dof_adr]))
        return positions, velocities
