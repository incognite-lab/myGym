"""
Minimal example: use Pinocchio for 6-DOF (position + orientation) inverse
kinematics on a URDF robot, then mirror the resulting joint configuration
into PyBullet for physics simulation / visualization.

Install:
    pip install pin pybullet numpy --break-system-packages

Notes:
- Pinocchio does the IK math (fast, analytic Jacobians, CLIK solver).
- PyBullet only receives the solved joint angles and steps physics/render.
- Swap URDF_PATH and END_EFFECTOR_FRAME for your robot (e.g. Cyril's arm).
"""

import numpy as np
import pinocchio as pin
import pybullet as pb
import pybullet_data

URDF_PATH = "./envs/robots/unitree/g1_mygym.urdf"
END_EFFECTOR_FRAME = "endeffector"   # frame name in the URDF

# ---------------------------------------------------------------------
# 1. Load the model into Pinocchio (kinematics/IK only, no physics)
# ---------------------------------------------------------------------
model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()
ee_frame_id = model.getFrameId(END_EFFECTOR_FRAME)

def solve_ik(target_pos, target_quat_xyzw, q_init=None,
             max_iters=200, eps=1e-4, damp=1e-6):
    """
    CLIK (closed-loop inverse kinematics) via damped least squares.
    target_pos: (3,) array
    target_quat_xyzw: (4,) array, scalar-last quaternion
    Returns: joint configuration q (model.nq,)
    """
    R = pin.Quaternion(np.array(target_quat_xyzw)).toRotationMatrix()
    target_SE3 = pin.SE3(R, np.array(target_pos))

    q = q_init.copy() if q_init is not None else pin.neutral(model)

    for _ in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        current_SE3 = data.oMf[ee_frame_id]
        err = pin.log6(current_SE3.inverse() * target_SE3).vector  # 6D error

        if np.linalg.norm(err) < eps:
            break

        J = pin.computeFrameJacobian(
            model, data, q, ee_frame_id, pin.ReferenceFrame.LOCAL
        )
        # Damped least squares step
        JJt = J @ J.T + damp * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        q = pin.integrate(model, q, dq)

    return q

# ---------------------------------------------------------------------
# 2. Set up PyBullet purely for physics/visualization
# ---------------------------------------------------------------------
pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.81)
pb.loadURDF("plane.urdf")
robot_id = pb.loadURDF(URDF_PATH, useFixedBase=True)

# Map Pinocchio joint order -> PyBullet joint indices (names must match URDF)
pb_joint_indices = []
for j_name in model.names[1:]:  # skip "universe"
    for i in range(pb.getNumJoints(robot_id)):
        info = pb.getJointInfo(robot_id, i)
        if info[1].decode() == j_name:
            pb_joint_indices.append(i)
            break

def apply_to_pybullet(q):
    for idx, joint_idx in enumerate(pb_joint_indices):
        pb.resetJointState(robot_id, joint_idx, q[idx])

# ---------------------------------------------------------------------
# 3. Example: solve IK for a target pose and push it into PyBullet
# ---------------------------------------------------------------------
if __name__ == "__main__":
    target_position = [0.4, 0.1, 0.5]
    target_orientation_xyzw = [0.0, 0.0, 0.0, 1.0]  # identity orientation

    q_solution = solve_ik(target_position, target_orientation_xyzw)
    apply_to_pybullet(q_solution)

    print("Solved joint configuration:", q_solution)

    while pb.isConnected():
        pb.stepSimulation()
