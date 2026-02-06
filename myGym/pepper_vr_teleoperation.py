#!/usr/bin/env python3
"""
Pepper Robot Teleoperation Script with Meta Quest 3 Support

This script enables teleoperation of the Pepper robot using Meta Quest 3 VR headset
and controllers via OpenXR or ALVR. It provides:
- Head tracking from VR headset to control robot head orientation
- Right hand controller position mapped to right arm end-effector
- Left hand controller position mapped to left arm end-effector (if available)
- IK solver for precise arm control

Requirements:
- pyopenxr (pip install pyopenxr) for OpenXR support
- OR socket connection to ALVR server
- pybullet for physics simulation and IK
- numpy for math operations

Usage:
    # With OpenXR:
    python pepper_vr_teleoperation.py --mode openxr
    
    # With simulated input (testing):
    python pepper_vr_teleoperation.py --mode simulated
    
    # With ALVR (socket-based):
    python pepper_vr_teleoperation.py --mode alvr --alvr-host 127.0.0.1 --alvr-port 9943

Author: myGym Team
License: MIT
"""

import pybullet as p
import pybullet_data
import numpy as np
import argparse
import time
import sys
import os
import json
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

# Try importing VR libraries
try:
    import xr
    OPENXR_AVAILABLE = True
except ImportError:
    OPENXR_AVAILABLE = False
    print("Warning: pyopenxr not available. Install with: pip install pyopenxr")

try:
    import socket
    SOCKET_AVAILABLE = True
except ImportError:
    SOCKET_AVAILABLE = False


@dataclass
class VRPose:
    """Represents a VR device pose (position and orientation)"""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # quaternion [x, y, z, w]
    
    def __init__(self, position=None, orientation=None):
        self.position = position if position is not None else np.array([0.0, 0.0, 0.0])
        self.orientation = orientation if orientation is not None else np.array([0.0, 0.0, 0.0, 1.0])


class VRInputHandler:
    """Base class for VR input handling"""
    
    def __init__(self):
        self.headset_pose = VRPose()
        self.left_controller_pose = VRPose()
        self.right_controller_pose = VRPose()
        
    def update(self) -> bool:
        """Update VR poses. Returns True if successful."""
        raise NotImplementedError
        
    def close(self):
        """Clean up VR resources"""
        pass


class SimulatedVRInput(VRInputHandler):
    """Simulated VR input for testing without hardware"""
    
    def __init__(self):
        super().__init__()
        self.time_offset = time.time()
        print("Using simulated VR input (for testing)")
        
    def update(self) -> bool:
        """Generate simulated VR poses"""
        t = time.time() - self.time_offset
        
        # Simulate headset - slowly moving in a circle
        self.headset_pose.position = np.array([
            0.3 * np.sin(t * 0.3),
            0.3 * np.cos(t * 0.3),
            1.6  # Head height
        ])
        # Looking forward with slight rotation
        yaw = t * 0.1
        self.headset_pose.orientation = p.getQuaternionFromEuler([0, 0, yaw])
        
        # Simulate right controller - moving hand
        self.right_controller_pose.position = np.array([
            0.3 + 0.15 * np.sin(t * 0.5),
            0.2 + 0.1 * np.cos(t * 0.7),
            1.2 + 0.1 * np.sin(t * 0.6)
        ])
        self.right_controller_pose.orientation = p.getQuaternionFromEuler([0, np.pi/2, 0])
        
        # Simulate left controller
        self.left_controller_pose.position = np.array([
            -0.3 + 0.1 * np.sin(t * 0.4),
            0.2 + 0.1 * np.cos(t * 0.6),
            1.2 + 0.1 * np.sin(t * 0.5)
        ])
        self.left_controller_pose.orientation = p.getQuaternionFromEuler([0, -np.pi/2, 0])
        
        return True


class OpenXRInput(VRInputHandler):
    """OpenXR input handler for Meta Quest 3"""
    
    def __init__(self):
        super().__init__()
        if not OPENXR_AVAILABLE:
            raise ImportError("pyopenxr is not installed. Install with: pip install pyopenxr")
        
        # Initialize OpenXR
        try:
            self.instance = xr.create_instance()
            self.system = self.instance.get_system()
            self.session = self.system.create_session()
            print("OpenXR initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenXR: {e}")
    
    def update(self) -> bool:
        """Update VR poses from OpenXR"""
        try:
            # Poll events
            self.instance.poll_events()
            
            # Get frame state
            frame_state = self.session.wait_frame()
            
            # Get view poses (headset)
            views = self.session.locate_views(frame_state)
            if views:
                # Average position from both eyes for headset position
                left_pos = np.array(views[0].pose.position)
                right_pos = np.array(views[1].pose.position)
                self.headset_pose.position = (left_pos + right_pos) / 2.0
                self.headset_pose.orientation = np.array(views[0].pose.orientation)
            
            # Get controller poses
            # Note: Actual implementation depends on OpenXR extension for controller tracking
            # This is a simplified version
            
            return True
        except Exception as e:
            print(f"Error updating OpenXR poses: {e}")
            return False
    
    def close(self):
        """Clean up OpenXR resources"""
        if hasattr(self, 'session'):
            self.session.destroy()
        if hasattr(self, 'instance'):
            self.instance.destroy()


class ALVRInput(VRInputHandler):
    """ALVR input handler using socket connection"""
    
    def __init__(self, host='127.0.0.1', port=9943):
        super().__init__()
        if not SOCKET_AVAILABLE:
            raise ImportError("socket module not available")
        
        self.host = host
        self.port = port
        self.socket = None
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((host, port))
            self.socket.settimeout(0.01)  # 10ms timeout
            print(f"ALVR socket listening on {host}:{port}")
        except Exception as e:
            raise RuntimeError(f"Failed to create ALVR socket: {e}")
    
    def update(self) -> bool:
        """Update VR poses from ALVR socket"""
        try:
            data, addr = self.socket.recvfrom(1024)
            # Parse ALVR data (format depends on ALVR version)
            # This is a placeholder - actual parsing depends on ALVR protocol
            pose_data = json.loads(data.decode('utf-8'))
            
            # Update poses from received data
            if 'headset' in pose_data:
                self.headset_pose.position = np.array(pose_data['headset']['position'])
                self.headset_pose.orientation = np.array(pose_data['headset']['orientation'])
            
            if 'left_controller' in pose_data:
                self.left_controller_pose.position = np.array(pose_data['left_controller']['position'])
                self.left_controller_pose.orientation = np.array(pose_data['left_controller']['orientation'])
            
            if 'right_controller' in pose_data:
                self.right_controller_pose.position = np.array(pose_data['right_controller']['position'])
                self.right_controller_pose.orientation = np.array(pose_data['right_controller']['orientation'])
            
            return True
        except socket.timeout:
            # No data received, keep previous poses
            return True
        except Exception as e:
            print(f"Error receiving ALVR data: {e}")
            return False
    
    def close(self):
        """Clean up socket"""
        if self.socket:
            self.socket.close()


class PepperTeleoperation:
    """Main teleoperation class for Pepper robot"""
    
    def __init__(self, urdf_path: str, vr_input: VRInputHandler, 
                 scale_factor: float = 1.0, workspace_offset: np.ndarray = None):
        """
        Initialize Pepper teleoperation
        
        Args:
            urdf_path: Path to Pepper URDF file
            vr_input: VR input handler
            scale_factor: Scale factor for VR movements (smaller = more precise)
            workspace_offset: Offset to apply to VR coordinates
        """
        self.vr_input = vr_input
        self.scale_factor = scale_factor
        self.workspace_offset = workspace_offset if workspace_offset is not None else np.array([0.0, 0.5, 0.0])
        
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Load Pepper robot
        try:
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            )
            print(f"Loaded Pepper robot from: {urdf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Pepper URDF: {e}")
        
        # Get joint information
        self._discover_joints()
        
        # Find end effectors
        self._find_end_effectors()
        
        # Create visual markers for target positions
        self._create_visual_markers()
        
        # Camera setup
        self._setup_camera()
        
        print("Pepper teleoperation initialized successfully!")
        self._print_instructions()
    
    def _discover_joints(self):
        """Discover controllable joints in the robot"""
        num_joints = p.getNumJoints(self.robot_id)
        
        self.right_arm_joints = []
        self.left_arm_joints = []
        self.head_joints = []
        
        print("\n=== Discovering Pepper Joints ===")
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Skip fixed joints
            if joint_type == p.JOINT_FIXED:
                continue
            
            # Categorize joints
            if 'RShoulder' in joint_name or 'RElbow' in joint_name or 'RWrist' in joint_name:
                self.right_arm_joints.append(joint_idx)
                print(f"  Right arm joint: {joint_name} (idx {joint_idx})")
            elif 'LShoulder' in joint_name or 'LElbow' in joint_name or 'LWrist' in joint_name:
                self.left_arm_joints.append(joint_idx)
                print(f"  Left arm joint: {joint_name} (idx {joint_idx})")
            elif 'Head' in joint_name:
                self.head_joints.append(joint_idx)
                print(f"  Head joint: {joint_name} (idx {joint_idx})")
        
        print(f"\nFound {len(self.right_arm_joints)} right arm joints")
        print(f"Found {len(self.left_arm_joints)} left arm joints")
        print(f"Found {len(self.head_joints)} head joints")
        print("=" * 35)
    
    def _find_end_effectors(self):
        """Find end effector link indices"""
        num_joints = p.getNumJoints(self.robot_id)
        
        self.right_ee_idx = None
        self.left_ee_idx = None
        
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            link_name = joint_info[12].decode('utf-8')
            
            # Look for end effector links
            if 'RWrist' in link_name or ('r_gripper' in link_name.lower()):
                if self.right_ee_idx is None:
                    self.right_ee_idx = joint_idx
                    print(f"Right end effector: {link_name} (link {joint_idx})")
            elif 'LWrist' in link_name or ('l_gripper' in link_name.lower()):
                if self.left_ee_idx is None:
                    self.left_ee_idx = joint_idx
                    print(f"Left end effector: {link_name} (link {joint_idx})")
        
        # If wrist not found, use last joint of each arm
        if self.right_ee_idx is None and self.right_arm_joints:
            self.right_ee_idx = self.right_arm_joints[-1]
            print(f"Using last right arm joint as end effector: {self.right_ee_idx}")
        
        if self.left_ee_idx is None and self.left_arm_joints:
            self.left_ee_idx = self.left_arm_joints[-1]
            print(f"Using last left arm joint as end effector: {self.left_ee_idx}")
    
    def _create_visual_markers(self):
        """Create visual markers for target positions"""
        # Right hand target marker (red)
        self.right_target_marker = self._create_sphere_marker([1, 0, 0, 0.5], 0.03)
        
        # Left hand target marker (blue)
        self.left_target_marker = self._create_sphere_marker([0, 0, 1, 0.5], 0.03)
        
        # Head target marker (green)
        self.head_target_marker = self._create_sphere_marker([0, 1, 0, 0.5], 0.05)
    
    def _create_sphere_marker(self, color: List[float], radius: float) -> int:
        """Create a sphere marker for visualization"""
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 0]
        )
        return marker_id
    
    def _setup_camera(self):
        """Setup camera view"""
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=45,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.5]
        )
    
    def _print_instructions(self):
        """Print usage instructions"""
        print("\n" + "=" * 60)
        print("PEPPER VR TELEOPERATION - READY")
        print("=" * 60)
        print("Controls:")
        print("  - Right controller -> Right arm end effector")
        print("  - Left controller  -> Left arm end effector")
        print("  - Headset orientation -> Head orientation (if available)")
        print("\nPress Ctrl+C to exit")
        print("=" * 60 + "\n")
    
    def _apply_ik_to_arm(self, target_pos: np.ndarray, target_orn: np.ndarray,
                         arm_joints: List[int], ee_idx: int) -> bool:
        """
        Apply IK solution to arm joints
        
        Args:
            target_pos: Target position [x, y, z]
            target_orn: Target orientation as quaternion [x, y, z, w]
            arm_joints: List of joint indices for this arm
            ee_idx: End effector link index
            
        Returns:
            True if IK was successful
        """
        if not arm_joints or ee_idx is None:
            return False
        
        try:
            # Calculate IK
            ik_solution = p.calculateInverseKinematics(
                self.robot_id,
                ee_idx,
                target_pos,
                target_orn,
                maxNumIterations=100,
                residualThreshold=0.001
            )
            
            # Apply IK solution to arm joints
            for idx, joint_idx in enumerate(arm_joints):
                if idx < len(ik_solution):
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=ik_solution[idx],
                        force=500,
                        maxVelocity=2.0
                    )
            
            return True
        except Exception as e:
            print(f"IK failed: {e}")
            return False
    
    def _transform_vr_to_robot(self, vr_pos: np.ndarray) -> np.ndarray:
        """
        Transform VR coordinates to robot workspace coordinates
        
        Args:
            vr_pos: Position in VR space
            
        Returns:
            Position in robot space
        """
        # Apply scaling and offset
        robot_pos = vr_pos * self.scale_factor + self.workspace_offset
        return robot_pos
    
    def update(self):
        """Main update loop - process VR input and control robot"""
        # Update VR input
        if not self.vr_input.update():
            print("Warning: Failed to update VR input")
            return False
        
        # Get VR poses
        headset = self.vr_input.headset_pose
        right_ctrl = self.vr_input.right_controller_pose
        left_ctrl = self.vr_input.left_controller_pose
        
        # Transform VR positions to robot workspace
        right_target_pos = self._transform_vr_to_robot(right_ctrl.position)
        left_target_pos = self._transform_vr_to_robot(left_ctrl.position)
        
        # Update visual markers
        p.resetBasePositionAndOrientation(
            self.right_target_marker,
            right_target_pos,
            right_ctrl.orientation
        )
        p.resetBasePositionAndOrientation(
            self.left_target_marker,
            left_target_pos,
            left_ctrl.orientation
        )
        
        # Update head marker
        head_pos = self._transform_vr_to_robot(headset.position)
        p.resetBasePositionAndOrientation(
            self.head_target_marker,
            head_pos,
            headset.orientation
        )
        
        # Apply IK to right arm
        if self.right_arm_joints and self.right_ee_idx is not None:
            self._apply_ik_to_arm(
                right_target_pos,
                right_ctrl.orientation,
                self.right_arm_joints,
                self.right_ee_idx
            )
        
        # Apply IK to left arm (if available)
        if self.left_arm_joints and self.left_ee_idx is not None:
            self._apply_ik_to_arm(
                left_target_pos,
                left_ctrl.orientation,
                self.left_arm_joints,
                self.left_ee_idx
            )
        
        # Control head orientation (if available)
        if self.head_joints:
            # Convert headset orientation to head joint angles
            euler = p.getEulerFromQuaternion(headset.orientation)
            for i, joint_idx in enumerate(self.head_joints[:2]):  # Yaw and Pitch
                if i < len(euler):
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=euler[i],
                        force=100
                    )
        
        return True
    
    def run(self):
        """Main run loop"""
        try:
            while True:
                if not self.update():
                    break
                time.sleep(0.01)  # 100Hz update rate
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.vr_input.close()
        p.disconnect()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pepper Robot VR Teleoperation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use simulated VR input (for testing):
  python pepper_vr_teleoperation.py --mode simulated
  
  # Use OpenXR (Meta Quest 3):
  python pepper_vr_teleoperation.py --mode openxr
  
  # Use ALVR:
  python pepper_vr_teleoperation.py --mode alvr --alvr-host 127.0.0.1 --alvr-port 9943
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['simulated', 'openxr', 'alvr'],
        default='simulated',
        help='VR input mode (default: simulated)'
    )
    
    parser.add_argument(
        '--urdf',
        type=str,
        default=None,
        help='Path to Pepper URDF file (default: auto-detect)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Scale factor for VR movements (default: 0.5)'
    )
    
    parser.add_argument(
        '--alvr-host',
        type=str,
        default='127.0.0.1',
        help='ALVR server host (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--alvr-port',
        type=int,
        default=9943,
        help='ALVR server port (default: 9943)'
    )
    
    args = parser.parse_args()
    
    # Find Pepper URDF if not specified
    if args.urdf is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, 'envs', 'robots', 'pepper', 'pepper.urdf')
        if not os.path.exists(urdf_path):
            print(f"Error: Could not find Pepper URDF at {urdf_path}")
            print("Please specify URDF path with --urdf")
            return 1
        args.urdf = urdf_path
    
    # Validate URDF exists
    if not os.path.exists(args.urdf):
        print(f"Error: URDF file not found: {args.urdf}")
        return 1
    
    # Create VR input handler
    print(f"Initializing VR input in {args.mode} mode...")
    try:
        if args.mode == 'simulated':
            vr_input = SimulatedVRInput()
        elif args.mode == 'openxr':
            vr_input = OpenXRInput()
        elif args.mode == 'alvr':
            vr_input = ALVRInput(args.alvr_host, args.alvr_port)
        else:
            print(f"Error: Unknown mode: {args.mode}")
            return 1
    except Exception as e:
        print(f"Error: Failed to initialize VR input: {e}")
        return 1
    
    # Create teleoperation instance
    try:
        teleop = PepperTeleoperation(
            urdf_path=args.urdf,
            vr_input=vr_input,
            scale_factor=args.scale
        )
    except Exception as e:
        print(f"Error: Failed to initialize teleoperation: {e}")
        vr_input.close()
        return 1
    
    # Run teleoperation
    teleop.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
