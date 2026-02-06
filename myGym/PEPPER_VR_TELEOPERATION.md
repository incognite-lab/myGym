# Pepper Robot VR Teleoperation

This document describes the VR teleoperation system for the Pepper robot using Meta Quest 3 headset and controllers.

## Overview

The `pepper_vr_teleoperation.py` script enables real-time control of the Pepper robot through Virtual Reality. The system maps:

- **Right Controller Position** → Right arm end-effector position
- **Left Controller Position** → Left arm end-effector position (if available)
- **Headset Orientation** → Robot head orientation
- **Controller Orientations** → End-effector orientations

The system uses PyBullet's inverse kinematics (IK) solver to compute joint angles needed to reach the target positions.

## Features

- ✅ **Multiple VR Input Modes**: OpenXR, ALVR, or simulated (for testing)
- ✅ **Real-time IK Solving**: Smooth arm movements using PyBullet IK
- ✅ **Visual Feedback**: Color-coded target markers in simulation
- ✅ **Configurable Scaling**: Adjustable movement scale factor
- ✅ **Head Tracking**: Headset orientation controls robot head (if joints are available)
- ✅ **Dual Arm Control**: Simultaneous control of both arms

## Requirements

### Software Dependencies

```bash
# Core dependencies (already in myGym)
pip install pybullet numpy

# For OpenXR support (Meta Quest 3 native):
pip install pyopenxr

# For ALVR support (wireless streaming):
# No additional Python packages needed, uses socket connection
```

### Hardware

- **Meta Quest 3** VR headset
- **PC with USB or WiFi connection** to Quest 3
- For wireless: **ALVR** (Air Light VR) server installed

## Installation

### Option 1: OpenXR (Recommended for wired connection)

1. Install OpenXR runtime for Meta Quest:
   ```bash
   # Install Oculus PC software
   # Download from: https://www.meta.com/quest/setup/
   ```

2. Install Python OpenXR bindings:
   ```bash
   pip install pyopenxr
   ```

3. Connect Quest 3 via USB cable or Air Link

### Option 2: ALVR (Recommended for wireless)

1. Install ALVR server on PC:
   ```bash
   # Download from: https://github.com/alvr-org/ALVR/releases
   # Follow installation instructions for your OS
   ```

2. Install ALVR client on Quest 3:
   - Install from SideQuest or App Lab
   - Connect to your PC

3. No additional Python packages needed (uses socket communication)

### Option 3: Simulated Mode (Testing without VR)

No additional setup required - uses simulated VR input for testing.

## Usage

### Basic Usage

Navigate to the myGym directory:
```bash
cd /path/to/myGym/myGym
```

#### Simulated Mode (Testing)
```bash
python pepper_vr_teleoperation.py --mode simulated
```

#### OpenXR Mode (Meta Quest 3)
```bash
# Make sure Quest 3 is connected via USB or Air Link
python pepper_vr_teleoperation.py --mode openxr
```

#### ALVR Mode (Wireless)
```bash
# Start ALVR server first, then run:
python pepper_vr_teleoperation.py --mode alvr
```

### Advanced Options

```bash
# Custom URDF path
python pepper_vr_teleoperation.py --urdf /path/to/custom/pepper.urdf

# Adjust movement scale (smaller = more precise)
python pepper_vr_teleoperation.py --mode simulated --scale 0.3

# Custom ALVR connection
python pepper_vr_teleoperation.py --mode alvr --alvr-host 192.168.1.100 --alvr-port 9943
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | VR input mode: `simulated`, `openxr`, or `alvr` | `simulated` |
| `--urdf` | Path to Pepper URDF file | Auto-detected |
| `--scale` | Movement scale factor (0.1-2.0) | `0.5` |
| `--alvr-host` | ALVR server IP address | `127.0.0.1` |
| `--alvr-port` | ALVR server port | `9943` |

## How It Works

### Coordinate Transformation

The system transforms VR controller positions to robot workspace:

```
robot_position = vr_position × scale_factor + workspace_offset
```

Default workspace offset places the robot's reachable space at comfortable controller positions.

### Inverse Kinematics

For each arm:
1. Get target position from controller
2. Get target orientation from controller rotation
3. Compute IK solution using PyBullet
4. Apply joint angles to robot

### Update Loop

The system runs at ~100Hz:
1. Read VR headset and controller poses
2. Update visual target markers
3. Compute IK for both arms
4. Apply joint commands
5. Update head orientation

## Visual Markers

During operation, you'll see colored spheres representing targets:

- 🔴 **Red Sphere**: Right hand target position
- 🔵 **Blue Sphere**: Left hand target position
- 🟢 **Green Sphere**: Head/headset position

## Pepper Robot Configuration

### Available Joints

Based on the Pepper URDF:

**Right Arm** (5 DOF - Controllable):
- RShoulderPitch_rjoint
- RShoulderRoll_rjoint
- RElbowYaw_rjoint
- RElbowRoll_rjoint
- RWristYaw_rjoint

**Left Arm** (Fixed in current URDF):
- LShoulder, LElbow, LWrist joints are fixed

**Head** (Fixed in current URDF):
- HeadYaw, HeadPitch joints are currently fixed

### Modifying the URDF

To enable left arm and head control, modify `envs/robots/pepper/pepper.urdf`:

Change joint types from `fixed` to `revolute`:
```xml
<!-- Before -->
<joint name="LShoulderPitch" type="fixed">

<!-- After -->
<joint name="LShoulderPitch" type="revolute">
```

## Troubleshooting

### "pyopenxr not available"
- Install with: `pip install pyopenxr`
- Or use `--mode simulated` or `--mode alvr` instead

### "Failed to initialize OpenXR"
- Ensure Meta Quest Link/Air Link is active
- Check Oculus software is running
- Try restarting the Oculus service

### "ALVR socket error"
- Ensure ALVR server is running
- Check firewall allows UDP on port 9943
- Verify IP address with `--alvr-host`

### Robot not moving smoothly
- Adjust `--scale` to smaller value for finer control
- Increase IK iterations in code if needed
- Check for joint limits in URDF

### IK fails frequently
- Controllers may be outside robot's reachable workspace
- Try moving closer to the visual markers
- Adjust workspace offset in code

## Performance Tips

1. **Lower PyBullet Graphics**: Set `p.COV_ENABLE_GUI=0` for headless mode
2. **Reduce Update Rate**: Increase `time.sleep()` in run loop
3. **Use Direct Mode**: Set `p.setRealTimeSimulation(0)` for stepped simulation

## Code Architecture

### Main Components

1. **VRInputHandler** (Base Class)
   - Abstract interface for VR input
   - Manages headset and controller poses

2. **SimulatedVRInput**
   - Generates fake VR data for testing
   - Useful for development without VR hardware

3. **OpenXRInput**
   - Interfaces with OpenXR runtime
   - Supports native Meta Quest 3 tracking

4. **ALVRInput**
   - Receives tracking data via UDP socket
   - Enables wireless VR operation

5. **PepperTeleoperation**
   - Main controller class
   - Manages robot, IK, and visualization
   - Transforms VR space to robot space

### Extending the System

To add custom VR input sources:

```python
class CustomVRInput(VRInputHandler):
    def __init__(self):
        super().__init__()
        # Initialize your VR system
    
    def update(self) -> bool:
        # Update self.headset_pose
        # Update self.left_controller_pose
        # Update self.right_controller_pose
        return True
    
    def close(self):
        # Clean up resources
        pass
```

Then use it:
```python
vr_input = CustomVRInput()
teleop = PepperTeleoperation(urdf_path, vr_input)
teleop.run()
```

## Safety Considerations

⚠️ **Important Safety Notes**:

1. This is a **simulation-only** system currently
2. Before deploying to real Pepper robot:
   - Add velocity limits
   - Implement emergency stop
   - Add collision detection
   - Validate workspace boundaries
   - Test thoroughly in simulation

3. When transitioning to real hardware:
   - Start with reduced speed
   - Keep emergency stop accessible
   - Monitor joint torques
   - Implement soft limits

## Future Enhancements

Potential improvements:

- [ ] Force feedback to controllers
- [ ] Haptic feedback on collision
- [ ] Recording and playback of motions
- [ ] Multi-user collaboration
- [ ] Integration with ROS
- [ ] Real robot interface (not just simulation)
- [ ] Gesture recognition
- [ ] Voice commands via VR
- [ ] Hand tracking (Quest 3 native)

## Examples

### Example 1: Quick Test
```bash
# Test the system without VR hardware
cd myGym
python pepper_vr_teleoperation.py --mode simulated
```

### Example 2: With Meta Quest 3
```bash
# 1. Put on Quest 3 headset
# 2. Enable Oculus Link
# 3. Run the script
python pepper_vr_teleoperation.py --mode openxr --scale 0.4
```

### Example 3: Wireless with ALVR
```bash
# Terminal 1: Start ALVR server
./ALVR

# Terminal 2: Run teleoperation
cd myGym
python pepper_vr_teleoperation.py --mode alvr --scale 0.5
```

## References

- [PyBullet Documentation](https://pybullet.org/)
- [OpenXR Specification](https://www.khronos.org/openxr/)
- [ALVR Project](https://github.com/alvr-org/ALVR)
- [Meta Quest 3](https://www.meta.com/quest/quest-3/)
- [Pepper Robot Documentation](https://www.aldebaran.com/en/pepper)

## License

This teleoperation system is part of myGym and follows the same MIT License.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Open an issue on the myGym GitHub repository
3. Contact the myGym maintainers

---

**Author**: myGym Development Team  
**Last Updated**: 2026-02-06  
**Version**: 1.0
