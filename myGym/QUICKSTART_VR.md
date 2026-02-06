# Quick Start Guide - Pepper VR Teleoperation

## Quick Test (No VR Hardware)

The easiest way to test the system:

```bash
cd myGym
python pepper_vr_example.py
```

This will:
1. Load the Pepper robot in simulation
2. Use simulated VR input (no hardware needed)
3. Show animated arm movements following simulated controllers
4. Display color-coded target markers

## Installation

### Minimal (Testing Only)
```bash
pip install pybullet numpy
```

### With OpenXR (Meta Quest 3)
```bash
pip install pybullet numpy pyopenxr
```

### With ALVR (Wireless VR)
```bash
pip install pybullet numpy
# Install ALVR separately: https://github.com/alvr-org/ALVR
```

## Usage Examples

### 1. Simulated Mode (Default)
Perfect for testing without VR hardware:
```bash
python myGym/pepper_vr_teleoperation.py --mode simulated
```

### 2. OpenXR Mode (Meta Quest 3 via Link)
Connect Quest 3 via USB or Air Link, then:
```bash
python myGym/pepper_vr_teleoperation.py --mode openxr
```

### 3. ALVR Mode (Wireless)
Start ALVR server, then:
```bash
python myGym/pepper_vr_teleoperation.py --mode alvr
```

### 4. Custom URDF
Use a different robot URDF:
```bash
python myGym/pepper_vr_teleoperation.py --urdf /path/to/robot.urdf
```

### 5. Adjust Scaling
Make movements more/less sensitive:
```bash
python myGym/pepper_vr_teleoperation.py --scale 0.3  # More precise
python myGym/pepper_vr_teleoperation.py --scale 1.0  # Larger movements
```

## What You'll See

When running, you'll see:
- Pepper robot in PyBullet simulation
- Three colored spheres (target markers):
  - 🔴 Red = Right hand controller target
  - 🔵 Blue = Left hand controller target  
  - 🟢 Green = Head/headset position
- Robot arms moving to follow the controller positions

## Controls

In **Simulated Mode**:
- Positions are automatically animated
- Just watch the robot follow the simulated controllers

In **VR Mode** (OpenXR/ALVR):
- Move your controllers → Robot arms follow
- Move your head → Robot head tracks (if enabled)
- Natural 1:1 mapping with configurable scaling

## Keyboard Controls (PyBullet Window)

While focused on PyBullet window:
- Mouse drag = Rotate view
- Ctrl+Mouse = Pan view
- Mouse wheel = Zoom
- Ctrl+C in terminal = Exit

## Troubleshooting

### "Module not found" errors
```bash
pip install pybullet numpy
```

### "Could not find Pepper URDF"
Make sure you're running from the myGym directory:
```bash
cd /path/to/myGym
python myGym/pepper_vr_teleoperation.py
```

### "pyopenxr not available"
- For simulated/ALVR modes, this is just a warning (ignore it)
- For OpenXR mode, install with: `pip install pyopenxr`

### Robot not moving smoothly
Try reducing the scale factor:
```bash
python myGym/pepper_vr_teleoperation.py --scale 0.3
```

### GUI window doesn't appear
- Check if you have a display server running (required for PyBullet GUI)
- For remote servers, use X11 forwarding or VNC

## File Locations

- Main script: `myGym/pepper_vr_teleoperation.py`
- Simple example: `myGym/pepper_vr_example.py`
- Full documentation: `myGym/PEPPER_VR_TELEOPERATION.md`
- Pepper URDF: `myGym/envs/robots/pepper/pepper.urdf`
- Test suite: `myGym/test_pepper_vr.py`

## Next Steps

1. **Try it**: Start with simulated mode
2. **Read docs**: See `PEPPER_VR_TELEOPERATION.md` for details
3. **Customize**: Modify scale, workspace offset, or add features
4. **Deploy**: Connect real VR hardware when ready

## Support

For detailed documentation, see: `PEPPER_VR_TELEOPERATION.md`

For issues:
1. Check the troubleshooting section above
2. Review the full documentation
3. Open an issue on GitHub

---

**Quick Reference:**
- `--mode simulated` = No VR hardware needed
- `--mode openxr` = Meta Quest 3 via Link
- `--mode alvr` = Wireless VR via ALVR
- `--scale 0.5` = Movement sensitivity (default)
- `--urdf <path>` = Custom robot URDF
