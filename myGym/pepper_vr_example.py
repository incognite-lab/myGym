#!/usr/bin/env python3
"""
Simple example demonstrating Pepper VR teleoperation in simulated mode

This is a minimal example showing how to use the teleoperation system.
Run this to see the system in action with simulated VR input.

Usage:
    python pepper_vr_example.py
"""

import sys
import os

# Add myGym to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run a simple example of Pepper teleoperation"""
    
    print("=" * 70)
    print("PEPPER VR TELEOPERATION - SIMPLE EXAMPLE")
    print("=" * 70)
    print("\nThis example demonstrates Pepper robot teleoperation with")
    print("simulated VR input (no VR hardware required).")
    print("\nThe simulated VR will:")
    print("  - Move the headset in a circular pattern")
    print("  - Animate the right hand controller")
    print("  - Animate the left hand controller")
    print("\nYou'll see:")
    print("  🔴 Red sphere = Right hand target")
    print("  🔵 Blue sphere = Left hand target")
    print("  🟢 Green sphere = Head position")
    print("\nPress Ctrl+C to exit")
    print("=" * 70 + "\n")
    
    # Import after path is set
    from pepper_vr_teleoperation import main as teleop_main
    
    # Set simulated mode as default
    if '--mode' not in sys.argv:
        sys.argv.extend(['--mode', 'simulated'])
    
    # Run teleoperation
    return teleop_main()

if __name__ == '__main__':
    sys.exit(main())
