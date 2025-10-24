#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Emergency stop button for ROS (Tiago)
Publishes "true" on the external emergency topic when the emergency key is pressed.

Run me in "sudo" mode - requires access to /dev/input
"""
import sys
import rospy
from std_msgs.msg import Bool
import threading
import evdev
from evdev import ecodes
from select import select
import argparse


def _translate_key_code_to_ecode(key_str):
    key_str = key_str.upper()
    if key_str.startswith("KEY_"):
        key_str = key_str[4:]

    for key, value in vars(ecodes).items():
        if isinstance(value, int) and "KEY_" in key and key[4:].upper() == key_str:
            return value

    # Return the translated key or 'Unknown key' if not found
    return -1


class GlobalKeyListenerNode(object):
    def __init__(self, topic_name="/emergency_stop",
                 startup_key=ecodes.KEY_E,
                 emergency_keys=None
                 ):
        rospy.init_node('emergency_stop_key')
        self.pub = rospy.Publisher(topic_name, Bool, queue_size=1)

        self.startup_key = startup_key
        self.emergency_keys = emergency_keys or [ecodes.KEY_SPACE]

        self.emergency_mode = False
        self.control_down = False
        self.shift_down = False
        self._stop = False

        # Find keyboard event devices
        self.devices = []
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            # Filter for devices that report keys
            caps = dev.capabilities()
            if ecodes.EV_KEY in caps:
                print("Listening on device: %s (%s)" % (path, dev.name))
                self.devices.append(dev)

        if not self.devices:
            print("No input devices with key capability found. Exiting.")
            print('\a')
            raise RuntimeError("No keyboard devices")

        # Create a thread to read events
        self.listener_thread = threading.Thread(target=self._evdev_loop)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        keystr = ' or '.join(["'%s'" % self._translate_ecode_to_str(key) for key in self.emergency_keys])

        print("GlobalKeyListenerNode: started")
        print("Press 'Ctrl+Shift+%s' to toggle vigilance mode (waiting for emergency key)" % self._translate_ecode_to_str(self.startup_key))
        print("Press %s for EMERGENCY STOP when in vigilance mode" % keystr)
        print("Publishing emergency stop messages on topic: %s" % topic_name)

    def _translate_ecode_to_str(self, ecode):
        # Define a dictionary to store the key translations
        key_code_mapping = {}

        # Populate the dictionary with key translations
        for key, value in vars(ecodes).items():
            if isinstance(value, int) and "KEY_" in key:
                key_code_mapping[value] = key

        # Return the translated key or 'Unknown key' if not found
        return key_code_mapping.get(ecode, 'Unknown key')

    def _evdev_loop(self):
        # Use select to multiplex multiple devices
        dev_fds = {dev.fd: dev for dev in self.devices}

        while not rospy.is_shutdown() and not self._stop:
            r, _, _ = select(dev_fds.keys(), [], [], 0.1)
            for fd in r:
                dev = dev_fds[fd]
                try:
                    for ev in dev.read():
                        # print(ev)
                        if ev.type == ecodes.EV_KEY:
                            self._handle_key_event(ev)
                except OSError as e:
                    if e.errno == 19:
                        # Device removed
                        # dev_fds.pop(fd)  # don't want to remove the device
                        print("Device removed: %s (%s)" % (dev.name, dev.path))
                        print("Restart the node!!!")
                        print('\a')
                    continue

    def _handle_key_event(self, ev):
        # Key press (value = 1) or release (value = 0)
        key = ev.code
        if ev.value == 0:
            if key == ecodes.KEY_ESC:
                rospy.signal_shutdown("User requested shutdown")
            elif key == ecodes.KEY_RIGHTCTRL or key == ecodes.KEY_LEFTCTRL:
                self.control_down = False
            elif key == ecodes.KEY_RIGHTSHIFT or key == ecodes.KEY_LEFTSHIFT:
                self.shift_down = False

        if ev.value == 2:
            if key == ecodes.KEY_RIGHTCTRL or key == ecodes.KEY_LEFTCTRL:
                self.control_down = True
            elif key == ecodes.KEY_RIGHTSHIFT or key == ecodes.KEY_LEFTSHIFT:
                self.shift_down = True

        if ev.value != 1:
            return

        if key == ecodes.KEY_RIGHTCTRL or key == ecodes.KEY_LEFTCTRL:
            self.control_down = True
        elif key == ecodes.KEY_RIGHTSHIFT or key == ecodes.KEY_LEFTSHIFT:
            self.shift_down = True
        # If not in emergency mode, look for startup key
        if not self.emergency_mode:
            if self.control_down and self.shift_down and key == self.startup_key:
                print("Startup key pressed -> Entering emergency mode")
                self.emergency_mode = True
        else:
            # In emergency mode: check for emergency_key or startup key again
            if self.control_down and self.shift_down and key == self.startup_key:
                print("Startup key pressed -> exit emergency mode, publish False")
                self.release_emergency_stop()
                self.emergency_mode = False
            elif key in self.emergency_keys:
                print("Emergency key pressed -> publish True")
                self.do_emergency_stop()

    def do_emergency_stop(self):
        self.pub.publish(Bool(True))

    def release_emergency_stop(self):
        self.pub.publish(Bool(False))

    def shutdown(self):
        self._stop = True
        self.listener_thread.join()
        if self.emergency_mode:
            self.release_emergency_stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("emergency_topic", type=str, nargs='?', default="/external_stop_trigger",
                        help="Emergency stop topic where the message should be published")
    parser.add_argument("--emergency_key", "--key", "-k", action="append",
                        help="Emergency stop key (default: 'kpenter' = keypad enter key). Use this argument multiple times to add more keys.")
    args = parser.parse_args()

    emergency_topic = args.emergency_topic
    emergency_keys = []
    if args.emergency_key is None:
        args.emergency_key = ["kpenter"]
    for key in args.emergency_key:
        key_code = _translate_key_code_to_ecode(key)
        if key_code == -1:
            print("Unknown key: %s" % key)
            continue
        emergency_keys.append(key_code)

    if len(emergency_keys) == 0:
        print("No valid emergency keys found. Exiting.")
        sys.exit(1)

    node = GlobalKeyListenerNode(
        topic_name=emergency_topic,
        startup_key=ecodes.KEY_E,
        emergency_keys=emergency_keys
    )
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
