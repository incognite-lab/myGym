#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Emergency stop button for ROS (Tiago)
Publishes "true" on the external emergency topic when the emergency key is pressed.

Run me in "sudo" mode - requires access to /dev/input
"""
from re import A
import rospy
from std_msgs.msg import Bool
import threading
import evdev
from evdev import ecodes
from select import select
import argparse


class GlobalKeyListenerNode(object):
    def __init__(self, topic_name="/emergency_stop",
                 startup_key=ecodes.KEY_E,
                 emergency_key=ecodes.KEY_SPACE
                 ):
        rospy.init_node('global_key_listener')
        self.pub = rospy.Publisher(topic_name, Bool, queue_size=1)

        self.startup_key = startup_key
        self.emergency_key = emergency_key

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
                rospy.loginfo("Listening on device: %s (%s)" % (path, dev.name))
                self.devices.append(dev)

        if not self.devices:
            rospy.logerr("No input devices with key capability found. Exiting.")
            raise RuntimeError("No keyboard devices")

        # Create a thread to read events
        self.listener_thread = threading.Thread(target=self._evdev_loop)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        rospy.loginfo("GlobalKeyListenerNode: started")
        rospy.logwarn("Press 'Ctrl+Shift+%s' to toggle vigilance mode (waiting for emergency key)" % self._translate_ecode_to_str(self.startup_key))
        rospy.logwarn("Press '%s' for EMERGENCY STOP when in vigilance mode" % self._translate_ecode_to_str(self.emergency_key))
        rospy.loginfo("Publishing emergency stop messages on topic: %s" % topic_name)

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
            r, w, x = select(dev_fds.keys(), [], [], 0.5)
            for fd in r:
                dev = dev_fds[fd]
                for ev in dev.read():
                    # print(ev.value)
                    if ev.type == ecodes.EV_KEY:
                        self._handle_key_event(ev)

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
                rospy.loginfo("Startup key pressed -> Entering emergency mode")
                self.emergency_mode = True
        else:
            # In emergency mode: check for emergency_key or startup key again
            if self.control_down and self.shift_down and key == self.startup_key:
                rospy.loginfo("Startup key pressed -> exit emergency mode, publish False")
                self.pub.publish(Bool(False))
                self.emergency_mode = False
            elif key == self.emergency_key:
                rospy.loginfo("Emergency key pressed -> publish True")
                self.pub.publish(Bool(True))

    def shutdown(self):
        self._stop = True
        self.listener_thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("emergency_topic", type=str, nargs='?', default="/external_stop_trigger")
    args = parser.parse_args()

    emergency_topic = args.emergency_topic

    node = GlobalKeyListenerNode(topic_name=emergency_topic)
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
