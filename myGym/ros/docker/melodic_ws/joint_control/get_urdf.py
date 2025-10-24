#!/usr/bin/env python
import rospy
import os
import argparse

def main(dump_path):
    rospy.init_node("dump_urdf")
    urdf_str = rospy.get_param("/robot_description")
    with open(os.path.join(dump_path, "robot.urdf"), "w") as f:
        f.write(urdf_str)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("dump_path", type=str, default="/tmp")
    args = argparse.parse_args()
    dump_path = args.dump_path
    main(dump_path=dump_path)
