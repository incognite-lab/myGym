#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
import numpy as np
import os
import tf
from geometry_msgs.msg import PoseStamped

#from object_detector_msgs.srv import detectron2_service_server, estimate_pointing_gesture, estimate_poses
#from robokudo_msgs.msg import GenericImgProcAnnotatorAction, GenericImgProcAnnotatorResult, GenericImgProcAnnotatorFeedback, GenericImgProcAnnotatorGoal
#import actionlib
#from sensor_msgs.msg import Image, RegionOfInterest



class MarkerTargetWriter:
    def __init__(self, target_file, topic_name="/gdrnet_meshes_estimated"):
        self.target_file = target_file

        # 安定化用
        self.last_pos = None           # 直前の座標
        self.stable_count = 0          # 連続で同じ座標が来た回数
        #self.REQUIRED_STABLE = 5       # 何フレーム続いたら「確定」とするか
        self.POS_EPS = 0.01            # 許容誤差 [m]
        
        #TF
        self.listener = tf.TransformListener()
        rospy.sleep(1.0)

        self.sub = rospy.Subscriber(topic_name, Marker,
                                    self.get_marker, queue_size=1)

        rospy.loginfo("MarkerTargetWriter started.")
        rospy.loginfo("  subscribing topic : %s", topic_name)
        rospy.loginfo("  writing target to : %s", self.target_file)
        
    # gripper_link to base_footprint
    def _transform_to_base(self, marker):
        rospy.loginfo("Detected object: %s", marker.ns)
        # Marker -> PoseStamped（frame_id is marker.header.frame_id = "gripper_link"）
        pose_gripper = PoseStamped()
        #pose_gripper.header = marker.header
        pose_gripper.header.stamp = rospy.Time(0)
        pose_gripper.header.frame_id = "xtion_rgb_optical_frame"
        
        pose_gripper.pose = marker.pose

        target_frame = "base_footprint"   # Tiago camera

        try:
            #pose_camera = self.listener.transformPose("camera_origin", pose_gripper)
            pose_base = self.listener.transformPose(target_frame, pose_gripper)
        except (tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException) as e:
            rospy.logwarn("TF transform failed: %s", str(e))
            return None

        p = pose_base.pose.position
        #p = pose_gripper.pose.position
        
        rospy.loginfo("target in %s: x=%.3f y=%.3f z=%.3f",
                    target_frame, p.x, p.y, p.z)
        return p


    def get_marker(self, marker):
        p = self._transform_to_base(marker)
        if p is None:
            return  

        x, y, z = p.x, p.y, p.z
        y_c = x
        x_c = -y
        #z_c = p.z
        #z_m = -z_c
        line = f"{x_c -0.1:.6f} {y_c - 0.2:.6f} {z - 1.0:.6f}\n"#difined manualy

        dirname = os.path.dirname(self.target_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(self.target_file, "w") as f:
            f.write(line)

        rospy.loginfo("wrote_file: %s", line.strip())


    # def marker_cb(self, marker):

    #     p = marker.pose.position
    #     pos = np.array([p.x, p.y, p.z], dtype=float)
    #     if self.last_pos is not None:
    #         dist = np.linalg.norm(pos - self.last_pos)
    #     else:
    #         dist = float("inf")

    #     if dist < self.POS_EPS:
    #         self.stable_count += 1
    #     else:
    #         self.stable_count = 1
    #         self.last_pos = pos

    #     rospy.loginfo("marker pos = (%.3f, %.3f, %.3f), stable_count = %d",
    #                 pos[0], pos[1], pos[2], self.stable_count)

    #     if self.stable_count >= self.REQUIRED_STABLE:
    #         line = f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"

    #         dirname = os.path.dirname(self.target_file)
    #         if dirname:
    #             os.makedirs(dirname, exist_ok=True)

    #         with open(self.target_file, "w") as f:
    #             f.write(line)

    #         rospy.loginfo("=== TARGET FIXED ===")
    #         rospy.loginfo("wrote to %s : %s", self.target_file, line.strip())

    #         self.stable_count = 0
    #         self.last_pos = None

def main():
    rospy.init_node("marker_to_targets_file")

    # same with target.py
    target_file = rospy.get_param("~target_file",
                                "/home/student/Documents/myGym/zmq-comm/targets.txt")

    # where test_obj_det_dev.py published
    topic_name = rospy.get_param("~marker_topic", "/gdrnet_meshes")

    MarkerTargetWriter(target_file, topic_name)
    #print("wait")
    rospy.spin()

if __name__ == "__main__":
    main()
