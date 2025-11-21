#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
import numpy as np
import os

class MarkerTargetWriter:
    def __init__(self, target_file, topic_name="/gdrnet_meshes_estimated"):
        self.target_file = target_file

        # 安定化用
        self.last_pos = None           # 直前の座標
        self.stable_count = 0          # 連続で同じ座標が来た回数
        self.REQUIRED_STABLE = 5       # 何フレーム続いたら「確定」とするか
        self.POS_EPS = 0.01            # 許容誤差 [m]

        self.sub = rospy.Subscriber(topic_name, Marker,
                                    self.marker_cb, queue_size=1)

        rospy.loginfo("MarkerTargetWriter started.")
        rospy.loginfo("  subscribing topic : %s", topic_name)
        rospy.loginfo("  writing target to : %s", self.target_file)

    def marker_cb(self, marker):
        # ★オブジェクト名 marker.ns / marker.id は一切使わない
        p = marker.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=float)

        # 直前の座標との距離で「同じ物体かどうか」を判定
        if self.last_pos is not None:
            dist = np.linalg.norm(pos - self.last_pos)
        else:
            dist = float("inf")

        if dist < self.POS_EPS:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_pos = pos

        rospy.loginfo("marker pos = (%.3f, %.3f, %.3f), stable_count = %d",
                    pos[0], pos[1], pos[2], self.stable_count)

        # 一つに定まったとみなせるまで待つ
        if self.stable_count >= self.REQUIRED_STABLE:
            line = f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"

            # ディレクトリの作成
            dirname = os.path.dirname(self.target_file)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            with open(self.target_file, "w") as f:
                f.write(line)

            rospy.loginfo("=== TARGET FIXED ===")
            rospy.loginfo("wrote to %s : %s", self.target_file, line.strip())

            # もう一回指し直す用にリセットしておいても良い
            self.stable_count = 0
            self.last_pos = None

def main():
    rospy.init_node("marker_to_targets_file")

    # send_joint_gdrnet.py で指定する targets_file と同じ絶対パスにする
    target_file = rospy.get_param("~target_file",
                                "/home/student/Documents/myGym/zmq-comm/targets.txt")

    # test_obj_det_dev.py の publish 先に合わせる（ここでは /gdrnet_meshes_estimated）
    topic_name = rospy.get_param("~marker_topic", "/gdrnet_meshes_estimated")

    MarkerTargetWriter(target_file, topic_name)
    rospy.spin()

if __name__ == "__main__":
    main()
