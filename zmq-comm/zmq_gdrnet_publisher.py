#!/usr/bin/python

import rospy, zmq, json, tf
import numpy as np

from sensor_msgs.msg import Image
from object_detector_msgs.srv import detectron2_service_server, estimate_poses
from object_detector_msgs.msg import PoseWithConfidence
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf.transformations

# Set up ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5557")

rospy.init_node('gdrnet_publisher')
tf_listener = tf.listener.TransformListener()

def detect_gdrn(rgb):
    rospy.wait_for_service('detect_objects')
    try:
        detect_objects = rospy.ServiceProxy('detect_objects', detectron2_service_server)
        response = detect_objects(rgb)
        return response.detections.detections
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def estimate_gdrn(rgd, depth, detection):
    rospy.wait_for_service('estimate_poses')
    try:
        estimate_pose = rospy.ServiceProxy('estimate_poses', estimate_poses)
        response = estimate_pose(detection, rgd, depth)
        return response.poses
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def process_gdrn_output(gdrn):
    """
    Function to process gdrn output from estimate_pose.srv -> dict
    """
    # gdrn is list of PoseWithConfidence objects
    def gdrn_to_base(pose: PoseWithConfidence):
        trans, rot = tf_listener.lookupTransform("/base_link","/xtion_rgb_optical_frame", rospy.Time(0))
        pos_vector = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, 1.0]
        pos_orient = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w,]
        rot_matrix = tf.transformations.quaternion_matrix(rot)
        tf_pose = list(np.array(rot_matrix.dot(pos_vector)) + np.array([*trans,1.0]))
        tf_orient = tf.transformations.quaternion_multiply(rot, pos_orient)
        ret_pose = PoseWithConfidence()
        ret_pose.name = pose.name
        ret_pose.confidence = pose.confidence
        ret_pose.pose.position.x = tf_pose[0]
        ret_pose.pose.position.y = tf_pose[1]
        ret_pose.pose.position.z = tf_pose[2]
        ret_pose.pose.orientation.x = tf_orient[0]
        ret_pose.pose.orientation.y = tf_orient[1]
        ret_pose.pose.orientation.z = tf_orient[2]
        ret_pose.pose.orientation.w = tf_orient[3]
        return ret_pose

    ret = dict()
    for obj in gdrn:
        # PoseWithConfidence as attributes 
        # name (string), pose (Pose) and confidence (float32)
        base_pose = gdrn_to_base(obj)

        ret["name"] = base_pose.name
        ret["confidence"] = base_pose.confidence
        
        ret["position"] = [
            base_pose.pose.position.x,
            base_pose.pose.position.y,
            base_pose.pose.position.z ]
        
        ret["orientation"] = [
            base_pose.pose.orientation.x,
            base_pose.pose.orientation.y,
            base_pose.pose.orientation.z,
            base_pose.pose.orientation.w ]
    
    return ret

def publish_gdrnet(event):
    # detects objects, estimate their poses and publishes then via ZeroMQ
    detected_objects = dict() # Dictionary with number of each detected objects

    gdrnet_objects = []
    rgb = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    depth = rospy.wait_for_message('/xtion/depth/image_raw', Image)
    detections = detect_gdrn(rgb)
    if detections is not None or len(detections) > 0:
        for detect in detections:
            pose = estimate_gdrn(rgb, depth, detect)
            gdrn_object = process_gdrn_output(pose)

            if gdrn_object["name"] in detected_objects.keys():
                detected_objects[gdrn_object["name"]] += 1
                gdrn_object["name"] = "%s_%d"% (gdrn_object["name"],detected_objects[gdrn_object["name"]])
            else:
                detected_objects[gdrn_object["name"]] = 1
                gdrn_object["name"] = "%s_%d"% (gdrn_object["name"],detected_objects[gdrn_object["name"]])

            gdrnet_objects.append(gdrn_object)
    else:
        rospy.loginfo("Nothing detected by GDR-Net++")

    # print("List of detection: %s" % gdrnet_objects)
    print("Detected objects: %s" % detected_objects.keys())
    try:
        print("%s position: %s" % (gdrnet_objects[0]["name"], gdrnet_objects[0]["position"]))
    except:
        pass
    socket.send_string(json.dumps(gdrnet_objects))

rate = 2.0 # Hz
rospy.Timer(rospy.Duration(1.0 / rate), publish_gdrnet)
rospy.spin()