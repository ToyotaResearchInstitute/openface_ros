#!/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
import cv2
import sys

from openface_ros.srv import LearnFace, DetectFace
from std_srvs.srv import Empty

from std_msgs.msg import String
import code # for code.interact
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
import message_filters
import image_geometry
from sensor_msgs.msg import CameraInfo
import tf

_SLOP = (1000./5.)/2. # 5 is the Hz of the topic
_BASE_FRAME = "/head_l_stereo_camera_frame"
left_camera_info_topic_name = "/hsrb/head_l_stereo_camera/camera_info"
right_camera_info_topic_name = "/hsrb/head_r_stereo_camera/camera_info"


####################
# init the the ROS node, etc.
rospy.init_node('triangulate')

def msg_pub_with_pos(name, x, y, z):
    br = tf.TransformBroadcaster()
    br.sendTransform( (x, y, z),
                      tf.transformations.quaternion_from_euler(0, 0, 0),
                      rospy.Time.now(),
                      name,
                      _BASE_FRAME)

def callback(left_data, right_data, camera_info_left, camera_info_right):
    left_name = left_data.header.frame_id
    left_point_c = left_data.pose.position.x
    left_point_r = left_data.pose.position.y
    print "triangulate callback left: %s, r = %d, c = %d" % (left_name, left_point_r, left_point_c)

    right_name = right_data.header.frame_id
    if ( left_name != right_name ):
	return
    right_point_c = right_data.pose.position.x
    right_point_r = right_data.pose.position.y
    print "triangulate callback right: %s, r = %d, c = %d" % (right_name, right_point_r, right_point_c)

    disparity = left_point_c - right_point_c

    # https://github.com/rll/stereo_click/blob/master/src/stereo_converter.py and https://github.com/rll/stereo_click/blob/master/src/click_window.py
    # http://docs.ros.org/api/image_geometry/html/python/
    stereo_model = image_geometry.StereoCameraModel()
    stereo_model.fromCameraInfo(camera_info_left, camera_info_right)
    (x,y,z) = stereo_model.projectPixelTo3d( (left_point_c, left_point_r), disparity )
    print "triangulate callback: x %.2g y %.2g z %.2g with disparity %d" % (x,y,z, disparity)
    print " "
    msg_pub_with_pos( left_data.header.frame_id, x, y, z )

# subscribes to images with their callback
# message filters: http://wiki.ros.org/message_filters#Example_.28Python.29-1
image_sub_left = message_filters.Subscriber("/face_recognition_name_l", PoseStamped)
image_sub_right = message_filters.Subscriber("/face_recognition_name_r", PoseStamped)
camera_info_sub_left = message_filters.Subscriber(left_camera_info_topic_name, CameraInfo)
camera_info_sub_right = message_filters.Subscriber(right_camera_info_topic_name, CameraInfo)

print "starting triangulate.py"
ts = message_filters.ApproximateTimeSynchronizer( [image_sub_left, image_sub_right, camera_info_sub_left, camera_info_sub_right], 10, 1 )
#print "hi"
ts.registerCallback( callback )
#print "hi"
rospy.spin()
