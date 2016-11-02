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

####################
# init the the ROS node, etc.
rospy.init_node('test_learn_detect',anonymous=True)

bridge = CvBridge()

camera_lr = rospy.get_param("~camera_lr","");
rospy.loginfo( "camera_lr " + camera_lr )

_MAX_DISTANCE_FOR_MATCH = rospy.get_param( "~max_distance_for_match", "0.5" )
rospy.loginfo( "max_distance_for_match " + str( _MAX_DISTANCE_FOR_MATCH ) )
_SHOW = rospy.get_param( "~show", "true" )
rospy.loginfo( "show " + str( _SHOW ) )
_INTERACTIVE = rospy.get_param( "~interactive", "false" )
rospy.loginfo( "interactive " + str( _INTERACTIVE ) )

try:
    external_api_request = rospy.get_param("~external_api_request")
except KeyError as e:
    rospy.logerr("Please specify param: %s", e)
    sys.exit(1)

####################
# wait for learn, detect and clear services
learn_srv_name = "learn"
detect_srv_name = "detect"
clear_srv_name = "clear"

rospy.loginfo("Waiting for services %s, %s and %s" % (learn_srv_name, detect_srv_name, clear_srv_name))

rospy.wait_for_service(learn_srv_name)
rospy.wait_for_service(detect_srv_name)
rospy.wait_for_service(clear_srv_name)

learn_srv = rospy.ServiceProxy(learn_srv_name, LearnFace)
detect_srv = rospy.ServiceProxy(detect_srv_name, DetectFace)
clear_srv = rospy.ServiceProxy(clear_srv_name, Empty)

def msg_pub(name):
    pub = rospy.Publisher('face_recognition_name', String, queue_size=10)
    pub.publish( name )

def msg_pub_with_pos(name, r, c):
    if ( not camera_lr ):
        pub = rospy.Publisher('face_recognition_name', PoseStamped, queue_size=10)
    else:
        pub = rospy.Publisher('face_recognition_name_'+camera_lr, PoseStamped, queue_size=10)
    point = PoseStamped()
    point.header.stamp = rospy.Time.now()
    point.header.frame_id = name
    point.pose.position.x = c
    point.pose.position.y = r
    pub.publish( point )

def callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    if _SHOW:
	cv2.imshow("Image window", cv_image)

    if ( _INTERACTIVE ):
        key = cv2.waitKey(10)
    else:
        cv2.waitKey(10)
        key = 1048676

    if key == 1048684 or key == 108: # L
        learns = learn_srv(image=data, name=raw_input("Name? "))
        rospy.loginfo( learns )
    elif key == 1048676 or key == 100: # D
        detections = detect_srv(image=data, external_api_request=external_api_request)
        #print detections

        if ( detections.face_detections.__len__() == 0):
            disp_str = "(no detections)"
            if ( camera_lr ):
                disp_str = camera_lr + ": " + disp_str
            rospy.loginfo( disp_str )

            if ( not camera_lr ):
                msg_pub( '(no detections)' )
            else:
                msg_pub_with_pos( '(no detections)', -1, -1 )
            return

        # unpack detections, find the winner
        names = detections.face_detections[0].names
        distances = detections.face_detections[0].l2_distances
        c = detections.face_detections[0].x + detections.face_detections[0].width/2.
        r = detections.face_detections[0].y + detections.face_detections[0].height/2.

        min_distance = 1000.
        min_distance_idx = 1000
        for idx, val in enumerate( distances ):
            if ( val < min_distance ):
                min_distance = val
                min_distance_idx = idx

        if min_distance < _MAX_DISTANCE_FOR_MATCH:
            match_name = names[ min_distance_idx ]
            disp_str = names[ min_distance_idx ] + ' with distance ' + str( min_distance )
            if ( camera_lr ):
                disp_str = camera_lr + ": " + disp_str
            rospy.loginfo( disp_str )
        else:
            match_name = '(unknown)'
            # occasionally, the first catch of no detections will not catch so make sure a distances really has something
            if ( min_distance_idx != 1000 ):
                disp_str ='(unknown)' + '; closest is ' + names[ min_distance_idx ] + ' with distance ' + str( min_distance ) 
            else:
                disp_str = '(unknown)'
            if ( camera_lr ):
                disp_str = camera_lr + ": " + disp_str
            rospy.loginfo( disp_str )

        # publish
        if ( not camera_lr ):
            msg_pub( match_name )
        else:
            msg_pub_with_pos( match_name, r, c )
    elif key == 1048675 or key == 99: # C
        print clear_srv()

    return

# subscribes to images with its callback (its callback learns/detects, calling the server)
image_sub = rospy.Subscriber("image", Image, callback)
rospy.loginfo("Listening to %s -- spinning .." % image_sub.name)
rospy.loginfo("Usage: L to learn, D to detect")

rospy.spin()
