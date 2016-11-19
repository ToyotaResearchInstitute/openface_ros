#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError

from openface_ros.srv import LearnFace, DetectFace
from openface_ros.msg import FaceDetection
from std_srvs.srv import Empty

import numpy as np
import cv2
import os
from datetime import datetime

import dlib
import openface

from face_client import FaceClient

from timeout import Timeout
import pickle
import rospkg
import sys

# For attributes
face_client = FaceClient('69efefc20c7f42d8af1f2646ce6742ec', '5fab420ca6cf4ff28e7780efcffadb6c')
def _external_request_with_timeout(buffers):
    timeout_duration = 60
    rospy.loginfo("Trying external API request for %d seconds", timeout_duration)
    timeout_function = Timeout(face_client.faces_recognize, timeout_duration)
    return timeout_function(buffers)


def _set_label(img, label, origin):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1

    text = cv2.getTextSize(label, font, scale, thickness)
    p2 = (origin[0] + text[0][0], origin[1] -text[0][1])
    cv2.rectangle(img, origin, p2, (0, 0, 0), -1)
    cv2.putText(img, label, origin, font, scale, (255, 255, 255), thickness, 8)


def _get_roi(bgr_image, detection):
    factor_x = 0.1
    factor_y = 0.2

    # Get the roi
    min_y = detection.top()
    max_y = detection.bottom()
    min_x = detection.left()
    max_x = detection.right()

    dx = max_x - min_x
    dy = max_y - min_y

    padding_x = int(factor_x * dx)
    padding_y = int(factor_y * dy)

    # Don't go out of bound
    min_y = max(0, min_y - padding_y)
    max_y = min(max_y + padding_y, bgr_image.shape[0]-1)
    min_x = max(0, min_x - padding_x)
    max_x = min(max_x + padding_x, bgr_image.shape[1]-1)

    return bgr_image[min_y:max_y, min_x:max_x]


def _get_min_l2_distance(vector_list_a, vector_b):
    return min([np.dot(vector_a - vector_b, vector_a - vector_b) for vector_a in vector_list_a])


class OpenfaceROS:
    def __init__(self, align_path, net_path, storage_folder, dictionary_file, save):
        self._bridge = CvBridge()
        self._learn_srv = rospy.Service('learn', LearnFace, self._learn_face_srv)
        self._detect_srv = rospy.Service('detect', DetectFace, self._detect_face_srv)
        self._clear_srv = rospy.Service('clear', Empty, self._clear_faces_srv)


        # Init align and net
        self._align = openface.AlignDlib(align_path)
        self._net = openface.TorchNeuralNet(net_path, imgDim=96, cuda=False)
        self._face_detector = dlib.get_frontal_face_detector()
        self._face_dict = {}  # Mapping from string to list of reps
        self._dictionary_file = dictionary_file
        self._save = save

        if ( self._dictionary_file != "" ):
          with open( self._dictionary_file, 'rb') as f:
            self._face_dict = pickle.load( f )
            rospy.loginfo( "read _face_dict: " + self._dictionary_file )

        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)

        self._storage_folder = storage_folder

    def _update_detections_with_recognitions(self, detections):
        detections = [self._update_detection_with_recognition(d) for d in detections]

        # Now find the detection index with highest name probability, only for image in /tmp/faces
        for name in self._face_dict.keys():
            try:
                l2_distances = [ dict(zip(d["names"], d["l2_distances"]))[name] for d in detections ]
                min_index = l2_distances.index(min(l2_distances))
                detections[min_index]["name"] = name # @bug
                # @todo fix bug in labeling; for now all detections get last name in dictionary
            except: # If recognizer does not find face in detection
                pass

        return detections

    def _update_detection_with_recognition(self, detection):
        detection["names"] = []
        detection["l2_distances"] = []

        try:
            recognition_rep = self._get_rep(detection["roi"])
            detection["names"] = self._face_dict.keys()
            detection["l2_distances"] = [_get_min_l2_distance(reps, recognition_rep) for reps in self._face_dict.values()]
        except Exception as e:
            warn_msg = "Could not get representation of face image but detector found one: %s" % e
            rospy.logwarn(warn_msg)

        return detection

    def _update_detections_with_attributes(self, detections):
        buffers = [cv2.imencode('.jpg', d["roi"])[1].tostring() for d in detections]

        try:
            response = _external_request_with_timeout(buffers)
            for i, photo in enumerate(response["photos"]):
                detections[i]["attrs"] = photo["tags"][0]["attributes"]
        except Exception as e:
            rospy.logerr("External API call failed: %s", e)

        return detections

    def _get_rep(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        bb = self._align.getLargestFaceBoundingBox(rgb_image)
        if bb is None:
            raise Exception("Unable to find a face in image")

        aligned_face = self._align.align(96, rgb_image, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align face bb image")

        return self._net.forward(aligned_face)

    def _learn_face_srv(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        now = datetime.now()
        cv2.imwrite("%s/%s_learn_%s.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%S-%f"), req.name), bgr_image)

        try:
            rep = self._get_rep(bgr_image)
        except Exception as e:
            error_msg = "Could not get representation of face image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        if req.name not in self._face_dict:
            self._face_dict[req.name] = []

        self._face_dict[req.name].append(rep)

        rospy.loginfo("Succesfully learned face of '%s'" % req.name)

        # from http://www.diveintopython3.net/serializing.html
        if ( self._dictionary_file != "" ):
          with open( self._dictionary_file, 'wb' ) as f:
            pickle.dump( self._face_dict, f );
            rospy.loginfo( "wrote _face_dict: " + self._dictionary_file )

        return {"error_msg": ""}

    def _save_images(self, detections, bgr_image):
        now = datetime.now()
        cv2.imwrite("%s/%s_detect.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%d-%f")), bgr_image)

        for d in detections:
            now = datetime.now()
            cv2.imwrite("%s/roi_%s_detection.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%d-%f")), d["roi"])

            cv2.rectangle(bgr_image, (d["x"], d["y"]), (d["x"] + d["width"], d["y"] + d["height"]), (0, 0, 255), 3)

            txt = ""
            try:
                if "name" in d:
                    txt += d["name"]
                txt += " (" + d["attrs"]["gender"]["value"] + ", " + d["attrs"]["age_est"]["value"] + ")"
            except KeyError:
                pass

            _set_label(bgr_image, txt, (d["x"], d["y"]))

        cv2.imwrite("%s/%s_annotated.jpeg" % (self._storage_folder, now.strftime("%Y-%m-%d-%H-%M-%d-%f")), bgr_image)

        rospy.loginfo("Wrote images to '%s'", self._storage_folder)

    def _clear_faces_srv(self, req):
        rospy.loginfo("Cleared all faces")
        self._face_dict = {}
        return {}

    def _detect_face_srv(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            return {"error_msg": error_msg}

        # Display the resulting frame
        detections = [{"roi": _get_roi(bgr_image, d), "x": d.left(), "y": d.top(),
                       "width": d.width(), "height": d.height()} for d
                      in self._face_detector(bgr_image, 1)]  # 1 = upsample factor

        # Try to recognize
        detections = self._update_detections_with_recognitions(detections)

        # @warn: verbose!
        rospy.logdebug("This is for debugging")
        rospy.logdebug(detections)

        # Try to add attributes
        if req.external_api_request:
            detections = self._update_detections_with_attributes(detections)

        # Save images
        if self._save:
            self._save_images(detections, bgr_image)

        return {
            "face_detections": [FaceDetection(names=(d["names"] if "name" in d else []),
                                              l2_distances=(d["l2_distances"] if "name" in d else []),
                                              x=d["x"], y=d["y"], width=d["width"], height=d["height"],
                                              gender_is_male=(d["attrs"]["gender"]["value"] == "male" if "attrs" in d else 0),
                                              gender_score=(float(d["attrs"]["gender"]["confidence"]) if "attrs" in d else 0),
                                              age_score=(1.0 if "attrs" in d else 0),
                                              age=(int(d["attrs"]["age_est"]["value"]) if "attrs" in d else 0))
                                for d in detections],
            "error_msg": ""
        }

if __name__ == '__main__':
    ####################
    # main creates an instance of OpenfaceROS
    rospy.init_node('openface')

    openface_path_param = rospy.get_param( "~openface_path", "~/openface/" )
    if ( openface_path_param.find( '~' ) != -1 ):
      openface_path_param = os.path.expanduser( openface_path_param )
    rospy.loginfo( "openface_path_param " + openface_path_param )

    align_path_param = rospy.get_param('~align_path', openface_path_param + 'models/dlib/shape_predictor_68_face_landmarks.dat')
    rospy.loginfo( "align_path_param " + align_path_param )

    net_path_param = rospy.get_param('~net_path', openface_path_param + 'models/openface/nn4.small2.v1.t7')
    rospy.loginfo( "net_path_param " + net_path_param )

    if (not os.path.isfile( align_path_param ) or\
        not os.path.isfile( net_path_param ) ):
         rospy.error( "openface missing models (be sure to do get_models.sh in openface/models); exiting" )
         sys.exit()

    storage_folder_param = rospy.get_param('~storage_folder', os.path.expanduser('/tmp/faces'))

    dictionary_file_param = rospy.get_param( '~dictionary_file', '' )
    if ( not os.path.isfile( dictionary_file_param ) ):
      rospy.logwarn( "dictionary file missing; it will not be loaded nor updated" )
      dictionary_file_param = ""
    else:
      rospy.loginfo( "dictionary_file_param " + dictionary_file_param )

    save_param = rospy.get_param( "~save", "false" )
    rospy.loginfo( "save_param " + str( save_param ) )
    
    openface_node = OpenfaceROS(align_path_param, net_path_param, storage_folder_param, dictionary_file_param, save_param)
    rospy.spin()
