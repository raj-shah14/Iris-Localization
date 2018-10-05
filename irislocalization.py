# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:53:05 2018

@author: Raj Shah
"""
#
import glob
import cv2
import os
import sys
import dlib
from skimage import io
from imutils import face_utils
import pandas as pd
import numpy as np

#########################Prediction############################################
# predictor = dlib.shape_predictor("predictor_eye_combine_landmarks.dat")
# detector = dlib.simple_object_detector("detector_eye.svm")
# #detector = dlib.get_frontal_face_detector()
# faces_folder = "/home/raj/Iris Project/BioID/BioID-Faces/"

# ## Now let's run the detector and shape_predictor over the images in the faces
# ## folder and display the results.
# print("Showing detections and predictions on the images in the faces folder...")
# ##win = dlib.image_window()
# for f in glob.glob(faces_folder+"*.png"):
#     print("Processing file: {}".format(f))
#     img = cv2.imread(f)

# ##
# #    #win.clear_overlay()
# #    #win.set_image(img)
# ##
# ##    # Ask the detector to find the bounding boxes of each face. The 1 in the
# ##    # second argument indicates that we should upsample the image 1 time. This
# ##    # will make everything bigger and allow us to detect more faces.
#     dets = detector(img)
#     print("Number of pair of eyes detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
# ##        # Get the landmarks/parts for the face in box d.
# #        
#         shape = predictor(img,d)
#         print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
#         shape = face_utils.shape_to_np(shape)
  
# ##        # Draw the face landmarks on the screen.
# #    #win.add_overlay(shape)
#     #cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)
#     einter=np.sqrt((shape[0][0]-shape[1][0])**2+(shape[0][1]-shape[1][1])**2)
#     cv2.circle(img,(shape[0][0],shape[0][1]),1,(255,0,0),-1)
#     cv2.circle(img,(shape[0][0],shape[0][1]),int(0.1*einter),(0,0,255))
#     cv2.circle(img,(shape[1][0],shape[1][1]),1,(255,0,0),-1)
#     cv2.circle(img,(shape[1][0],shape[1][1]),int(0.1*einter),(0,0,255))
#     print(0.1*einter)
#     cv2.imshow("temp",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#################################With Video#######################################
predictor = dlib.shape_predictor("predictor_eye_combine_landmarks.dat")
detector = dlib.simple_object_detector("detector_eye.svm")
detector2 = dlib.get_frontal_face_detector()

win_det = dlib.image_window()
win_det.set_image(detector)
## Now let's run the detector and shape_predictor over the images in the faces
## folder and display the results.
cap= cv2.VideoCapture(0)
i=0
while (True):
    ret,frame=cap.read()
    dets2=detector2(frame)
    dets = detector(frame)
    print("Number of pair of eyes detected: {}".format(len(dets)))
    print("Number of faces detected: {}".format(len(dets2)))
    if(dets):
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            shape = predictor(frame,d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
            shape = face_utils.shape_to_np(shape)

        #cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)
        einter=np.sqrt((shape[0][0]-shape[1][0])**2+(shape[0][1]-shape[1][1])**2)
        cv2.circle(frame,(shape[0][0],shape[0][1]+3),1,(255,0,0),-1)
        cv2.circle(frame,(shape[0][0],shape[0][1]+3),int(0.1*einter),(0,0,255))
        cv2.circle(frame,(shape[1][0],shape[1][1]+3),1,(255,0,0),-1)
        cv2.circle(frame,(shape[1][0],shape[1][1]+3),int(0.1*einter),(0,0,255))
        print(0.1*einter)
    cv2.imshow("frame",frame)
    cv2.imwrite("/home/raj/Iris Project/Output/detected"+str(i)+".jpg",frame)
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
   
   
