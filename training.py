import numpy as np
import dlib
from imutils import face_utils
from skimage import io
import glob
import pandas as pd
import cv2
import os



###################################Gi4e Database Parsing################################################
g1_path="/home/raj/Iris Project/gi4e_database/images/"
for i in glob.glob("/home/raj/Iris Project/gi4e_database/labels/ima*"):
   print(i)
   data = pd.read_csv(i, sep="\t", header=None)
   data = data.iloc[:,0:13]

data=data[[0,3,4,9,10]].values
for i in data:
   img=cv2.imread(os.path.join(g1_path,i[0]))
   cv2.circle(img,(int(i[1]),int(i[2])),1,(255,0,0),-1)
   cv2.circle(img,(int(i[3]),int(i[4])),1,(255,0,0),-1)
   cv2.imshow("temp",img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
##################################################################################

##############################BioID Face Database Parsing################################# 
#image=cv2.imread("/home/raj/Iris Project/BioID/BioID-FaceDatabase-V1.2/BioID_0023.pgm")
#point=open("/home/raj/Iris Project/BioID/BioID-FaceDatabase-V1.2/BioID_0023.eye","r")
#    
#print(point.readline())
#points=[]
#for i in point:
#    points=i.split()
#    
#     
#cv2.circle(image,(int(points[0]),int(points[1])),1,(255,0,0),-1)
#cv2.circle(image,(int(points[2]),int(points[3])),1,(255,0,0),-1)
#
##image=image[94:110,120:250]
#image1=image[int(points[1])-10:int(points[1])+10,int(points[0])-10:int(points[0])+10]
#image2=image[int(points[3])-10:int(points[3])+10,int(points[2])-10:int(points[2])+10]
##cv2.circle(image,(30,10),1,(255,0,0),-1)
##cv2.circle(image,(int(points[0])-int(points[2])+30,10),1,(255,0,0),-1)
#
#cv2.imshow("temp",image1)
#cv2.imshow("temp2",image2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#######################################################################################


faces_folder = "/home/raj/Iris Project/BioID/BioID-Faces/"
options = dlib.shape_predictor_training_options()

# Now make the object responsible for training the model.
# This algorithm has a bunch of parameters you can mess with.  The
# documentation for the shape_predictor_trainer explains all of them.
# You should also read Kazemi's paper which explains all the parameters
# in great detail.  However, here I'm just setting three of them
# differently than their default values.  I'm doing this because we
# have a very small dataset.  In particular, setting the oversampling
# to a high amount (300) effectively boosts the training set size, so
# that helps this example.
options.oversampling_amount = 300
# I'm also reducing the capacity of the model by explicitly increasing
# the regularization (making nu smaller) and by using trees with
# smaller depths.
options.num_trees_per_cascade_level = 200
options.cascade_depth = 10
options.nu = 0.05
options.tree_depth = 4
options.be_verbose = True

# dlib.train_shape_predictor() does the actual training.  It will save the
# final predictor to predictor.dat.  The input is an XML file that lists the
# images in the training dataset and also contains the positions of the face
# parts.

##training_xml_path = os.path.join(faces_folder, "training_with_face_landmarks.xml")
training_xml_path = os.path.join(faces_folder, "mytest.xml")
dlib.train_shape_predictor(training_xml_path, "predictor_eye_combine_landmarks.dat", options)


# Now that we have a model we can test it.  dlib.test_shape_predictor()
# measures the average distance between a face landmark output by the
# shape_predictor and where it should be according to the truth data.
print("\nTraining accuracy: {}".format(dlib.test_shape_predictor(training_xml_path, "predictor_eye_landmarks.dat")))
# The real test is to see how well it does on data it wasn't trained on.  We
# trained it on a very small dataset so the accuracy is not extremely high, but
# it's still doing quite good.  Moreover, if you train it on one of the large
# face landmarking datasets you will obtain state-of-the-art results, as shown
# in the Kazemi paper.
#testing_xml_path = os.path.join(faces_folder, "testing_with_face_landmarks.xml")
#print("Testing accuracy: {}".format(
#   dlib.test_shape_predictor(testing_xml_path, "detector.dat")))

## Now let's use it as you would in a normal application.  First we will load it
## from disk. We also need to load a face detector to provide the initial
## estimate of the facial location.

