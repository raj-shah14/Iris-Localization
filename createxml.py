# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:00:24 2018

@author: Raj Shah
"""
import glob
import lxml.etree as etree
from skimage import io
import dlib
import os
import pandas as pd

################################################BioID Dataset##############################
#detector = dlib.simple_object_detector("detector_eye.svm")
#detector = dlib.get_frontal_face_detector()
#root=etree.Element("dataset")
#doc=etree.ElementTree(root)
#etree.SubElement(root,"name").text="Training fo Bio ID dataset"
#etree.SubElement(root,"comment").text="XMLFile used for Bioid eye detection points"
#images=etree.SubElement(root,"images")
#
#undetected=[]
#imagepath="C:/Users/Raj Shah/Downloads/Iris Project/BioID/BioID-FaceDatabase-V1.2/"
#for i in glob.glob(imagepath+'*.pgm'):
#    img = io.imread(i)
#    dets = detector(img)
#    temp=i.split('\\')
#    temp[1]=temp[1][:-4]+".jpg"
#    print("Number of pair of eyes detected: {}".format(len(dets)))
#    if len(dets) != 0:
#        image=etree.SubElement(images,"image",file=temp[1])
#        for k, d in enumerate(dets):
#            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
#            box=etree.SubElement(image,"box",top=str(d.top()),left=str(d.left()),width=str(d.right()-d.left()),height=str(d.bottom()-d.top()))
#            i=i[:-4]+".eye"
#    
#            point=open(i,"r")
#            point.readline()
#            for j in point:
#                points=j.split()
#                etree.SubElement(box,"part",name='00',x=str(int(points[0])),y=str(int(points[1])))
#                etree.SubElement(box,"part",name='01',x=str(int(points[2])),y=str(int(points[3])))
#
#    else:
#        undetected.append(temp[1])
#
#doc.write("mytest_bioid.xml",pretty_print=True)
##############################################################################################################

################################################Gi4e Dataset#################################################
        
detector = dlib.get_frontal_face_detector()
root=etree.Element("dataset")
doc=etree.ElementTree(root)
etree.SubElement(root,"name").text="Training Gi4e Dataset"
etree.SubElement(root,"comment").text="XMLFile used for Gi4e eye detection points"
images=etree.SubElement(root,"images")

undetected=[]        
imagepath_g1="C:/Users/Raj Shah/Downloads/Iris Project/gi4e_database/images/"


for i in glob.glob("C:/Users/Raj Shah/Downloads/Iris Project/gi4e_database/labels/ima*"):
    print(i)
    data = pd.read_csv(i, sep="\t", header=None)
    data = data.iloc[:,0:13]

data=data[[0,3,4,9,10]].values
for i in data:
    img=io.imread(os.path.join(imagepath_g1,i[0]))
    dets = detector(img)
    print("Number of pair of eyes detected: {}".format(len(dets)))
    if len(dets) != 0:
        image=etree.SubElement(images,"image",file=i[0])
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            box=etree.SubElement(image,"box",top=str(d.top()),left=str(d.left()),width=str(d.right()-d.left()),height=str(d.bottom()-d.top()))
            etree.SubElement(box,"part",name='00',x=str(int(i[1])),y=str(int(i[2])))
            etree.SubElement(box,"part",name='01',x=str(int(i[3])),y=str(int(i[4])))
    else:
        undetected.append(i[0])
        
    
doc.write("mytest_gi4e.xml",pretty_print=True)
####################################################################################################################

