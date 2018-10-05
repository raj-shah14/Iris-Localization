
#import glob
#from moviepy.editor import ImageSequenceClip
#images=glob.glob('/home/raj/Iris Project/Output/*.jpg')

#clip=ImageSequenceClip(images,fps=10)
#clip.write_videofile("movie.mp4")


import cv2
import numpy as np
import glob
import natsort

# import os
# i=1
# for filename in os.listdir("."):
#     if filename.startswith("temp_"):
#         os.rename(filename, str(i)+".png")
#         i=i+1

img=[]
imgfilename=[i for i in glob.glob("/home/raj/Iris Project/Output/*.jpg")]
    #imgfilename.append(i)
    

imgfilename=natsort.natsorted(imgfilename)
for i in imgfilename:
    imgfile=cv2.imread(i)
    #imgfile=cv2.cvtColor(imgfile,cv2.COLOR_BGR2RGB)
    img.append(imgfile)

height,width,layers=img[1].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter('video.mp4', fourcc, 10.0, (width, height))

for i in img:
    out.write(i)
    cv2.imshow("sg",i)

out.release()
cv2.destroyAllWindows()

