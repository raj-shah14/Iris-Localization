# Iris-Localization
Iris Localization using Eye Detector and DLib shape predictor.
![](https://github.com/raj-shah14/Iris-Localization/blob/master/irislocalizattion.gif)

## Dataset 
BioID - https://www.bioid.com/About/BioID-Face-Database.
Talking Faces - http://www-prima.inrialpes.fr/FGnet/data/01-TalkingFace/talking_face.html
GI4E - http://gi4e.unavarra.es/databases/gi4e/

## Eye Detector
Using `detector.py` create HOG based eye Detector.

## Training
We need to create a shape predictor for eyes. We achieve this by using `createxml.py` and `training.py`. Once the eye Detector and shape predictor are create we can do iris localization. 

## Iris Localization
Iris Localization is carried out by using `irislocalization.py` . We use dlib frontal face detector to detect face in the image. Once a face is found we run HoG based eye detector followed by shape predictor.

## Results

![alt_text] (https://github.com/raj-shah14/Iris-Localization/blob/master/glasses1.png)
![alt_text] (https://github.com/raj-shah14/Iris-Localization/blob/master/glasses2.png)
