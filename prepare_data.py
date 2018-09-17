import cv2
import numpy as np 
import random
import glob
import pickle
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import os
import re

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


files = glob.glob ("clustering_data2/*.jpg")
files.sort(key=natural_keys)
print(files)
count=0
data=[]
for file in files:
	count+=1
	print(count,"\n")
	image = cv2.imread(file,0)
	image = cv2.resize(image, (16, 24))
	image = img_to_array(image)
	data.append(image)


#data = np.array(data, dtype="float") / 255.0
with open('temp_cluster2.pickle','wb') as f:
	pickle.dump(data,f) 

#print(data[0])
