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
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

files = glob.glob ("Challenge-3-ForTrain/train_image/*.jpg")
files=sorted(files)
#print(files)

img_names=[]
labels=[]
#file_label=glob.glob("Challenge-3-ForTrain/gt_train.txt")
with open("Challenge-3-ForTrain/gt_train.txt",'r') as lab:
	mapped=lab.readlines()

code=0
prev_name='0'
for line in mapped:
	name=line.split(';')
	if(name[1]!=prev_name):
		code=code+1
	labels.append(code)
	prev_name=name[1]

print(mapped[:140])
print(labels[:140])

count=0
data=[]

for file in files:
	count+=1
	print(count,"\n")
	image = cv2.imread(file,0)
	image = cv2.resize(image, (16, 24))
	image = img_to_array(image)
	data.append(image)


with open('data.pickle','wb') as f:
	pickle.dump(data,f) 

with open('labels.pickle','wb') as f:
	pickle.dump(labels,f) 
print(data[0])
print(labels)
