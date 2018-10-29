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

with open("Challenge-3-ForTrain/list_class_name.txt",'r') as lab:
	names=lab.readlines()

with open('Challenge-3-ForTrain/nb_train_each_class.txt','r') as f:
	no=f.readlines()

ct=0
nos=[]
d=dict()
dno=dict()
dno2=dict()
for name in names:
	nm=name.strip('\n')
	d[nm]=ct
	dno[nm]=int(no[ct].strip('\n'))
	dno2[nm]=int(no[ct].strip('\n'))
	ct=ct+1


print((d.keys()))
print(dno)

code=0
prev_name='0'
aaa=[]
for mp in mapped:
	name=mp.split(';')[1]
	pres=dno[name]
	ex=200-pres
	for i in range(ex):
		labels.append(d[name])
	dno[name]=200
	labels.append(d[name])

print(np.unique(labels))
print(len(labels))
count=0
data=[]
print(dno2)
for file in files:
	count+=1
	nmm=file.split('/')[2]
	nmm=nmm.split('.')[0]
	nmm=nmm.split('_')[0]
	pres=dno2[nmm]
	#print(pres)
	ex=200-pres
	#print(file,"\n")
	image = cv2.imread(file,0)
	image=cv2.resize(image,(24,16))
	image = img_to_array(image)
	data.append(image)
	for i in range(ex):
		data.append(image)
	dno2[nmm]=200

print(len(data))
print(labels)

#data=np.array(data)
with open('khmer_data.pickle','wb') as f:
	pickle.dump(data,f) 


with open('khmer_labels.pickle','wb') as f:
	pickle.dump(labels,f) 

data=np.array(data)
labels=np.array(labels)
print(data.shape)
print(labels.shape)
#print(data.shape)
#print(labels)
