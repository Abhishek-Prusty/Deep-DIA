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
for name in names:
	nm=name.strip('\n')
	d[nm]=ct
	dno[nm]=int(no[ct].strip('\n'))
	ct=ct+1


print((d.keys()))
print(dno)

code=0
prev_name='0'
aaa=[]
for mp in mapped:
	name=mp.split(';')[1]
	labels.append(d[name])
print(np.unique(labels))


count=0
data=[]
for file in files:
	count+=1
	nmm=file.split('/')[2]
	nmm=nmm.split('.')[0]
	nmm=nmm.split('_')[0]
	pres=dno[nmm]
	#print(pres)
	ex=200-pres
	#print(file,"\n")
	image = cv2.imread(file,0)
	image=cv2.resize(image,(16,24))
	image = img_to_array(image)
	data.append(image)
	for i in range(ex):
		data.append(image)
	dno[nmm]=200

print(len(data))
#data=np.array(data)
with open('data1.pickle','wb') as f:
	pickle.dump(data,f) 


with open('labels1.pickle','wb') as f:
	pickle.dump(labels,f) 



#print(data.shape)
#print(labels)
