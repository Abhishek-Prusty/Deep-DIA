import cv2
import numpy as np 
import random
import glob
import pickle
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import os
import re
from collections import Counter
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Dropout,BatchNormalization,Activation,AveragePooling2D
from keras.models import Model
from keras import backend as K
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
from math import*
import os
from sklearn.neighbors import NearestNeighbors

def square_rooted(x):
 
	return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
 
	numerator = sum(a*b for a,b in zip(x,y))
	denominator = square_rooted(x)*square_rooted(y)
	return round(numerator/float(denominator),3)
 
# path = 'da/'
# i = 0
# for filename in os.listdir(path):
#     os.rename(os.path.join(path,filename), os.path.join(path,str(i)+'.jpg'))
#     i = i +1

autoencoder=load_model('model-2018-10-09 03:37:12.591428.h5')
files = glob.glob ("data/*.png")
#files=sorted(files)

print(len(files))
data=[]
for file in files:
	image = cv2.imread(file,0)
	image=cv2.resize(image,(24,16))
	image = img_to_array(image)
	data.append(image)
data=np.array(data,dtype="object")/255.0

input_img = Input(shape=(None, None, 1))
x = Conv2D(64, (3, 3), padding='same',weights=autoencoder.layers[1].get_weights())(input_img)
x = BatchNormalization(weights=autoencoder.layers[2].get_weights())(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same',weights=autoencoder.layers[5].get_weights())(x)
x = BatchNormalization(weights=autoencoder.layers[6].get_weights())(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same',weights=autoencoder.layers[9].get_weights())(x)
x = BatchNormalization(weights=autoencoder.layers[10].get_weights())(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

model_new=Model(input_img,encoded)

encoded=model_new.predict(data)
encoded=encoded.reshape(encoded.shape[0],-1)
print(encoded[0].shape)
print(encoded.shape)

#encoded=encoded.reshape(-1,1)
#print(encoded.shape)
neigh = NearestNeighbors(n_neighbors=10,algorithm='auto')
neigh.fit(encoded)
#print(neigh.kneighbors([encoded[1]],return_distance=False)) 

# store={}
# for i in range(len(files)):
# 	store[i]=[]
# 	for j in range(len(files)):
# 		#dist=cosine_similarity(encoded[i], encoded[j])
# 		#print(dist)
# 		dist = np.linalg.norm(encoded[i]-encoded[j])
# 		store[i].append([j,dist])
	 
# final={}
# for i in range(len(store)):
# 	d = sorted(store[i],key=lambda kv: kv[1],reverse=False)
# 	if(i==0):
# 		print(d)
# 	d=d[1:12]
# 	final[i]=d

# print(final[0])
for i in range(len(files)):
	fig=plt.figure(figsize=(15, 15))
	fig.add_subplot(1,11,1)
	img1 = cv2.imread('data/'+str(i)+'.png',0)
	plt.imshow(img1,cmap='gray')
	ind=neigh.kneighbors([encoded[i]],return_distance=False)
	ind=np.squeeze(ind)
	print(len(ind))
	for j in range(len(ind)):
		print(ind[j])
		img = cv2.imread('data/'+str(ind[j])+'.png',0)
		fig.add_subplot(1, 11, j+1)
		plt.xlabel(str(ind[j]))
		plt.imshow(img,cmap='gray')
	plt.savefig('results/'+str(i)+'.png')


# for temp in range(len(final)):
# 	fig=plt.figure(figsize=(15, 15))
# 	fig.add_subplot(1,11,1)
# 	img1 = cv2.imread('da/'+str(temp)+'.png',0)
# 	plt.imshow(img1,cmap='gray')
# 	for i in range(len(final[temp])):
# 		print(final[temp][i][0])
# 		img = cv2.imread('da/'+str(final[temp][i][0])+'.png',0)
# 		fig.add_subplot(1, 11, i+1)
# 		plt.xlabel(str(final[temp][i][1]))
# 		plt.imshow(img,cmap='gray')
# 	plt.savefig('res/'+str(temp)+'.png')

