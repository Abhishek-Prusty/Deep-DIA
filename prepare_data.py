import cv2
import numpy as np 
import random
import glob
import pickle
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


files = glob.glob ("clustering_data/*.jpg")
files=sorted(files)
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
with open('temp_cluster.pickle','wb') as f:
	pickle.dump(data,f) 

print(data[0])
