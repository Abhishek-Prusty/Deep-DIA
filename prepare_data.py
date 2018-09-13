import cv2
import numpy as np 
import random
import glob
import pickle
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


files = glob.glob ("Challenge-3-ForTrain/train_image/*.jpg")
count=0
data=[]
for file in files:
	count+=1
	print(count,"\n")
	image = cv2.imread(file,0)
	image = cv2.resize(image, (12, 20))
	ret,thresh_img = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
	thresh_img = img_to_array(thresh_img)
	data.append(thresh_img)

#data = np.array(data, dtype="float") / 255.0
with open('data_arr.pickle','wb') as f:
	pickle.dump(data,f) 

print(data[0])
