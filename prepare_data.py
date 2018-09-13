import cv2
import numpy as np 
import random
import glob
import pickle
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


files = glob.glob ("Challenge-3-ForTrain/train_image/*.jpg")
files=sorted(files)
count=0
data=[]
for file in files:
	count+=1
	print(count,"\n")
	image = cv2.imread(file,0)
	image = cv2.resize(image, (12, 20))
	#cv2.imshow('yo1',image)
	#ret,thresh_img = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
	th=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
	#print(thresh_img)
	
	#cv2.imshow('yo',th)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	thresh_img = img_to_array(th)
	#print("8===================D")
	#print(thresh_img)
	data.append(thresh_img)

#data = np.array(data, dtype="float") / 255.0
with open('data_arr.pickle','wb') as f:
	pickle.dump(data,f) 

print(data[0])
