import cv2
import numpy as np 
import random
import glob
import pickle

X_data =[]
files = glob.glob ("Challenge-3-ForTrain/train_image/*.jpg")
count=0
for myFile in files:
	count+=1
	print(count,"\n")
	image = cv2.imread (myFile)
	print(image.shape)
	X_data.append(image)

X_data=np.array(X_data)
with open('data.pickle','wb') as f:
	pickle.dump(X_data,f) 
print('X_data shape:', np.array(X_data).shape)