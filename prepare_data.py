import cv2
import numpy as np 
import random
import glob
import pickle
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


files = glob.glob ("Challenge-3-ForTrain/train_image/*.jpg")
count=0

x = np.array([np.array(cv2.imread(fname,0))[:,:,np.newaxis] for fname in files])

with open('data.pickle','wb') as f:
	pickle.dump(x,f) 
print(x)