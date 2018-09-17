from keras.models import load_model
import numpy as np 
import cv2
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import argparse
import math
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Dropout,BatchNormalization,Activation
from keras.models import Model
from keras import backend as K
import pickle

autoencoder=load_model('model-2018-09-17 05:17:06.590277.h5')
print(autoencoder.layers)

with open('temp_cluster.pickle','rb') as f:
	data=pickle.load(f)

data=np.array(data,dtype="float")/255.0
count=0
input_img = Input(shape=(24, 16, 1))
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
print(encoded[0])
print(encoded.shape)
with open('extracted_features.pickle','wb') as f:
	pickle.dump(encoded,f) 

