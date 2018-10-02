from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import model
from keras.datasets import mnist
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import cv2
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from itertools import cycle
import glob
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


INIT_LR=1e-3
EPOCHS=10
BATCH_SIZE=256
autoencoder=model.makeModel()	
#print(autoencoder.summary())

opt2=RMSprop(lr=INIT_LR, decay=1e-6)
autoencoder.compile(loss="mean_squared_error", optimizer = opt2)

with open('data1.pickle','rb') as f:
	data=pickle.load(f)

with open('labels1.pickle','rb') as f:
    labels=pickle.load(f)

#####
files = glob.glob ("Challenge-3-ForTrain/train_image/*.jpg")
files=sorted(files)

files2 = glob.glob ("Challenge-3-ForTest/test_image_random/*.jpg")
files2=sorted(files2)

data = np.array(data, dtype="float") / 255.0
#print(data.shape)

#print(data[:3])
#print(len(labels))

x_train,x_test,y_train,y_test=train_test_split(data,data,test_size=0.1)

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def imageLoader(files, batch_size=1):

    L = len(files)    
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            file=files[batch_start:limit][0]
            image = cv2.imread(file,0)
            image = img_to_array(image)
            X=image
            X=np.array(X)
            a=next_power_of_2(X.shape[0])
            b=next_power_of_2(X.shape[1])

            result = np.zeros((a,b))
            a=a-X.shape[0]
            b=b-X.shape[1]
            x_offset = a  
            y_offset = b  
            X=X.reshape((X.shape[0],X.shape[1]))
            result[x_offset:X.shape[0]+x_offset,y_offset:X.shape[1]+y_offset] = X
            X=np.array(result)
            X=X.reshape((X.shape[0],X.shape[1],1))
            X = X.reshape((1,)+X.shape)
            #print(X.shape)
            X=np.array(X, dtype="float") / 255.0
            #print(len(X))
            yield (X,X)   

            batch_start += batch_size   
            batch_end += batch_size


print(len(files))
print(len(files2))

hist = autoencoder.fit_generator(
                imageLoader(files,1),
                epochs=EPOCHS,
                steps_per_epoch=len(files),
                verbose=1,
                validation_data=imageLoader(files2,1),
                validation_steps=len(files2),
                callbacks=[TensorBoard(log_dir='/tmp/run23')]
                )


'''
autoencoder.fit(x_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                verbose=1,
                callbacks=[TensorBoard(log_dir='/tmp/run20')]
                )
'''
name='model-{}'.format(str(datetime.now()))
autoencoder.save(name+'.h5')

decoded_imgs=autoencoder.predict(x_test)
print(decoded_imgs[0])
#decoded_imgs=255*decoded_imgs

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):

    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n)
    im=decoded_imgs[i]
    #ret,thresh_img = cv2.threshold(im,140,255,cv2.THRESH_BINARY)
    plt.imshow(im)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(name+'.jpg')
plt.show()