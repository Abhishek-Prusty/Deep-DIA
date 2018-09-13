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
from datetime import datetime


INIT_LR=1e-3
EPOCHS=200


autoencoder=model.makeModel()	
#print(autoencoder.summary())

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

with open('data_arr.pickle','rb') as f:
	data=pickle.load(f)


data = np.array(data, dtype="float") / 255.0
#print(data[0])

x_train,x_test,_,_=train_test_split(data,data,test_size=0.20, random_state=42)

autoencoder.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/run4')])

name='model-{}'.format(str(datetime.now()))
autoencoder.save(name+'.h5')

decoded_imgs=autoencoder.predict(x_test)
decoded_imgs=255*decoded_imgs

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):

    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(20,12))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n)
    im=decoded_imgs[i].reshape(20,12)
    ret,thresh_img = cv2.threshold(im,140,255,cv2.THRESH_BINARY)
    plt.imshow(thresh_img)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(name+'.jpg')
plt.show()