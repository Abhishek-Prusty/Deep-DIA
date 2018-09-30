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


INIT_LR=1e-3
EPOCHS=100
BATCH_SIZE=64
autoencoder=model.makeModel()	
#print(autoencoder.summary())

opt2=RMSprop(lr=INIT_LR, decay=1e-6)
autoencoder.compile(loss="mean_squared_error", optimizer = opt2)

with open('data1.pickle','rb') as f:
	data=pickle.load(f)

with open('labels1.pickle','rb') as f:
    labels=pickle.load(f)

#####


data = np.array(data, dtype="float") / 255.0
#print(data.shape)

#print(data[:3])
#print(len(labels))

x_train,x_test,y_train,y_test=train_test_split(data,data,test_size=0.1)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels),
                                                 labels)


class_weights_dict = dict(zip(np.unique(labels), class_weights))
#print(class_weights_dict)
cl=[]
for i in labels:
    cl.append(class_weights_dict[i])
cl=np.array(cl)

autoencoder.fit(x_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                verbose=1,
                callbacks=[TensorBoard(log_dir='/tmp/run18')]
                )

name='model-{}'.format(str(datetime.now()))
autoencoder.save(name+'.h5')

decoded_imgs=autoencoder.predict(x_test)
print(decoded_imgs[0])
#decoded_imgs=255*decoded_imgs

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):

    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(24,16))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n)
    im=decoded_imgs[i].reshape(24,16)
    #ret,thresh_img = cv2.threshold(im,140,255,cv2.THRESH_BINARY)
    plt.imshow(im)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(name+'.jpg')
plt.show()