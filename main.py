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
autoencoder=model.makeModel()	
#print(autoencoder.summary())
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

with open('data.pickle','rb') as f:
	data=pickle.load(f)

print(data)
x_train,_,x_test,_=train_test_split(data,data,test_size=0.20, random_state=42)

autoencoder.fit(x_train[0], x_train[0],
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test[0], x_test[0]),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()