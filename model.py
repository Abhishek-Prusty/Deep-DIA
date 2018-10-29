from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Dropout,BatchNormalization,Activation,AveragePooling2D
from keras.models import Model
from keras import backend as K
from keras_sequential_ascii import keras2ascii
import netron
from keras.utils import plot_model

def makeModel():

	input_img = Input(shape=(None, None, 1))
	x = Conv2D(64, (3, 3), padding='same')(input_img)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(32, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(16, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)
 
	x = Conv2D(16, (3, 3), padding='same')(encoded)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(64, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(3, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
 
	autoencoder = Model(input_img, decoded)
	return autoencoder 

#auto=makeModel()
#plot_model(auto, show_shapes=True,show_layer_names=True,to_file='model.png')
#print(keras2ascii(auto))
	
