from keras.models import load_model
import numpy as np 
import cv2
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image")

args = vars(ap.parse_args())
im_path=args["image"]

autoencoder=load_model('model-2018-10-09 03:37:12.591428.h5')

image=cv2.imread(im_path,0)
image = cv2.resize(image, (32, 32))
thresh_img = img_to_array(image)


thresh_img = np.array(thresh_img, dtype="float") / 255.0
thresh_img=thresh_img[np.newaxis,:,:,:]
decoded_img=autoencoder.predict(thresh_img)
print(decoded_img.shape	)
im=np.array(decoded_img.reshape(32,32))
print(im.shape)

plt.subplot(1,2,1)
plt.imshow(image)
plt.gray()

plt.subplot(1,2,2)
plt.imshow(im)
#plt.savefig(im_path.split('/')[2])
plt.savefig(im_path.split('/')[1])
plt.gray()
plt.show()
