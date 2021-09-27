import keras
from PIL import Image, ImageOps
import numpy as np


def CNNClassifier(img, weights_file):
    model = keras.models.load_model(weights_file) #Upload the model 
    ximg=img                                      #the main point here is that most of function used to pass images to CNN use thier location\
                                                  #here the case is difference, the image by itself is based, then we need to 
    size = (299, 299)       
    ximg = ImageOps.fit(ximg, size, Image.ANTIALIAS) # 1- modfy its sze to be the same as CNN first input
    ximg = ImageOps.grayscale(ximg)                  # 2- convert it to grayscale so as the all images used for training and testting 
    ximg = np.asarray(ximg)                          # 3- convert it to array as CNN only accept numpy array          
    ximg =ximg.reshape(1,299,299,1)                  # 4- reshap the image inot (number of samples=1,number of widht=299,number of height=299, number of shape=1)
    prediction = model.predict(ximg)                 # 5- Now it is the time to pass hte image to the model   
    return np.argmax(prediction)                     # 6- return the results
