# -*- coding: utf-8 -*-

import keras
from preprocess import batch_tiny_imagenet
from keras.models import Sequential
from keras.layers import Conv2D
import numpy as np
from keras.models import model_from_json
import cv2
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from time import time
import os
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
summary_dir = os.getcwd + 'summaries'

def train_(data_size, img_size):
    val_size = int(data_size/10)
    _, X_train, Y_train = batch_tiny_imagenet('train', data_size)
    _, X_val, Y_val = batch_tiny_imagenet('val', val_size)
    X_train = np.asarray(X_train).reshape(data_size,img_size,img_size,1)
    Y_train = np.asarray(Y_train).reshape(data_size,img_size,img_size,2)
    X_val = np.asarray(X_val).reshape(val_size,img_size,img_size,1)
    Y_val = np.asarray(Y_val).reshape(val_size,img_size,img_size,2)
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=(2,2), padding='same', activation='relu', input_shape=(img_size, img_size, 1)))
    model.add(Conv2D(128, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=3, strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=3, strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    
    model.add(Conv2D(128, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(keras.layers.UpSampling3D(size=(2, 2, 1), data_format=None))
    model.add(Conv2D(64, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(keras.layers.UpSampling3D(size=(2, 2, 1), data_format=None))
    model.add(Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(2, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
    model.add(keras.layers.UpSampling3D(size=(2, 2, 1), data_format=None))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    #validation_writer = tf.summary.FileWriter(summary_dir+'/validation')
    model.fit(X_train, Y_train, epochs = 50, batch_size=64, validation_data=(X_val,Y_val), callbacks=[tensorboard])
    
    #guardar modelo
    model_json = model.to_json()
    with open("model_mse.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_mse.h5")
    print("Saved model to disk")
    
def test_(test_size, img_size):
    #cargar modelo
    json_file = open('model_mse.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_mse.h5")
    print("Loaded model from disk")
    
    test_size = 1
    _, X_test, Y_test = batch_tiny_imagenet('test',test_size)
    X_test = np.asarray(X_test).reshape(test_size,img_size,img_size,1)
    Y_test = np.asarray(Y_test).reshape(test_size,img_size,img_size,2)
    img = loaded_model.predict(X_test)
    #img2 = np.concatenate((X_test[0], np.expand_dims(img[0][:,:,1],2), np.expand_dims(img[0][:,:,0],2)),2)
    img2 = np.concatenate((X_test[0], img[0]),2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_LAB2RGB)
    plt.imshow(X_test[0][:,:,0],cmap = 'gray'); plt.axis('off');plt.show()
    plt.imshow(img2); plt.axis('off'); plt.show()
    return img2
 
      
train_(100, 24)
#train_(1000, 224)
#test_(1,224)


