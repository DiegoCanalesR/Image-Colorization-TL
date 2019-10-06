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
from datetime import datetime

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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
    tensorboard = TensorBoard(log_dir="logs/scalars_mse/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    model.fit(X_train, Y_train, epochs = 50, batch_size=64, validation_data=(X_val,Y_val), callbacks=[tensorboard])
    
    #guardar modelo
    model_json = model.to_json()
    with open("model_mse.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_mse.h5")
    print("Modelo guardado")
    
def test_compare(img_size):
    #cargar modelo mse
    json_file = open('model_mse.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_mse.h5")
    print("Modelo 1 cargado")
    
     #cargar modelo logcosh
    json_file2 = open('model_logcosh.json', 'r')
    loaded_model_json2 = json_file2.read()
    json_file2.close()
    loaded_model2 = model_from_json(loaded_model_json2)
    loaded_model2.load_weights("model_logcosh.h5")
    print("Modelo 2 cargado")
    
    test_size = 1
    im_orig, X_test, Y_test = batch_tiny_imagenet('test',test_size)
    im_orig = cv2.imread(im_orig[0])
    im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)
    X_test = np.asarray(X_test).reshape(test_size,img_size,img_size,1)
    Y_test = np.asarray(Y_test).reshape(test_size,img_size,img_size,2)
    img = loaded_model.predict(X_test)
    img_ = loaded_model2.predict(X_test)
    #img2 = np.concatenate((X_test[0], np.expand_dims(img[0][:,:,1],2), np.expand_dims(img[0][:,:,0],2)),2)
    img2 = np.concatenate((X_test[0], img[0]),2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_LAB2RGB)
    img2_ = np.concatenate((X_test[0], img_[0]),2)
    img2_ = cv2.cvtColor(img2_, cv2.COLOR_LAB2RGB)
    
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(1,4,1)
    ax1.imshow(im_orig)
    ax1.axis('off')
    ax2 = fig.add_subplot(1,4,2)
    ax2.imshow(X_test[0][:,:,0],cmap ='gray')
    ax2.axis('off')
    ax3 = fig.add_subplot(1,4,3)
    ax3.imshow(img2)
    ax3.axis('off')
    ax4 = fig.add_subplot(1,4,4)
    ax4.imshow(img2_)
    ax4.axis('off')
    

    return img2, img2_

#train_(1000, 224)



