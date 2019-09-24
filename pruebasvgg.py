# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


def vgg_features():
    base_model = VGG16(weights='imagenet')
    
    layers = ['block1_pool','block3_pool']
    features = []
    for layer in layers:
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)
        
        img_path = 'hola.jpeg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        plt.imshow(img)
        
        layer_features = model.predict(x)
        features.append(layer_features)
    return features

def hypercolumn(features):
    feature_maps = []
    for feature in features:
        fmap = tf.image.resize_bilinear(feature, (224, 224))
        feature_maps.append(fmap)
    hypercol = tf.concat(feature_maps, 3)
    return hypercol
        

features = vgg_features()
hc = hypercolumn(features)
#layer1 = tf.image.resize_bilinear(block4_pool_features, (224, 224))
