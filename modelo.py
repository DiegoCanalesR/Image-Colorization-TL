# -*- coding: utf-8 -*-

import tensorflow as tf
from preprocess import batch_tiny_imagenet
from preprocess import join_image
import time
import os

image_size = 224
batch_size = 1
epochs = 10
model_path = os.getcwd() + '\\model1'


##Capas a utilizar
#Capa convolucional
def conv_layer(layer_name, tensor, shape, stride):
    weights = tf.get_variable("weights", shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases", [shape[3]], initializer=tf.constant_initializer(0.05))
    tf.summary.histogram(layer_name + "/weights", weights)
    tf.summary.histogram(layer_name + "/biases", biases)
    conv = tf.nn.conv2d(tensor, weights, stride, padding='SAME')
    return tf.nn.relu(conv + biases)

#Capa fully-connected
def fc_layer(layer_name, tensor, shape):
    weights = tf.get_variable("weights", shape, initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", [shape[1]], initializer=tf.constant_initializer(0.0))
    tf.summary.histogram(layer_name + "/weights", weights)
    tf.summary.histogram(layer_name + "/biases", biases)
    mult_out = tf.matmul(tensor, weights)
    return tf.nn.relu(mult_out+biases)

#Capa de fusion de features
def fusion_layer(midlvl_features, global_features, shape, stride):
    midlvlft_shape = midlvl_features.get_shape().as_list()
    new_shape = [batch_size, midlvlft_shape[1]*midlvlft_shape[2], 256]
    midlvlft_reshaped = tf.reshape(midlvlft_shape, new_shape)
    fusion_lvl = []
    for j in range(midlvlft_reshaped[0]):
        for i in range(midlvlft_reshaped[1]):
             see_mid = midlvlft_reshaped[j, i, :]
             see_mid_shape = see_mid.get_shape().as_list()
             see_mid = tf.reshape(see_mid, [1, see_mid_shape[0]])
             global_features_shape = global_features[j, :].get_shape().as_list()
             see_global = tf.reshape(global_features[j, :], [1, global_features_shape[0]])
             fusion = tf.concat([see_mid, see_global], 1)
             fusion_lvl.append(fusion)
    fusion_lvl = tf.stack(fusion_lvl, 1)
    fusion_shape = [batch_size, 28, 28, 512]
    fusion_lvl = tf.reshape(fusion_lvl, fusion_shape)
    return conv_layer('Fusion', fusion_lvl, shape, stride)

#Capa de salida
def output_layer(tensor, shape, stride):
    weights = tf.get_variable("weights", shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases", [shape[3]], initializer=tf.constant_initializer(0.05))
    conv = tf.nn.conv2d(tensor, weights, stride, padding='SAME')
    output_data = tf.nn.sigmoid(tf.nn.bias_add(conv, biases))
    return output_data


#Clase para el modelo
class model():
    
    #Inicializacion
    def __init__(self):
        self.inputs = tf.placeholder(shape=[batch_size, image_size, image_size, 1], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[batch_size, image_size, image_size, 2], dtype=tf.float32)
        self.loss = None
        self.output = None
        
    def optimizer(self):
        return tf.train.AdamOptimizer(1e-4).minimize(self.loss)
    
    #Construccion del modelo
    def build_model(self):
        img = self.inputs
        #capas para low-level features
        lowlvl = conv_layer('low_lvl_conv1', img, shape=[3, 3, 1, 64], stride=[1, 2, 2, 1])
        lowlvl = conv_layer('low_lvl_conv2', lowlvl, shape=[3, 3, 64, 128], stride=[1, 1, 1, 1])
        lowlvl = conv_layer('low_lvl_conv3', lowlvl, shape=[3, 3, 128, 128], stride=[1, 2, 2, 1])
        lowlvl = conv_layer('low_lvl_conv4', lowlvl, shape=[3, 3, 128, 256], stride=[1, 1, 1, 1])
        lowlvl = conv_layer('low_lvl_conv5', lowlvl, shape=[3, 3, 256, 256], stride=[1, 2, 2, 1])
        lowlvl = conv_layer('low_lvl_conv6', lowlvl, shape=[3, 3, 256, 512], stride=[1, 1, 1, 1])
        
        #capas para mid-level features
        midlvl = conv_layer('mid_lvl_conv1', lowlvl, shape=[3, 3, 512, 512], stride=[1, 1, 1, 1])
        midlvl = conv_layer('mid_lvl_conv2', midlvl, shape=[3, 3, 512, 256], stride=[1, 1, 1, 1])

        #capas conv para global features
        globalft = conv_layer('globalft_conv1', lowlvl, shape=[3, 3, 512, 512], stride=[1, 2, 2, 1])
        globalft = conv_layer('globalft_conv2', globalft, shape=[3, 3, 512, 512], stride=[1, 1, 1, 1])
        globalft = conv_layer('globalft_conv3', globalft, shape=[3, 3, 512, 512], stride=[1, 2, 2, 1])
        globalft = conv_layer('globalft_conv4', globalft, shape=[3, 3, 512, 512], stride=[1, 1, 1, 1])
        #capas fc para global features
        global_flat = tf.reshape(globalft, [batch_size, -1])
        dim = global_flat.get_shape()[1].value
        globalft = fc_layer('global_fc1', global_flat, shape=[dim, 1024])
        globalft = fc_layer('global_fc2', globalft, shape=[1024, 512])
        globalft = fc_layer('global_fc3', globalft, shape=[512, 256])

        #capa de fusion
        ft = fusion_layer(midlvl, globalft, shape=[1, 1, 512, 256], stride=[1, 1, 1, 1])
        
        #capa de colorizacion
        ft = conv_layer('color_conv1', ft, shape=[3, 3, 256, 128], stride=[1, 1, 1, 1])
        ft = tf.image.resize_images(ft, [56, 56], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        ft = conv_layer('color_conv2', ft, shape=[3, 3, 128, 64], stride=[1, 1, 1, 1])
        ft = conv_layer('color_conv3', ft, shape=[3, 3, 64, 64], stride=[1, 1, 1, 1])
        ft = tf.image.resize_images(ft, [112, 112], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        ft = conv_layer('color_conv4', ft, shape=[3, 3, 64, 32], stride=[1, 1, 1, 1])

        #capa de salida 
        output = output_layer(ft, shape=[3, 3, 32, 2], stride=[1, 1, 1, 1])
        output = tf.image.resize_images(output, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.output = tf.image.resize_images(output, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.output))
        
   #Para entrenar
    def train_model(self, imgs):
        optimizer = self.optimizer()
        saver = tf.train.Saver()
        num_batches = int(len(imgs)/batch_size)
        with tf.Session as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                t_epoch_0 = time.time()
                loss_sum = 0
                for batch in range(num_batches):
                    _, X, Y = batch_tiny_imagenet('train', batch_size)
                    dict_train = {self.inputs: X, self.labels: Y}
                    opt , loss_val = sess.run([optimizer, self.loss], dict_train)
                    loss_sum = (loss_sum + loss_val)/batch
                    #print('batch: '+batch+', loss: '+loss_val)
                t_epoch = time.time() - t_epoch_0
                #loss = loss_sum/num_batches
                print('epoch: '+epoch+', loss: '+loss_sum+', time: '+t_epoch)
            save_model = saver.save(sess, model_path)
    
    #Para testing     
    def test_model(self, imgs):
        saver = tf.train.Saver()
        num_batches = int(len(imgs)/batch_size)
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            loss_sum = 0
            for batch in range(num_batches):
                imgs, X, Y = batch_tiny_imagenet('test', batch_size)
                dict_test = {self.inputs: X, self.labels: Y}
                Y_pred, loss_val = sess.run([self.output, self.loss], dict_test)
                join_image(X, Y_pred, batch_size, imgs)
                loss_sum = (loss_sum + loss_val)/batch
            print('loss: '+loss_sum)
                
            
        









