# -*- coding: utf-8 -*-

import tensorflow as tf

# Tipos de capa

#Capa convolucional
def conv_layer(layer_name, tensor, kernel_shape, stride):
    weights = tf.get_variable("weights", kernel_shape,
                               initializer = tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases", [kernel_shape[3]],
                             initializer=tf.constant_initializer(0.05))
    tf.summary.histogram(layer_name + "/weights", weights)
    tf.summary.histogram(layer_name + "/biases", biases)
    conv = tf.nn.conv2d(tensor, weights, stride, padding='SAME')
    return tf.nn.relu(conv + biases)

#Capa fully-connected
def fc_layer(layer_name, tensor, weights_shape):
    weights = tf.get_variable("weights", weights_shape,
                              initializer = tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", [weights_shape[1]],
                             initializer=tf.constant_initializer(0.0))
    tf.summary.histogram(layer_name + "/weights", weights)
    tf.summary.histogram(layer_name + "/biases", biases)
    mult_out = tf.matmul(tensor, weights)
    return tf.nn.relu(mult_out+biases)

#Capa de fusion de features
def fusion_layer(midlvl_features, global_features, stride):
    midlvlft_shape = midlvl_features.get_shape().as_list()
    new_shape = [config.batch_size, midlvlft_shape[1]*midlvlft_shape[2], 256]
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
    fusion_shape = [config.batch_size, 28, 28, 512]
    fusion_lvl = tf.reshape(fusion_lvl, fusion_shape)
    return conv_layer('Fusion', fusion_lvl, stride)

#Capa de salida
def output_layer(tensor, kernel_shape, stride):
    weights = tf.get_variable("weights", kernel_shape,
                               initializer = tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases", [kernel_shape[3]],
                             initializer=tf.constant_initializer(0.05))
    conv = tf.nn.conv2d(tensor, weights, stride, padding='SAME')
    output_data = tf.nn.sigmoid(tf.nn.bias_add(conv, biases))
    return output_data
    









