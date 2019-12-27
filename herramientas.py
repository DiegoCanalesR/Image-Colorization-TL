from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab
from preprocesamiento import path_imags
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import inception_resnet_v2
from skimage.transform import resize
import tensorflow as tf
import keras.backend as K
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def resnet_embedding(batch):
    resnet = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet')
    resnet.graph = tf.get_default_graph()
    batch_resized = []
    for i in batch:
        i = resize(i, (299, 299, 3), mode='constant')
        batch_resized.append(i)
    batch_resized = np.array(batch_resized, dtype=np.float16)
    batch_resized = inception_resnet_v2.preprocess_input(batch_resized)
    with resnet.graph.as_default():
        embed = resnet.predict(batch_resized)
    return embed

def get_ab_histogram():
    imags = path_imags('train',10000)
    a,b = [],[]
    for img in imags:
        image = img_to_array(load_img(img))
        image = np.array(image, dtype=np.float16)    
        img_lab = rgb2lab(image/255.)
        a.extend(img_lab[:,:, 1].ravel())
        b.extend(img_lab[:,:, 2].ravel())
    histogram, a_edges, b_edges = np.histogram2d(a, b, bins=22)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log(histogram), cmap='jet', interpolation='nearest', origin='low', extent=[a_edges[0], b_edges[-1], a_edges[0], b_edges[-1]])
    plt.gca().invert_yaxis()
    plt.xlabel('b')
    plt.ylabel('a')
    return histogram

def get_bin_weights():
    histogram = get_ab_histogram()
    histogram_vec = np.concatenate(histogram)
    probs = histogram_vec/(np.sum(histogram_vec))
    weights = (0.5*probs + 0.5/(len(probs)))**-1
    weights = weights/(sum(weights))
    weights = K.constant(weights)
    return weights
    
def new_loss(y_true, y_pred):
    weights = get_bin_weights()
    y_pred = y_pred + 1e-10
    loss = -K.sum(y_true*K.log(y_pred), axis=3)/(64*64)
    indices_q = K.argmax(y_true, axis=3)
    weights_v = K.gather(reference=weights, indices=indices_q)
    loss = K.sum(loss*weights_v)
    return loss

def output(Y):
    n_images = Y.shape[0]
    height = Y.shape[1]
    width = Y.shape[2]
    bins = range(-110, 110, 10)
    bins = np.array(bins)
    output = np.zeros((n_images, height, width, 22*22), dtype = np.float16)
    a_bins = np.digitize(Y[:, :, :, 0], bins)
    b_bins = np.digitize(Y[:, :, :, 1], bins)
    ab_bins = a_bins * 22 + b_bins
    for n in range(n_images):
        for i in range(height):
            for j in range(width):
                output[n, i, j, ab_bins[n, i, j]] = 1
    return output


def ab_from_output(output):
    bins = range(-110, 110, 10)
    bins = np.array(bins)
    T = 0.38
    ab_channels = np.zeros((output.shape[0], output.shape[1], output.shape[2], 2))
    for n in range(output.shape[0]):
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                K = np.exp(np.log(output[n, i, j]+1e-6)/T)/(np.sum(np.exp(np.log(output[n, i, j]+1e-6)/T)))
                index = np.argmax(K)
                #index = np.argmax(output[n, i, j])
                a_index = index / 22
                b_index = index % 22
                ab_channels[n, i, j, 0] = bins[int(a_index)]
                ab_channels[n, i, j, 1] = bins[int(b_index)]
    return ab_channels
   
    
    