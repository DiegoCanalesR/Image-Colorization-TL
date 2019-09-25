# -*- coding: utf-8 -*-

import os
import cv2
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np

def tiny_imagenet_txt(set_type, img_type):
    path = os.getcwd() + '\\tiny-imagenet-200\\'+ set_type
    imgs = []
    for (root,dirs,files) in os.walk(path, topdown=True): 
        for file in files:
            if file.endswith(img_type):
                imgs.append(os.path.join(root, file))
    txt = path + '\\'+set_type+'_paths.txt'
    with open(txt, 'w') as f:
        for img in imgs:
            f.write("%s\n" % img)

def get_gray_and_ab(imgs):
    a = []
    b = []
    gray = []
    ab = []
    for img in imgs:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray.append(img[:,:,0])
        a.append(img[:,:,1])
        b.append(img[:,:,2])
        ab.append(img[:,:,1:])
    return gray,a,b,ab

def batch_tiny_imagenet(set_type, batch_size):
    path = os.getcwd()  + '\\tiny-imagenet-200\\'+set_type
    txt = path + '\\'+set_type+'_paths.txt'
    lines = open(txt).read().splitlines()
    shuffle(lines)
    imgs = lines[:batch_size]
    gray, a, b, ab = get_gray_and_ab(imgs)
    return imgs, gray, ab

def join_image(X, Y_pred, batch_size, imgs):
    for batch in range(batch_size):
        image = np.concatenate((X,Y),2)
        image = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        name = imgs[batch][-14:]
        save_path = os.getcwd() + '\\tiny-imagenet-200\\img_model1'+name
  
#tiny_imagenet_txt('train', '.JPEG')
#tiny_imagenet_txt('test', '.JPEG')
#tiny_imagenet_txt('val', '.JPEG')  
  
imags, gray, ab = batch_tiny_imagenet('test', 2)  
for g in gray:
    plt.imshow(g, cmap ='gray'); plt.axis('off'); plt.show()
