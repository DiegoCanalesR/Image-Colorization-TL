# -*- coding: utf-8 -*-

import os
import cv2
from random import shuffle
import matplotlib.pyplot as plt

def tiny_imagenet_txt(set_type, img_type):
    path = os.getcwd()  + '\\tiny-imagenet-200\\'+ set_type
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
    for img in imgs:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray.append(img[:,:,0])
        a.append(img[:,:,1])
        b.append(img[:,:,2])
    return gray,a,b

def train_batch_tiny_imagenet(batch_size):
    path = os.getcwd()  + '\\tiny-imagenet-200\\train'#+ set_type
    txt = path + '\\train_paths.txt'
    lines = open(txt).read().splitlines()
    shuffle(lines)
    imgs = lines[:batch_size]
    return imgs
        
tiny_imagenet_txt('train', '.JPEG')
tiny_imagenet_txt('test', '.JPEG')
tiny_imagenet_txt('val', '.JPEG')  
  
imags = train_batch_tiny_imagenet(2)  

g,a,b = get_gray_and_ab(imags)
for gs in g:
    plt.imshow(gs, cmap ='gray'); plt.axis('off'); plt.show()
for aa in a:
    plt.imshow(aa); plt.axis('off'); plt.show()
for bb in b:
    plt.imshow(bb); plt.axis('off'); plt.show()