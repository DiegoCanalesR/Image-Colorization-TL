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
    
def gray(imgs):
    gray_imgs = []
    for img in imgs:
        img = cv2.imread(img)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_imgs.append(grayimg)
    return gray_imgs


def get_ab(imgs):
    a = []
    b = []
    for img in imgs:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a.append(img[:,:,1])
        b.append(img[:,:,2])
    return a,b

def train_set_tiny_imagenet(cant):
    path = os.getcwd()  + '\\tiny-imagenet-200\\train'#+ set_type
    txt = path + '\\train_paths.txt'
    lines = open(txt).read().splitlines()
    shuffle(lines)
    imgs = lines[:cant]
    return imgs
        
#tiny_imagenet_txt('train', '.JPEG')
#tiny_imagenet_txt('test', '.JPEG')
#tiny_imagenet_txt('val', '.JPEG')  
  
#imags = train_set_tiny_imagenet(10)  

#gry = gray(imags)   
#plt.imshow(gry[0], cmap = 'gray'); plt.axis('off'); plt.show()

#a,b = get_ab(imags)
#plt.imshow(a[0]); plt.axis('off'); plt.show()
#plt.imshow(b[0]); plt.axis('off'); plt.show()