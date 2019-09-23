# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
from preprocess import train_set_tiny_imagenet

imags = train_set_tiny_imagenet(1)
R,G,B = [],[],[]
L,a,b = [],[],[]

def get_histograms(imgs):
    for img in imgs:
        img = cv2.imread(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
        R.extend(img_rgb[:,:, 0].ravel())
        G.extend(img_rgb[:,:, 1].ravel())
        B.extend(img_rgb[:,:, 2].ravel())
        L.extend(img_lab[:,:, 0].ravel())
        a.extend(img_lab[:,:, 1].ravel())
        b.extend(img_lab[:,:, 2].ravel())
    return R,G,B,L,a,b
    
hist = list(get_histograms(imags))
canales_RGB = {'R':hist[0], 'G':hist[1], 'B':hist[2]}
canales_Lab = {'L':hist[3], 'a':hist[4], 'b':hist[5]}

def plot_histograms(canales):
    for canal in canales:
        if canal in ['R','G','B']:
            plt.hist(canales[canal], 100, color = canal)  
            plt.title('Histograma canal '+canal)  
            plt.show()
        else:
            plt.hist(canales[canal], 100, color = 'm')  
            plt.title('Histograma canal '+canal)  
            plt.show()
        
print('Colorspace RGB:')
print('-'*30)
plot_histograms(canales_RGB)
print('Colorspace L*a*b:')
print('-'*30)
plot_histograms(canales_Lab)  


"Proyecto-EL4106" 
