import os
import cv2
from random import shuffle
import numpy as np
from numpy import array

def tiny_imagenet_txt(set_type, img_type):
    path = os.getcwd() + '/tiny-imagenet-200/'+ set_type
    #path = '/home/colorizing/~colorizing_grupo1_noborrar/tiny-imagenet-200/'+ set_type
    imgs = []
    for (root,dirs,files) in os.walk(path, topdown=True): 
        for file in files:
            if file.endswith(img_type):
                imgs.append(os.path.join(root, file))
    txt = path + '/'+set_type+'_paths.txt'
    with open(txt, 'w') as f:
        for img in imgs:
            f.write("%s\n" % img)
            
def path_imags(set_type, batch_size):
    path = os.getcwd()  + '/tiny-imagenet-200/'+set_type
    #path = '/home/colorizing/~colorizing_grupo1_noborrar/tiny-imagenet-200/'+set_type
    txt = path + '/'+set_type+'_paths.txt'
    lines = open(txt).read().splitlines()
    shuffle(lines)
    imgs = lines[:batch_size]
    return imgs

def resize(img, size):
    resized = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    resized = np.expand_dims(resized,2)
    return resized
    
def get_channels(imgs):
    a = []
    b = []
    gray = []
    ab = []
    orig = []
    for img in imgs:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig.append(array(img))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        gray.append(array(img[:,:,0]))
        a.append(array(img[:,:,1]/128))
        b.append(array(img[:,:,2]/128))
        ab.append(array(img[:,:,1:]/128))
    a = array(a)
    b = array(b)
    ab = array(ab)
    gray = array(gray)
    return orig,gray,ab,a,b


#tiny_imagenet_txt('train', '.JPEG')   
#tiny_imagenet_txt('test', '.JPEG')  
#tiny_imagenet_txt('val', '.JPEG')  





