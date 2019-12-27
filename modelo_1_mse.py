from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D, UpSampling2D
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from preprocesamiento import path_imags
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from datetime import datetime
import os  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train_model():
    data_size = 30000
    #data_val = data_size/10
    X = []
    Y = []
    X_val = []
    Y_val = []
    paths = path_imags('train', data_size)
    paths_val = path_imags('val', 2000)
    for path in paths:
        image = img_to_array(load_img(path))
        image = np.array(image, dtype=float)    
        Xi = rgb2lab(image/255.)[:,:,0]
        Yi = rgb2lab(image/255.)[:,:,1:]
        Yi /= 128
        Xi = Xi.reshape(1, 64, 64, 1)
        Yi = Yi.reshape(1, 64, 64, 2)
        X.append(np.squeeze(Xi, axis = 0))
        Y.append(np.squeeze(Yi, axis = 0))
    X = np.asarray(X)
    Y = np.asarray(Y)
    for path in paths_val:
        image = img_to_array(load_img(path))
        image = np.array(image, dtype=float)    
        Xi = rgb2lab(image/255.)[:,:,0]
        Yi = rgb2lab(image/255.)[:,:,1:]
        Yi /= 128
        Xi = Xi.reshape(1, 64, 64, 1)
        Yi = Yi.reshape(1, 64, 64, 2)
        X_val.append(np.squeeze(Xi, axis = 0))
        Y_val.append(np.squeeze(Yi, axis = 0))
    X_val = np.asarray(X_val)
    Y_val = np.asarray(Y_val)
    
    inputs = Input(shape=(None,None,1))
    x = Conv2D(8, (3,3), strides=2, padding='same', activation='relu')(inputs)
    x = Conv2D(16, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(16, (3,3), strides=2, padding='same', activation='relu')(x)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3,3), strides=2, padding='same', activation='relu')(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    
    x = Conv2D(16, (3,3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(2, (3,3), padding='same', activation='tanh')(x)
    y = UpSampling2D((2, 2))(x)
    
    model = Model(inputs=[inputs], outputs=[y])
    model.compile(optimizer='adam', loss='mse',)
    tensorboard = TensorBoard(log_dir="logs/modelo_1" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    model.fit(X, Y, epochs=500, batch_size=64, validation_data=(X_val,Y_val), callbacks=[tensorboard])
    #guardar modelo
    model_json = model.to_json()
    with open("model_1.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_1.h5")
    print("Modelo guardado")

def test_model():
    #cargar modelo mse
    json_file = open('model_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model_1.h5")
    print("Modelo cargado")
    
    image = img_to_array(load_img(path_imags('test',1)[0]))
    image = np.array(image, dtype=float)  
    X = rgb2lab(image/255.)[:,:,0]
    X = X.reshape(1, 64, 64, 1)
    #print(model.evaluate(X, Y, batch_size=1))
    output = model.predict(X)
    output =  output*128

    out = np.zeros((64, 64, 3))
    out[:,:,0] = X[0][:,:,0]
    out[:,:,1:] = output[0]

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(1,4,1)
    ax1.imshow(image/255.)
    ax1.axis('off')
    ax2 = fig.add_subplot(1,4,2)
    ax2.imshow(rgb2gray(image),cmap='gray')
    ax2.axis('off')
    ax3 = fig.add_subplot(1,4,3)
    ax3.imshow(lab2rgb(out))
    ax3.axis('off')
    
#train_model()
#test_model()



