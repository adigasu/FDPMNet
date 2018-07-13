from __future__ import print_function

"Fingerprint Image denoising using M-net"

import os
import glob
import numpy as np
from PIL import Image
import datetime
import scipy as sp
import sys
import keras as keras
from keras.models import Model
from keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import SGD

# image_data_format = channels_last
K.set_image_data_format('channels_last')

# Parameter
##############################################################
modelName = 'FPD_M_net_weights'
lrate = 0.1
decay_Rate = 1e-6
height = 400
width = 275
padSz = 88
d = 8
padX = (d - height%d)
padY = (d - width%d)
ipDepth = 3
ipHeight = height + padX + padSz
ipWidth = width + padY + padSz
outDepth = 1
is_zeroMean = False
##############################################################

# Loss function
def my_loss(y_true, y_pred):
    l1_loss = K.mean(K.abs(y_pred - y_true))
    return l1_loss


# Define the neural network
def getFPDMNet(patchHeight, patchWidth, ipCh, outCh):

    # Input
    input1 = Input((patchHeight, patchWidth, ipCh))

    # Encoder
    conv1 = Conv2D(16, (3, 3), padding='same')(input1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)

    conv1 = concatenate([input1, conv1], axis=-1)
    conv1 = Conv2D(16, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    #
    input2 = MaxPooling2D(pool_size=(2, 2))(input1)
    conv21 = concatenate([input2, pool1], axis=-1)

    conv2 = Conv2D(32, (3, 3), padding='same')(conv21)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.2)(conv2)

    conv2 = concatenate([conv21, conv2], axis=-1)
    conv2 = Conv2D(32, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #
    input3 = MaxPooling2D(pool_size=(2, 2))(input2)
    conv31 = concatenate([input3, pool2], axis=-1)

    conv3 = Conv2D(64, (3, 3), padding='same')(conv31)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.2)(conv3)
    
    conv3 = concatenate([conv31, conv3], axis=-1)
    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    #
    input4 = MaxPooling2D(pool_size=(2, 2))(input3)
    conv41 = concatenate([input4, pool3], axis=-1)

    conv4 = Conv2D(128, (3, 3), padding='same')(conv41)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.2)(conv4)
    
    conv4 = concatenate([conv41, conv4], axis=-1)
    conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.2)(conv4)

    conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # Decoder
    conv5 = UpSampling2D(size=(2, 2))(conv4)
    conv51 = concatenate([conv3, conv5], axis=-1)

    conv5 = Conv2D(64, (3, 3), padding='same')(conv51)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.2)(conv5)
    
    conv5 = concatenate([conv51, conv5], axis=-1)
    conv5 = Conv2D(64, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    #
    conv6 = UpSampling2D(size=(2, 2))(conv5)
    conv61 = concatenate([conv2, conv6], axis=-1)

    conv6 = Conv2D(32, (3, 3), padding='same')(conv61)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.2)(conv6)
    
    conv6 = concatenate([conv61, conv6], axis=-1)
    conv6 = Conv2D(32, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    #
    conv7 = UpSampling2D(size=(2, 2))(conv6)
    conv71 = concatenate([conv1, conv7], axis=-1)

    conv7 = Conv2D(16, (3, 3), padding='same')(conv71)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)
    
    conv7 = concatenate([conv71, conv7], axis=-1)
    conv7 = Conv2D(16, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    # Final
    conv81 = UpSampling2D(size=(8, 8))(conv4)
    conv82 = UpSampling2D(size=(4, 4))(conv5)
    conv83 = UpSampling2D(size=(2, 2))(conv6)
    conv8 = concatenate([conv81, conv82, conv83, conv7], axis=-1)
    conv8 = Conv2D(outCh, (1, 1), activation='sigmoid')(conv8)

    ############
    model = Model(inputs=input1, outputs=conv8)

    sgd = SGD(lr=lrate, decay=decay_Rate, momentum=0.75, nesterov=True)
    model.compile(optimizer=sgd, loss=my_loss)

    return model

####################### Load & Preprocess Data #######################

# pad Data
def padData(img):

    # For grayscale image
    if len(img.shape) == 3:
        H, W, nB = img.shape
        img = np.transpose(img, (2,1,0))
        img = np.reshape(img, (nB, W, H, 1))
    
    # Pad X, Y such that it will be divisible by 8
    nB, H, W, D = img.shape
    temp = np.zeros((nB, H + padX, W + padY, D))
    temp[:, :H, :W, :] = img
    nB, H, W, D = temp.shape

    # Pad extra for network
    img = np.zeros((nB, H + padSz, W + padSz, D))
    for i in range(0, nB):
        for j in range(0, D):
            img[i,:,:,j] = np.lib.pad(temp[i,:,:,j], (np.int(padSz/2)), 'edge')

    img = (img.astype('float32'))
    return img

def load_data(dataPath):
    imgsX = imread(dataPath)

    if is_zeroMean:
        imgsX = np.array(imgsX)/127.5 - 1.
    else:
        imgsX = np.array(imgsX)/255.0

    # Pad images
    imgsX = padData(imgsX)
    return imgsX

def imread(path, is_gt=False):
    temp = sp.misc.imread(path).astype(np.float)
    if is_gt:
        temp = np.expand_dims(temp, -1)
    temp = np.expand_dims(temp, 0)
    return temp

####################################

def test_FPDMNet(dataPath):
    # Get FPD_M_Net
    loadWeightsPath = os.path.join("./weights", modelName + ".hdf5")
    fpDenNet = getFPDMNet(ipHeight, ipWidth, ipDepth, outDepth)

    # load weights
    fpDenNet.load_weights(loadWeightsPath)

    imgDataFiles = glob.glob(os.path.join(dataPath, '*.jpg'))
    imgDataFiles.sort()

    savePath = os.path.join(dataPath, "Results")
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for i in range(0, len(imgDataFiles)):

        print('-----------------------------> Test: ', i, ' <-----------------------------')

        # Read fundus data from imgDataFiles
        imTest = load_data(imgDataFiles[i])

        imPred = fpDenNet.predict(imTest, batch_size=1)
        #print('Before imPred: ', imPred.shape)

        imPred = np.squeeze(imPred)
        unpad = np.int(padSz/2)
        imPred = imPred[unpad:-unpad, unpad:-unpad]
        imPred = imPred[:-padX, :-padY]
        #print('After imPred: ', imPred.shape)

        Image.fromarray(np.uint8(imPred * 255)).save(os.path.join(savePath, os.path.basename(imgDataFiles[i])))

if __name__ == '__main__':
    dataPath = sys.argv[1]
    #dataPath = './test'

    test_FPDMNet(dataPath)



