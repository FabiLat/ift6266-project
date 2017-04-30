# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:37:20 2017

@author: calatfab
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import PIL.Image as Image
import theano


def load_mscoco(datapath):
    train_data_path = os.path.join(datapath, "train2014/")
    valid_data_path = os.path.join(datapath, "val2014/")
    test_data_path = os.path.join(datapath, "test2014/")
    
#   caption_path = os.path.join(datapath, "dict_key_imgID_value_caps_train_and_valid.pkl")
#   print 'caption_path',caption_path
#   with open(caption_path,'rb') as fd:
#        caption_dict = pkl.load(fd)

    #NB: caption_dict contains the captions for all training sets and validation set.
    #print 'len(caption_dict)',len(caption_dict)
    #print data_path + "/*.jpg"
    train_images = glob.glob(train_data_path + "/*.jpg")
    valid_images = glob.glob(valid_data_path + "/*.jpg")
    test_images = glob.glob(test_data_path + "/*.jpg")
    
    train_inputs=np.array([])
    train_targets=np.array([])
    valid_inputs=np.array([])
    valid_targets=np.array([])
    test_inputs=np.array([])
    test_targets=np.array([])
    
    #read training images
    for i, img_path in enumerate(train_images):
       # if (i<100):
            print 'Loading train image...',i
            img = Image.open(img_path)
            img_array = np.array(img)
    
            #caption id
            #cap_id = os.path.basename(img_path)[:-4]
    
            ### Get input/target from the images
            center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
            #keep RGB images only
            if len(img_array.shape) == 3:
                #need one copy  for the input and one copy for the target
                input = np.copy(img_array)
                input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                input=input.reshape(1,64*64,3)
                target=target.reshape(1,32*32,3)
                #convert images into numpy arrays    
                if (i==0):
                    train_inputs=input
                    train_targets=target
                else:
                    train_inputs=np.concatenate((train_inputs,input), axis=0)
                    train_targets=np.concatenate((train_targets,target), axis=0)
            
            
     #read validation images
    for i, img_path in enumerate(valid_images):
       #if (i<100):
            print 'Loading validation image...',i
            
            img = Image.open(img_path)
            img_array = np.array(img)
    
            #caption id
            #cap_id = os.path.basename(img_path)[:-4]
    
            ### Get input/target from the images
            center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
            #keep RGB images only
            if len(img_array.shape) == 3:
                #need one copy  for the input and one copy for the target
                input = np.copy(img_array)
                input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                input=input.reshape(1,64*64,3)
                target=target.reshape(1,32*32,3)
                #convert images into numpy arrays    
                if (i==0):
                    valid_inputs=input
                    valid_targets=target
                else:
                    valid_inputs=np.concatenate((valid_inputs,input), axis=0)
                    valid_targets=np.concatenate((valid_targets,target), axis=0)
           
    #read test images
    for i, img_path in enumerate(test_images):
       #if (i<100):
            print 'Loading test image...',i
            
            img = Image.open(img_path)
            img_array = np.array(img)
    
            #caption id
            #cap_id = os.path.basename(img_path)[:-4]
    
            ### Get input/target from the images
            center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
            #keep RGB images only
            if len(img_array.shape) == 3:
                #need one copy  for the input and one copy for the target
                input = np.copy(img_array)
                input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                input=input.reshape(1,64*64,3)
                target=target.reshape(1,32*32,3)
                #convert images into numpy arrays    
                if (i==0):
                    test_inputs=input
                    test_targets=target
                else:
                    test_inputs=np.concatenate((test_inputs,input), axis=0)
                    test_targets=np.concatenate((test_targets,target), axis=0)
    return [(train_inputs,train_targets),(valid_inputs,valid_targets),(test_inputs,test_targets)]
        

if __name__ == '__main__':
    
    theano.config.openmp = True
    print('Loading mscoco...')
    datasets = load_mscoco("inpainting/")
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print 'train_set_x.shape',train_set_x.shape
    print 'valid_set_x.shape',valid_set_x.shape
    print 'test_set_x.shape',test_set_x.shape
    
    #normalize values between 0 and 1
    train_set_x = train_set_x.astype('float32') / 255.
    train_set_y = train_set_y.astype('float32') / 255.
    
    valid_set_x = valid_set_x.astype('float32') / 255.
    valid_set_y = valid_set_y.astype('float32') / 255.
    
    test_set_x = test_set_x.astype('float32') / 255.
    test_set_y = test_set_y.astype('float32') / 255.
    
    
    train_set_x = np.reshape(train_set_x, (len(train_set_x), 64, 64, 3)) 
    train_set_y = np.reshape(train_set_y, (len(train_set_y), 32, 32, 3)) 
    valid_set_x = np.reshape(valid_set_x, (len(valid_set_x), 64, 64, 3)) 
    valid_set_y = np.reshape(valid_set_y, (len(valid_set_y), 32, 32, 3)) 
    test_set_x = np.reshape(test_set_x, (len(test_set_x), 64, 64, 3)) 
    test_set_y = np.reshape(test_set_y, (len(test_set_y), 32, 32, 3)) 
    
	#Input image format
    input_img = Input(shape=(64, 64, 3))  
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    print('encoded...')
    
    # at this point the representation is (16, 16, 32) 
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    print('decoded...')
	
	# Output format is (32, 32, 3) 
   
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    
    autoencoder.fit(train_set_x, train_set_y,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(valid_set_x, valid_set_y))
    print('autoencoder.fit...')
                    
    decoded_imgs = autoencoder.predict(valid_set_x)
    
    n = 100
    
    save_dir='Test/'
    for i in range(n):
        array=decoded_imgs[i].reshape(32, 32, 3)*255.
        output = Image.fromarray(array.astype('uint8'))
        num= "%05d" % i
        output.save(save_dir + "validation_center_" + num + '.jpg')
        valid_array=valid_set_x[i].reshape(64, 64, 3)*255.
        valid_full = Image.fromarray(valid_array.astype('uint8'))
        tofill = np.array(valid_full)
        center = (int(np.floor(tofill.shape[0] / 2.)), int(np.floor(tofill.shape[1] / 2.)))
        tofill[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :] = output
        filled_img = Image.fromarray(tofill)
        filled_img.save(save_dir + "validation_filled_" + num + '.jpg')

    test_decoded_imgs = autoencoder.predict(test_set_x)
   
    for i in range(n):
        array=test_decoded_imgs[i].reshape(32, 32, 3)*255.
        output = Image.fromarray(array.astype('uint8'))
        num= "%05d" % i
        output.save(save_dir + "test_center_" + num + '.jpg')
        test_array=test_set_x[i].reshape(64, 64, 3)*255.
        test_full = Image.fromarray(test_array.astype('uint8'))
        tofill = np.array(test_full)
        center = (int(np.floor(tofill.shape[0] / 2.)), int(np.floor(tofill.shape[1] / 2.)))
        tofill[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :] = output
        filled_img = Image.fromarray(tofill)
        filled_img.save(save_dir + "test_filled_" + num + '.jpg')
