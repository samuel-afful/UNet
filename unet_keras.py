# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:33:45 2024

@author: user
"""

from keras.models import Model
from keras.layers import Conv2D,Activation,Input,MaxPool2D,Conv2DTranspose,Concatenate,BatchNormalization
import tensorflow as tf

def conv_block(inputs,num_filter):
    x = Conv2D(filters=num_filter, kernel_size=3,padding='same')(inputs)
   # x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    
    x = Conv2D(filters=num_filter, kernel_size=3,padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
        

    return x




def downsample(inputs,num_filter):
    x = conv_block(inputs=inputs,num_filter=num_filter)
    p = MaxPool2D(pool_size=(2,2),strides=2)(x)
    return x,p

def upsample(inputs,skip_features,num_filter):
    x = Conv2DTranspose(filters=num_filter, kernel_size=2,padding='same')(inputs)
    x = Concatenate()([x,skip_features])
    out = conv_block(inputs=x, num_filter=num_filter)
    return out

def Unet(inputs):
    inputs = Input(shape=inputs)
    s1,p1 = downsample(inputs=inputs, num_filter=64)
    s2,p2= downsample(inputs=p1, num_filter=128)
    s3,p3 = downsample(inputs=p2, num_filter=256)
    s4,p4= downsample(inputs=p3, num_filter=512)
    
    
    bottle_neck = conv_block(inputs=s4, num_filter=1024)
    
    u_1 = upsample(inputs=bottle_neck, num_filter=512, skip_features=s4)
    u_2 = upsample(inputs=u_1, num_filter=256, skip_features=s3)
    u_3 = upsample(inputs=u_2, num_filter=128, skip_features=s2)
    u_4 = upsample(inputs=u_3, num_filter=64, skip_features=s1)
    
    out = Conv2D(filters=1, kernel_size=1,activation='sigmoid',padding='same')(u_4)
    
    return Model(inputs=inputs, outputs=out,name='U-net')

    

image = tf.random.normal((265,265,1))

model.summary()
    
    
        
    

