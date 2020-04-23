#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:42:07 2020

@author: qasymjomart
made in collaboration with the Capsnet implementation from:
https://keras.io/examples/cifar10_cnn_capsule/
"""


from __future__ import print_function
from keras import backend as K
import keras
from keras.layers import Layer
from keras import activations
# from keras import utils
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import *
from keras.callbacks import LearningRateScheduler

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from train_utils_bci import (NDStandardScaler, subject_specific, leave1out, importseveralsubjects, pad_with_zeros, pad_by_duplicating, generator)
from data_pooler_bci import data_pooler

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

#%%
class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                        )
    and the output shape is (batch_size,
                              num_capsule,
                              dim_capsule
                            )

    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                  num_capsule=2,
                  dim_capsule=256,
                  routings=3,
                  share_weights=False,
                  activation='squash',
                  **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                        self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                        self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                                (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    
    def get_config(self):
        config = {
        'num_capsule': self.num_capsule,
        'dim_capsule': self.dim_capsule

        }
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#%%

batch_size = 32
num_classes = 2
conv = [16,16,16]
kernel = 3
cap = 128
dense = 256
lr = 10e-4
epochs=50
shape = (16,78)

for ss in range(7,9,1):

    train_norm, y_train, val_subjects_norm, test_subjects_norm = data_pooler(dataset_name='TenHealthyData', subIndexTest = ss)
    x_train = train_norm
    x_val,y_val = val_subjects_norm[0]['xtrain'], val_subjects_norm[0]['ytrain']
    x_test,y_test = test_subjects_norm[0]['xtrain'], test_subjects_norm[0]['ytrain']
    del train_norm, val_subjects_norm, test_subjects_norm
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    totalTrain = len(x_train)
    totalVal = len(x_val)
    
    
    # A common Conv2D model
    print(str(conv), str(kernel), cap, dense, lr)
    input_image = Input(shape=(16, 78, 1))
    x = Conv2D(conv[0], (kernel, kernel), strides=2, activation = 'relu')(input_image)
    # x = Conv2D(64, (3, 3), activation='selu')(x)
    # x = MaxPooling2D((1, 2))(x)
    # x = Conv2D(conv[1], (kernel, kernel), strides=2, activation = 'relu')(x)
    # x = MaxPooling2D((1, 2))(x)
    # x = Conv2D(conv[2], (kernel, kernel), strides=2, activation = 'relu')(x)
    # x = MaxPooling2D((1, 2))(x)
    
    
    x = Reshape((-1, conv[-1]))(x)
    capsule = Capsule(16,2,3,False)(x)
    # output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(0.2)(capsule)
    # capsule = Dense(32, activation = 'relu')(capsule)
    capsule = Dense(dense, activation = 'relu')(capsule)
    output = Dense(1, activation='sigmoid')(capsule)
    model = Model(inputs=input_image, outputs=output)
    
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['acc'])
    
    
    
    train_gen = generator(x_train,
    	              y_train, 
    	              min_index=0,
    	              max_index=None,
    	              batch_size=batch_size,
                      desired_size = shape,                 #************WARNING************
    	              shuffle=True) #see what None does
    
    val_gen = generator(x_val,
                        y_val,
                        min_index=0,
                        max_index=None,
                        batch_size=batch_size,
                        desired_size = shape,                   #************WARNING************
                        shuffle=True)
    
    mcp_save = keras.callbacks.ModelCheckpoint("Test-subject-"+str(ss)+"_CapsNet_New_ERP_Best_model#"+str(epochs)+ ".h5",monitor='val_acc', verbose=0, 
    			save_best_only=True, save_weights_only=False, mode='max', period=1)
    
    H = model.fit_generator(
    			train_gen,
    			steps_per_epoch=totalTrain // batch_size,
    			validation_data=val_gen,
    			validation_steps=totalVal // batch_size,
    			epochs=epochs,
    			callbacks = [mcp_save])
    
    
    
    test_gen = generator(x_test,
                             y_test,
                             min_index=0,
                             max_index=None,
                             batch_size=batch_size,
                             desired_size = shape                  #************WARNING************
                             #color_mode="grayscale",
                             )
    
    model.save("Test-subject-"+str(ss)+"_CapsNet_ERP_model#"+str(epochs) + ".h5")

    model = load_model("Test-subject-"+str(ss)+"_CapsNet_New_ERP_Best_model#"+str(epochs) + ".h5", custom_objects={'Capsule':Capsule})
    
    fname1 = "Test-subject-"+str(ss)+"_AllLayer_modelHistory#"+str(epochs)+".txt"
    f = open(fname1,"w+")
    
    totalTest = len(x_test)
    predIdxs = model.predict_generator(test_gen,verbose=2,
    						steps=(totalTest // batch_size))
    
    predIdxs = np.where(predIdxs >= 0.5, 1, 0)
    predIdxs = predIdxs.reshape(len(predIdxs),)
    predIdxs = np.float32(predIdxs)
    #print(classification_report(test_gen.classes, predIdxs, target_names=y_test))
    			#test_gen.reset()
    auc = roc_auc_score(y_test[0:len(predIdxs)],predIdxs)
    print(auc)
    cm = confusion_matrix(y_test[0:len(predIdxs)], predIdxs)      #CHANGED EVERY testGen to valGen
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity0 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    			#sensitivity2 = cm[2, 2] / (cm[2, 2] + cm[2, 0] + cm[2,1] + cm[2,3])
    			#sensitivity3 = cm[3, 3] / (cm[3, 3] + cm[3, 0] + cm[3,1] + cm[3,2])
    sensitivity = (sensitivity1+sensitivity0)/2
    f.write("Test subject \n" + str(ss) + "\nValidation acc: " + str(round(H.history['val_acc'][epochs-1],4)) + "; Training acc: " + str(round(H.history['acc'][epochs-1],4)) + 
    				"\nAUC: " + str(round(auc,4)) + "; Sensitivity (avg): " + str(round(sensitivity,4)) + "; Test acc: " + str(round(acc,4)))
    f.write("\nSensitivity0: {:.4f}".format(sensitivity0))
    f.write("\nSensitivity1: {:.4f}".format(sensitivity1))
    f.write("\n" + str(cm))
    f.close()
    
    plt.figure()
    plt.plot(H.history['acc'])
    plt.plot(H.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.savefig('noaug_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_accuracy.png')
    plt.savefig("Test-subject-"+str(ss)+'_img_AllLayer_modelHistory#'+str(epochs)+'_accuracy.png')
      
    plt.figure()
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    #plt.savefig('noaug_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_loss.png')
    plt.savefig("Test-subject-"+str(ss)+'_img_AllLayer_modelHistory#'+str(epochs)+'_loss.png')
    
    del predIdxs
