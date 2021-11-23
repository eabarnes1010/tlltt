"""TLLT push prototypes from training set"""

import numpy as np
import tensorflow as tf
from icecream import ic          # pip install icecream
import network

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "June 28, 2021"


def get_similarity_maps(inputs, prototypes, local_mask, batch_size = 1_024):
    
    nchannels = inputs.shape[3]
    nprototypes = np.shape(prototypes)[-1]
    ONES = tf.constant(1, shape=(1, 1, nchannels, nprototypes), dtype=tf.float32)
    EPSILON = tf.constant(1e-4, shape=(1), dtype=tf.float32)

    NBATCHES   = int(np.ceil(np.shape(inputs)[0]/batch_size))    
    similarity_scores = np.zeros(shape=(np.shape(inputs)[0], np.shape(inputs)[1], np.shape(inputs)[2], nprototypes))
    
    for i in range(0,NBATCHES):
        batch_range = range(i*batch_size, min((i+1)*batch_size, np.shape(inputs)[0]))

        xTx = tf.nn.conv2d(inputs[batch_range,:,:,:]**2, ONES, strides=[1, 1, 1, 1], padding='VALID')           # shape = (batch_size, H, W, nprototypes)
        xTy = tf.nn.conv2d(inputs[batch_range,:,:,:],    prototypes, strides=[1, 1, 1, 1], padding='VALID')     # shape = (batch_size, H, W, nprototypes)
        yTy = tf.math.reduce_sum(prototypes**2, axis=[2])                                                       # shape = (1, 1, nprototypes)
        norms = tf.nn.relu(xTx - 2*xTy + yTy)                                                                   # shape = (batch_size, H, W, nprototypes)
        x = tf.math.log((norms + 1) / (norms + EPSILON))                                                        # shape = (batch_size, H, W, nprototypes)
        similarity_scores[batch_range,:,:,:] = x.numpy()
    
    similarity_scores = tf.math.multiply(similarity_scores, local_mask)
    return similarity_scores


def push(model, input_images, prototypes_of_correct_class_train, perform_push=False, batch_size=1_024, verbose=1):
    
    print('Running Prototype Push')
    
    model_prototype_layer = network.get_model_prototype_layer(model)
    # # model_prototype_layer.summary()

    model_final_conv_layer = network.get_model_final_conv_layer(model)
    # # model_final_conv_layer.summary()

    # run model.predict that can take up all of the memory
    nsamples = np.shape(input_images[0])[0]
    if(nsamples>9_000):
        nsamples_cut = int(nsamples/3.)

        m = model_final_conv_layer.predict([input_images[0][:2,:,:,:], input_images[1][:2,:]], 
                                                                         batch_size=32, 
                                                                         verbose=0)

        inputs_to_prototype_layer = np.zeros(shape=(nsamples,m.shape[1],m.shape[2],m.shape[3]), dtype=np.float32)       
        
        inputs_to_prototype_layer[:nsamples_cut,:,:,:] = model_final_conv_layer.predict([input_images[0][:nsamples_cut,:,:,:], input_images[1][:nsamples_cut,:]], 
                                                                     batch_size=batch_size, 
                                                                     verbose=1)        
        inputs_to_prototype_layer[nsamples_cut:-nsamples_cut,:,:,:] = model_final_conv_layer.predict([input_images[0][nsamples_cut:-nsamples_cut,:,:,:], input_images[1][nsamples_cut:-nsamples_cut,:]], 
                                                             batch_size=batch_size, 
                                                             verbose=1)
        inputs_to_prototype_layer[-nsamples_cut:,:,:,:] = model_final_conv_layer.predict([input_images[0][-nsamples_cut:,:,:,:], input_images[1][-nsamples_cut:,:]], 
                                                             batch_size=batch_size, 
                                                             verbose=1)        
    else:
        inputs_to_prototype_layer = model_final_conv_layer.predict(input_images, batch_size=batch_size, verbose=1)

#     raise ValueError('here')
    prototypes                = model.layers[-3].get_weights()[0]
    local_mask                = model.layers[-3].get_weights()[1]
    local_mask                = tf.math.exp(local_mask)
    similarity_scores         = get_similarity_maps(inputs_to_prototype_layer, prototypes, local_mask, batch_size=batch_size)

    new_prototype_sample      = np.zeros((similarity_scores.shape[-1],), dtype='int')
    new_prototype_sample_sim  = np.zeros((similarity_scores.shape[-1],))
    new_prototypes            = np.zeros(np.shape(prototypes))
    new_prototypes_indices    = np.zeros((similarity_scores.shape[-1],2), dtype='int')
    
    for prototype_index in range(0,similarity_scores.shape[-1]):

        m = np.max( similarity_scores[:,:,:,prototype_index], axis=(1,2) )
        m = m*prototypes_of_correct_class_train[:,prototype_index]  # only allow prototypes from correct class

        new_prototype_sample[prototype_index]     = int(np.argmax(m))
        new_prototype_sample_sim[prototype_index] = np.max(m)   

        x = similarity_scores[new_prototype_sample[prototype_index],:,:,prototype_index]
        j,k = np.unravel_index(np.argmax(x), shape=x.shape)
        push_prototype = inputs_to_prototype_layer[new_prototype_sample[prototype_index],j,k,:]
        new_prototypes[0,0,:,prototype_index] = push_prototype
        new_prototypes_indices[prototype_index,0] = j
        new_prototypes_indices[prototype_index,1] = k
        
        previous_prototype = prototypes[0,0,:,prototype_index]
        corr_coeff = np.corrcoef(push_prototype,previous_prototype)[0,1]
        if(verbose==1):
            print('    Prototype #' + str(prototype_index))
            print('      sim(old,new) = ' + str(np.round(new_prototype_sample_sim[prototype_index],3)))
            print('      r(old,new)   = ' + str(np.round(corr_coeff,3)))
        
    if(perform_push == True):
        # perform push
        print('Performing push of prototypes.')
        model.layers[-3].set_weights([new_prototypes[:,:,:,:],model.layers[-3].get_weights()[1][:,:,:]])

    return model, (new_prototype_sample, new_prototype_sample_sim, new_prototypes, similarity_scores, new_prototypes_indices)