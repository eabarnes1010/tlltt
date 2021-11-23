"""This Looks Like That experimental settings"""

import os
import numpy as np
import tensorflow as tf

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "June 20, 2021"

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def save_model(model, model_filename):

    print('saving model and weights to ' + model_filename)
    
    # save the weights only
    model.save_weights(model_filename + '.h5')

    # save the entire model in TF2.5
    tf.keras.models.save_model(model, model_filename, overwrite=True)       

def load_model(model_filename):

    print('loading model from ' + model_filename)

    # Restore the weights
    # model.load_weights('saved_models/' + model_name + '.h5')
    
    # Load full model
    model = tf.keras.models.load_model(model_filename)
    return model


def get_exp_directories(exp_name):
    # make model and figure directories if they do not exist
    
    model_diagnostics_dir = './figures/' + exp_name + '/model_diagnostics/' 
    if not os.path.exists(model_diagnostics_dir):
        os.makedirs(model_diagnostics_dir)   
    
    vizualization_dir = './figures/' + exp_name + '/vizualization/' 
    if not os.path.exists(vizualization_dir):
        os.makedirs(vizualization_dir)
    
    model_dir = './saved_models/' + exp_name + '/' 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
        
    return model_dir, model_diagnostics_dir, vizualization_dir