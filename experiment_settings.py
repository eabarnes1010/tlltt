"""This Looks Like That There experimental settings"""

import numpy as np

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "23 November 2021"

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def get_settings(exp_name):
    
    exps = {
           'quadrants': {
                        'data_name'           : 'data_quadrants',
                        'data_dir'            : './data/',
                        'colored'             : False, 
                        'standardize'         : 'all',
                        'shuffle'             : True,
                
                        'nclasses'            : 3,
                        'nlayers'             : 2, 
                        'nfilters'            : [32,32],
                        'double_conv'         : False,
                
                        'prototypes_per_class': [5,5,5],
                
                        'pretrain'            : True,
                        'pretrain_exp'        : None,
                        'nepochs_pretrain'    : 30,
                        'lr_cb_epoch_pretrain': 100,                
                        'lr_pretrain'         : 0.00005,            
                        'pretrain_patience'   : 5,                
                        'train_cnn_in_stage'  : True,
               
                
                        'nepochs'             : np.ones(15,dtype=np.int8)*10,
                        'lr_cb_epoch'         : 1000,                
                        'lr'                  : 0.01,
                        'cut_lr_stage'        : 6,   
                        'min_lr'              : 1.e-20,                              
                        'batch_size'          : 32,
                        'random_seed'         : 30,
                        'batch_size_predict'  : 1_024, 
                        'patience'            : 5,
                
                        'incorrect_strength'  : -0.5,                
                        'coeff_cluster'       :  0.17197201619672103,
                        'coeff_separation'    : -0.17197201619672103/10.,
                        'coeff_l1'            : 0.5,
                        'dense_nodes'         : 64,
                        'prototype_channels'  : 128,
                        'kernel_l1_coeff'     : 0.0,
                        'kernel_l2_coeff'     : 0.0,
                        'drop_rate'           : 0.0,
                        'drop_rate_final'     : .4,
               
                        'analyze_stage'      : 5,               
                       },    
           'mjo': {
                        'data_name'           : 'mjo',
                        'data_dir'            : './data/',         
                        'colored'             : False, 
                        'standardize'         : 'all',
                        'shuffle'             : False,
                
                        'nclasses'            : 9,
                        'nlayers'             : 3, 
                        'nfilters'            : [16,16,16],
                        'double_conv'         : False,
                
                        'prototypes_per_class': [10,10,10,10,10,10,10,10,10],
                
                        'pretrain'            : True,
                        'pretrain_exp'        : None,
                        'nepochs_pretrain'    : 100, 
                        'lr_cb_epoch_pretrain': 100,                
                        'lr_pretrain'         : 0.00017548,            
                        'pretrain_patience'   : 5,                
                        'train_cnn_in_stage'  : True,
                               
                        'nepochs'             : np.ones(15,dtype=np.int8)*10,
                        'lr_cb_epoch'         : 1000,                
                        'lr'                  : 0.01,
                        'cut_lr_stage'        : 4,   
                        'min_lr'              : 1.e-20,               
                        'batch_size'          : 32,
                        'random_seed'         : 30,
                        'batch_size_predict'  : 1_024, 
                        'patience'            : 5,
                
                        'incorrect_strength'  : -0.5,                
                        'coeff_cluster'       :  0.2,
                        'coeff_separation'    : -0.2/10.,
                        'coeff_l1'            : 0.1,
                        'dense_nodes'         : 32,
                        'prototype_channels'  : 64,
                        'kernel_l1_coeff'     : 0.00,
                        'kernel_l2_coeff'     : 0.00,
                        'drop_rate'           : 0.4,
                        'drop_rate_final'     : 0.2,
               
                        'analyze_stage'      : 9,               
                       },        
        

    }
    
    return exps[exp_name]
  
    
    