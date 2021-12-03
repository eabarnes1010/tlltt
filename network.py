import sys
import numpy as np
import tensorflow as tf

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "02 Decmeber 2021"


def get_model_prototype_layer(model):
    return tf.keras.models.Model(model.input,model.layers[-3].output)

def get_model_final_conv_layer(model):
    return tf.keras.models.Model(model.input,model.layers[-5].output)

def get_model_cnn_only_conv_layer(model):
    return tf.keras.models.Model(model.input,model.layers[-6].output)


class Prototype(tf.keras.layers.Layer):
    def __init__(self, nprototypes, nchannels, prototype_class_identity, coeff_cluster, coeff_separation, network_seed):
        super(Prototype, self).__init__()

        initializer = tf.random_uniform_initializer(minval=0., maxval=1., seed=network_seed)
        self.prototypes = tf.Variable(
            initializer(shape=(1, 1, nchannels, nprototypes), dtype=tf.float32),
            trainable=True, 
            name='prototypes',
        )
        self.proto_class_mask = prototype_class_identity

        self.max_dist = tf.constant(100.*1.*1.*nchannels)   #tf.constant(1000.)#tf.constant(1.*1.*nchannels)
        self.coeff_cluster = coeff_cluster
        self.coeff_separation = coeff_separation
        self.coeff_local_scale_l1 = .00001  #.00001
        
        self.ONES = tf.constant(1, shape=(1, 1, nchannels, nprototypes), dtype=tf.float32)
        self.EPSILON = tf.constant(1e-4, shape=(1), dtype=tf.float32)

        
    def build(self, input_shape):  # Create the state of the layer (weights)
        local_scale = tf.zeros_initializer()
        self.local_scale = tf.Variable(
            initial_value=local_scale(
                shape=(input_shape[1], input_shape[2], tf.shape(self.prototypes)[-1]),
                dtype='float32'),
            trainable=True, #set to False for "maskoff"
        )

        
    def call(self, inputs, prototypes_of_correct_class):
        """ For each ...
        
        Arguments:
            inputs   tf.Tensor(shape = (batch_size, H, W, nchannels))
            
            prototypes_of_correct_class    tf.Tensor(shape = (batch_size, nprototypes))
                ones and zeros only
            
        Returns:
            max_similarity_scores   tf.Tensor(shape = (batch_size, nprototypes))
            
        Notes:
        -- prototypes          tf.Tensor(shape = (1, 1, nchannels, nprototypes))
        -- distances           tf.Tensor(shape = (batch_size, H, W, nprototypes))
        -- similarity_scores   tf.Tensor(shape = (batch_size, H, W, nprototypes))
        
        -- The underlying computation is the squared L2 norm for the difference of 
           two vectors, say x and y.  We can write this computation (using 
           MATLAB-like notation) as 
            
                (x - y)' * (x - y) = (x' * x) - 2 * (x' * y) + (y' * y)
            
        -- In our case, the "x" and "y" vectors are of length nchannels.
        
        -- We have an "x" vector for each sample in the batch and for each row and column
           in H x W (i.e. for each patch). 
           
        -- We have a "y" vector for each prototype.
            
        -- The use of "relu" in the computation of normsq eliminates problems brought on 
           by very small rounding and truncation errors yielding negative normsq.
        
        """
        local_scale_factor = tf.math.exp(self.local_scale)                                        # shape = (H, W, nprototypes)
        
        xTx = tf.nn.conv2d(inputs**2, self.ONES,       strides=[1, 1, 1, 1], padding='VALID')     # shape = (batch_size, H, W, nprototypes)
        xTy = tf.nn.conv2d(inputs,    self.prototypes, strides=[1, 1, 1, 1], padding='VALID')     # shape = (batch_size, H, W, nprototypes)
        yTy = tf.math.reduce_sum(self.prototypes**2, axis=[2])                                    # shape = (1, 1, nprototypes)
        normsq = tf.nn.relu(xTx - 2*xTy + yTy)                                                    # shape = (batch_size, H, W, nprototypes)
        
        min_distances = tf.math.reduce_min(                                                       # shape = (batch_size, nprototypes)
            normsq / (local_scale_factor + self.EPSILON), 
            axis=[1, 2]
        )

        # Cluster cost
        inverted_cluster_cost = tf.math.reduce_max(                                               # shape = (batch_size,) 
            (self.max_dist - min_distances) * prototypes_of_correct_class, 
            axis=-1
        )          
        cluster_cost = tf.math.reduce_mean(self.max_dist - inverted_cluster_cost)                 # shape = (1,)
        self.add_loss(self.coeff_cluster * cluster_cost)                                          # shape = (1,)
        self.add_metric(cluster_cost, 'cluster_cost')
        
        # Separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_separation_cost = tf.math.reduce_max(                                            # shape = (batch_size,)
            (self.max_dist - min_distances) * prototypes_of_wrong_class, 
            axis=-1
        ) 
        separation_cost = tf.math.reduce_mean(self.max_dist - inverted_separation_cost)           # shape = (1,)
        self.add_loss(self.coeff_separation * separation_cost)                                    # shape = (1,)
        self.add_metric(separation_cost, 'separation_cost')        

        # Similarity scores
        similarity_scores = tf.math.log((normsq + 1) / (normsq + self.EPSILON))                     # shape = (batch_size, H, W, nprototypes)
        scaled_similarity_scores = tf.math.multiply(similarity_scores, local_scale_factor)        # shape = (batch_size, H, W, nprototypes)
        
        return tf.math.reduce_max(scaled_similarity_scores, axis=[1, 2], name="max_similarity_scores")
    
    
def createClassIdentity(prototypes_per_class):
    nclasses = len(prototypes_per_class)
    nprototypes = np.sum(prototypes_per_class)

    mask = np.zeros((nprototypes, nclasses), dtype=np.float32)

    start = 0
    for j, n in enumerate(prototypes_per_class):
        for i in range(start, start+n):
            mask[i, j] = 1
        start = start + n
    
    return mask


class FinalWeights(tf.keras.layers.Layer):

    def __init__(self, units, prototype_class_identity, coeff_l1, incorrect_strength, name='final_weights'):
        super(FinalWeights, self).__init__(name=name)
        self.units = units
        self.mask = prototype_class_identity
        self.coeff_l1 = coeff_l1
        self.incorrect_strength = incorrect_strength

    def build(self, input_shape):        
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.Ones(),
#             trainable=True,
            name='weights_tensor',
        )           

        init_weights = self.mask + self.incorrect_strength * (1-self.mask)
        init_weights = np.expand_dims(init_weights,axis=0)
        self.set_weights(init_weights)

    def call(self, inputs):
        
        l1_mask = 1 - self.mask
        l1 = tf.norm(tf.math.multiply(self.w,l1_mask), ord=1, name='l1_norm')
        self.add_loss(tf.math.multiply(self.coeff_l1, l1))
        self.add_metric(tf.math.multiply(self.coeff_l1, l1), 'l1_weights_cost')        
        
        return tf.matmul(inputs, self.w)  
    
def build_model(nlayers, 
                nfilters, 
                input_shape, 
                output_shape, 
                prototypes_per_class, 
                network_seed,
                double_conv=False,
                coeff_cluster=0.05,
                coeff_separation=-0.005,
                coeff_l1=0.01,
                incorrect_strength=-0.5,
                cnn_only=False,
                dense_nodes=8,
                prototype_channels=32,
                kernel_l1_coeff=0.,
                kernel_l2_coeff=0.,
                drop_rate=0.,
                drop_rate_final=0.,
                is_tuner = False,
                hp=None,
               ):

    if(is_tuner):
        if(hp==None):
            raise ValueError('cannot run tuner without HYPERPARAMTERS "hp"')
            
        double_conv              = hp.Boolean('double_conv', default=False)
        nlayers                  = hp.Int('nlayers',1,3,1)
        nfilt                    = hp.Choice('nfilters',[8,16,32,64])
        nfilters                 = np.ones(2*nlayers)*nfilt
        kernel_l1_coeff          = hp.Float('kernel_l1_coeff',0.0001,1.,sampling='log',default=0.001)
        kernel_l2_coeff          = hp.Float('kernel_l2_coeff',0.0001,1.,sampling='log',default=0.001)
        drop_rate                = hp.Float('drop_rate',0.,.5,step=.2,default=0.)
        drop_rate_final          = hp.Float('drop_rate_final',0.,.5,step=.2,default=.5)       
        dense_nodes              = hp.Choice('dense_nodes',[8,32,64])

        prototype_channels       = hp.Choice('nprotochannels',[8,16,32,64])    
        coeff_cluster            = hp.Float('coeff_cluster',0.0001,1.,sampling='log',default=0.01)
        coeff_separation         = -coeff_cluster/10.

        
        
    ACT_FUN                  = 'relu'    
    KERNEL_SIZE              = (3,3)
    STRIDES                  = (1,1)
    POOL_SIZE                = (2,2)
    POOL_STRIDE              = 2
    NPROTOTYPES              = np.sum(prototypes_per_class)
    PROTOTYPE_CLASS_IDENTITY = createClassIdentity(prototypes_per_class)
    
    print(nlayers)
    print(nfilters)
    print(input_shape)
    print(output_shape)
    print(prototypes_per_class)
    print(network_seed) 
    print(double_conv)
    print(coeff_cluster)
    print(coeff_separation)
    print(coeff_l1)
    print(incorrect_strength)
    print(cnn_only)
    print(dense_nodes)
    print(prototype_channels)
    print(kernel_l1_coeff)
    print(kernel_l2_coeff)
    print(drop_rate)
    print(drop_rate_final)
    print(is_tuner)
    

    
    inputs = tf.keras.Input(shape=input_shape, name='inputs')
    prototypes_of_correct_class = tf.keras.Input(shape=(NPROTOTYPES,), name="prototypes_of_correct_class")    
    x = inputs

    # first layer
    x = tf.keras.layers.Conv2D(
        nfilters[0], 
        KERNEL_SIZE, 
        strides=STRIDES, 
        padding='same', 
        data_format='channels_last',
        activation=ACT_FUN, 
        kernel_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
        bias_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
        kernel_regularizer =tf.keras.regularizers.l1_l2(kernel_l1_coeff, kernel_l2_coeff),     
        name='conv_0',
    )(x)
    x = tf.keras.layers.Dropout(rate=drop_rate,seed=network_seed)(x) 
    
    if(double_conv==True):
        x = tf.keras.layers.Conv2D(
            nfilters[0], 
            KERNEL_SIZE, 
            strides=STRIDES, 
            padding='same', 
            data_format='channels_last',
            activation=ACT_FUN, 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
            bias_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
            kernel_regularizer =tf.keras.regularizers.l1_l2(kernel_l1_coeff, kernel_l2_coeff),     
            name='conv_0x2',
        )(x)        
        x = tf.keras.layers.Dropout(rate=drop_rate,seed=network_seed)(x)         

    # first layer's max pooling
    x = tf.keras.layers.AveragePooling2D(
        pool_size=POOL_SIZE, 
        strides=POOL_STRIDE, 
        padding='valid',
        name='maxpooling_0',
    )(x)
#     x = tf.keras.layers.BatchNormalization()(x)
    
    # additional layers and max pooling
    for layer in np.arange(1,nlayers):
        
        # initialize layer
        x = tf.keras.layers.Conv2D(
            nfilters[layer], 
            KERNEL_SIZE, 
            strides=STRIDES, 
            padding='same', 
            data_format='channels_last',
            activation=ACT_FUN, 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
            bias_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
            kernel_regularizer =tf.keras.regularizers.l1_l2(kernel_l1_coeff, kernel_l2_coeff),
            name='conv_' + str(layer),         
        )(x)      
        x = tf.keras.layers.Dropout(rate=drop_rate,seed=network_seed)(x)                 
        
        if(double_conv==True):
            x = tf.keras.layers.Conv2D(
                nfilters[layer], 
                KERNEL_SIZE, 
                strides=STRIDES, 
                padding='same', 
                data_format='channels_last',
                activation=ACT_FUN, 
                kernel_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
                bias_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
                kernel_regularizer =tf.keras.regularizers.l1_l2(kernel_l1_coeff, kernel_l2_coeff),
                name='conv_' + str(layer) + 'x2',         
            )(x)
            x = tf.keras.layers.Dropout(rate=drop_rate,seed=network_seed)(x)         
            
        # layer's max pooling
        x = tf.keras.layers.AveragePooling2D(
            pool_size=POOL_SIZE, 
            strides=POOL_STRIDE, 
            padding='valid',
            name='maxpooling_' + str(layer)
        )(x)      
        
#         x = tf.keras.layers.BatchNormalization()(x)
        
    #----------------------------------------        
    if(cnn_only==False):    

        # first 1x1 convolutional layer
        x = tf.keras.layers.Conv2D(
            prototype_channels, 
            1, 
            strides=STRIDES, 
            padding='same', 
            data_format='channels_last',
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
            bias_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
            name= 'first_1x1_conv',
        )(x)     

        # second 1x1 convolutional layer    
        x = tf.keras.layers.Conv2D(
            prototype_channels, 
            1, 
            strides=STRIDES, 
            padding='same', 
            data_format='channels_last',
            activation='relu',               # <-------- did not seem to work at all with "sigmoid"
            kernel_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
            bias_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
            name= 'second_1x1_conv',  
        )(x) 

        # compute L2 similarity scores, outputs matrix of similarities
        x = Prototype(NPROTOTYPES, 
                      prototype_channels, 
                      PROTOTYPE_CLASS_IDENTITY,
                      coeff_cluster,
                      coeff_separation,
                      network_seed,
                     )(x, prototypes_of_correct_class)

        # final dense layer    
        x = FinalWeights(units=output_shape, 
                         prototype_class_identity=PROTOTYPE_CLASS_IDENTITY,
                         coeff_l1=coeff_l1, 
                         incorrect_strength=incorrect_strength,
                        )(x)

    else:
        # flatten layer
        x = tf.keras.layers.Flatten()(x)
        
        # dropout layer (not required)
        x = tf.keras.layers.Dropout(rate=drop_rate_final,seed=network_seed)(x)    
        
        # final dense layer
        x = tf.keras.layers.Dense(dense_nodes,
                                  activation='relu',
                                  use_bias=True,
                                  kernel_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
                                  bias_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
                                 )(x)
        # final output layer before softmax
        x = tf.keras.layers.Dense(output_shape,
                                  activation='relu',
                                  use_bias=True,
                                  kernel_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
                                  bias_initializer=tf.keras.initializers.HeNormal(seed=network_seed),
                                 )(x)        
        
    # final softmax layer
    x = tf.keras.layers.Softmax(name='softmax_output')(x)
    
    # finalize the model
    if(cnn_only==False):    
        model = tf.keras.models.Model(inputs=[inputs, prototypes_of_correct_class],outputs=x, name='full_model')
    else:
        model = tf.keras.models.Model(inputs=inputs,outputs=x, name='full_model')
    
    return model



def set_trainable_layers(model,trainable):
    """   
        
    Arguments:
        model        instantiated tensorflow model
        trainable    list of boolean True, False denoting whether layer is trainable

    Returns:
        model        updated tensorflow model with trainability of layers modified

    Notes:
    -- trainable is a list of length 4 specifying trainability for a layer/groups of layers
       0 --> Base Convolutional Layers
       1 --> 1x1 Convolutional Layers
       2 --> Prototype Layer
       3 --> Final Weights Layer

    """    
    
#     print('Setting layer trainability...')
    
    for layer in range(0,len(model.layers)):
        name = model.layers[layer].name
        if(name[:4]=='conv' or name[:10]=='maxpooling'):
            model.layers[layer].trainable = trainable[0]
            print('   ' + name + ' --> ' + str(model.layers[layer].trainable))
        elif(name[-8:]=='1x1_conv'):
            model.layers[layer].trainable = trainable[1]
            print('   ' + name + ' --> ' + str(model.layers[layer].trainable))
        elif(name=='prototype'):
            model.layers[layer].trainable = trainable[2]
            print('   ' + name + ' --> ' + str(model.layers[layer].trainable))
        elif(name=='final_weights'):
            model.layers[layer].trainable = trainable[3]
            print('   ' + name + ' --> ' + str(model.layers[layer].trainable))

    return model

class ReceptiveField:
    """
    Determines the set of input pixels that impact the out put pixels.
    
    Attributes
    ----------
    mask_shape : (# of input row, # of input columns)
    
    imin : numpy.array(# of output rows,)
        imin(m) index of the first input row impacting the m'th output row.

    imax : numpy.array(# of output rows,)
        imax(m) index of the last input row impacting the m'th output row.

    jmin : numpy.array(# of output columns,)
        jmin(n) index of the first input column impacting the n'th output column.

    jmax : numpy.array(# of output columns,)
        jmax(n) index of the last input column impacting the n'th output column.

    Methods
    -------
    __init__(self, model):
        Compute the input to output pixel mapping for the given tensorflow model. 
        
    call(m, n):
        Return a binary mask indicating the input pixels that impact output pixel (m, n).           
    
    Notes
    -----
    o The class uses a brute-force "ping" approach for determining the mapping 
    between the input and output pixels.  This approach is slow.  But, this approach 
    will work for any combination of Conv2D and MaxPooling2D layers. 
    
    o This code assumes that all of the the convolution layers are all: data_format='channels_last'.
    
    o 21Jul21: fixed initialization bug with self.imin and self.jmin. (RJB)
    """
    
    def __init__(self, model):
        """
        Compute the input to output pixel mapping for the given tensorflow model.
        
        Arguments
        ---------
        model : tensorflow.keras.Model
        
        Notes
        -----
        o The prototype layers and beyond should be truncated from the argument model.
        
        """
        clone = tf.keras.models.clone_model(model)
        clone._name = 'cloned_model'

        input_shape = clone._build_input_shape                
        output_shape = clone.output.shape

        self.mask_shape = (input_shape[1], input_shape[2])
        
        for layer in clone.layers:
            if type(layer).__name__ == "Conv2D":
                layer.activation = tf.keras.activations.linear
                layer.bias.assign(np.zeros(layer.bias.shape, np.float32))
                layer.kernel.assign(np.ones(layer.kernel.shape, np.float32))
        
        # Notation: (i,j) indexes the input, (m,n) indexes the output.
        self.imin = np.full(shape=output_shape[1], fill_value=np.iinfo(int).max, dtype=np.int)
        self.imax = np.full(shape=output_shape[1], fill_value=-1,                dtype=np.int)
        self.jmin = np.full(shape=output_shape[2], fill_value=np.iinfo(int).max, dtype=np.int)
        self.jmax = np.full(shape=output_shape[2], fill_value=-1,                dtype=np.int)

        n_channels = model.layers[0].get_input_shape_at(0)[-1]
        sample = np.zeros(shape=(1, input_shape[1], input_shape[2], n_channels), dtype=np.float32)


        for i in range(input_shape[1]):
            sample[0, i, 0, 0] = 1
            prediction = clone.predict(sample)
            sample[0, i, 0, 0] = 0               

            result = tf.math.reduce_max(prediction, axis=[-1])
            for m in range(output_shape[1]):
                if result[0, m, 0] != 0:
                    self.imin[m] = min(self.imin[m], i)
                    self.imax[m] = max(self.imax[m], i)
        
        for j in range(input_shape[2]):
            sample[0, 0, j, 0] = 1
            prediction = clone.predict(sample)
            sample[0, 0, j, 0] = 0               

            result = tf.math.reduce_max(prediction, axis=[-1])
            for n in range(output_shape[2]):
                if result[0, 0, n] != 0:
                    self.jmin[n] = min(self.jmin[n], j)
                    self.jmax[n] = max(self.jmax[n], j)

    def computeMask(self, m, n):
        """
        Return a binary mask indicating the input pixels that impact output pixel (m, n).
        
        Arguments
        ---------
        m : int
            output row of interest.
            
        n : int
            output column of interest.
        
        Returns
        -------
        mask : numpy.array(# of input rows, # of input columns)
            mask(i, j) is 1 if input pixel (i, j) impacts output pixel (m, n), and is 0 otherwise.
        
        """
        mask = np.zeros(self.mask_shape)
        mask[self.imin[m]:self.imax[m]+1, self.jmin[n]:self.jmax[n]+1] = 1
        return mask