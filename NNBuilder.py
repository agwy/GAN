import tensorflow as tf
import numpy as np
from SummaryFunctions import *

#----------------Various ways of initalising the parameters----------------------
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
  	
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return initial
  
def bias_variable(shape): 
	#the batch mean. Instead, the role of the bias is performed
	# by the new beta variable. See Section 3.2 of the BN2015 paper.
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
  
  #Random noise - potentially swap out for gaussian later 
def sample_Z(m, n):
    return np.random.normal(0, 1, size=[m, n])	  

    
  #Basic NN layer without link function, combine these 
def nn_layer(input_tensor, input_dim, output_dim, layer_name, double_input = None, dropout = 0):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(xavier_init([input_dim, output_dim]))
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            variable_summaries(preactivate)
            tf.summary.histogram('pre_activations', preactivate)
            if double_input != None:
                preactivate2 = tf.matmul(double_input, weights) + biases
                with tf.name_scope('2'):
                    tf.summary.scalar('min', tf.reduce_min(preactivate2))
                    tf.summary.scalar('max', tf.reduce_max(preactivate2))
            else: preactivate2 = None
            
            if dropout > 0: 
                mask_dropout = tf.nn.dropout(tf.constant(1.0, shape = [1, output_dim]), dropout)
                activate = tf.multiply(preactivate, mask_dropout)
                if double_input != None:
                    activate2 = tf.multiply(preactivate2, mask_dropout)
                else: 
                    activate2 = None                    
            else: 
                activate = preactivate
                activate2 = preactivate2        

    return activate, activate2, weights, biases
    
# NOT WORKING YET: mean and variance after normalization are not always 0 and 1 (see tensorboard summary) Also need tp edit in case of non-double input
# Also, need to correct for dropout? Or after second thought, not.
def bn_layer(input_tensor, layer_name, double_input = None, epsilon = 1e-12):
    with tf.name_scope('batch_norm'):
        if double_input != None:
            input_shape = tf.shape(input_tensor)
            input_shape2 = tf.shape(double_input)
            inputs = tf.concat(0,[input_tensor, double_input])
            mean, variance = tf.nn.moments(inputs, [0])
            norm_inputs = (inputs - mean) / tf.sqrt(tf.maximum(variance, epsilon))
            mean_check, variance_check = tf.nn.moments(norm_inputs, [0])
            tf.summary.scalar('variance_check', variance_check[2])
            tf.summary.scalar('mean_check', mean_check[2])
            # norm_inputs = tf.nn.l2_normalize(inputs, 0, epsilon=epsilon) # Only scales...
            norm_input1 = tf.slice(norm_inputs, [0, 0], input_shape)
            norm_input2 = tf.slice(norm_inputs, [input_shape[0], 0], input_shape2) # TODO: check if this yields the correct output input_shape is the right beginning # TODO: check normalization and 
            return norm_input1, norm_input2
        else: 
            return tf.nn.l2_normalize(input_tensor, 0, epsilon=epsilon), None ## TODO: NEED to edit still
        
         
  #----------Models-Copied from blogpost in slack, should give atleast a reasonable result------------------------------------------
  
  #Generator NN: z -> (100,128) -> reLU -> (128,784) -> Sigmoid
def Generator_NN(input, input_dim, output_dim, dropout = 0.8, bn = 'dont use batchnormalization'): # TODO: check if dropout goes ok.
    hidden_1, _, w1, b1 = nn_layer(input, input_dim, 202, 'layer1_G', dropout = dropout)	
    #if bn == 'l2': hidden_1, _ = bn_layer(hidden_1, 'norm1')			
    hidden_2, _, w2, b2 = nn_layer(tf.nn.relu(hidden_1), 202, output_dim, 'layer2_G')
    return (tf.nn.tanh(hidden_2) + 1)/2,[w1,w2,b1,b2]
  
  #Discrimiantor NN: x -> (784,128) -> reLU -> (128,1) -> sigmoid  
def Discrim_NN(input_x, input_dim, output_dim):
    hidden_1, _, w1, b1 = nn_layer(input_x, input_dim, 128, 'layer1_D')
    hidden_2, _, w2, b2 = nn_layer(tf.nn.relu(hidden_1), 128, output_dim, 'layer2_D')
    return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2]
  	
def Discrim_NN_fixed(input_x, input_G, input_dim, output_dim, dropout = 0.8, bn = 'dont use batchnormalization'):
    hidden_1, hidden_1_G, w1, b1 = nn_layer(input_x, input_dim, 126, 'layer1_D', double_input = input_G, dropout = dropout)
    #if bn == 'l2': hidden_1, hidden_1_G = bn_layer(hidden_1, 'norm1', double_input = hidden_1_G)
    hidden_2, hidden_2_G, w2, b2 = nn_layer(tf.nn.relu(hidden_1), 126, output_dim, 'layer2_D', double_input = hidden_1_G)
    return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2], tf.nn.sigmoid(hidden_2_G)
    
    
    
## DEPRECATED:   
#NOTE: epsilon makes sure that the variance is not estimated to be zero and is added to the variance estimation
#      Also, do NOT use this for the first layer as it has positive bias in the BN-translated inputs (easily modified)
def nn_bn_layer(input_tensor, input_dim, output_dim, layer_name, double_input = None, epsilon=0.01):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(xavier_init([input_dim, output_dim]))
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([1,output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.add( tf.matmul(input_tensor, weights), biases )
            batch_mean1, batch_var1 = tf.nn.moments(preactivate,[0])	#the second argument is simply a speci-
																							#fication of how moments are calculated     
            scale = tf.Variable(tf.ones([output_dim]))		#the scale variable (gamma) that is also learnt in BN
            beta = tf.Variable(tf.zeros([output_dim]))		#the beta variable that is also learnt in BN
             
			   #if you need double input, compute batch mean and variance over both inputs
            if double_input != None:
                preactivate2 = tf.add( tf.matmul(double_input, weights), biases)
                batch_mean2, batch_var2 = tf.nn.moments(preactivate2,[0])
                batch_mean = 0.5*(batch_mean1 + batch_mean2)		#get overall mean
                batch_var = (input_dim-1.0)/(2.0*input_dim - 1.0) * (batch_var1 + batch_var2)	#get overall variance   
                preactivate2 = tf.nn.batch_normalization(preactivate2,batch_mean,batch_var,beta,scale,epsilon)   
            #if you don't need double input, compute batch mean and batch variance only over preactivate1
            else: 
                preactivate2 = None     
            	 #ADD BN only for preactivate 1!
                batch_mean, batch_var = batch_mean1, batch_var1 #so the mean/var from first batch is used exclusively
					 
            preactivate = tf.nn.batch_normalization(preactivate,batch_mean,batch_var,beta,scale,epsilon)
            
            #lastly, add summary writers to the preactivations
            variable_summaries(preactivate)           
            tf.summary.histogram('pre_activations', preactivate)
        return preactivate, preactivate2, weights, biases
    
  
