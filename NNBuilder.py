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
    return np.random.uniform(-1., 1., size=[m, n])	  

    
  #Basic NN layer without link function, combine these 
def nn_layer(input_tensor, input_dim, output_dim, layer_name, double_input = None):
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
            else: preactivate2 = None
    return preactivate, preactivate2, weights, biases
    

#NOTE: epsilon makes sure that the variance is not estimated to be zero and is added to the variance estimation
#
def nn_bn_layer(input_tensor, input_dim, output_dim, layer_name, double_input = None, epsilon=0.01):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(xavier_init([input_dim, output_dim]))
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            batch_mean1, batch_var1 = tf.nn.moments(preactivate,[0])	#the second argument is simply a speci-
																							#fication of how moments are calculated     
            scale = tf.Variable(tf.ones([output_dim]))		#the scale variable (gamma) that is also learnt in BN
            beta = tf.Variable(tf.zeros([output_dim]))		#the beta variable that is also learnt in BN
             
			   #if you need double input, compute batch mean and variance over both inputs
            if double_input != None:
                preactivate2 = tf.matmul(double_input, weights) + biases
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
    
    

         
  #----------Models-Copied from blogpost in slack, should give atleast a reasonable result------------------------------------------
  
  #Generator NN: z -> (100,128) -> reLU -> (128,784) -> Sigmoid
def Generator_NN(input, input_dim, output_dim):
    hidden_1, _, w1, b1 = nn_b_layer(input, input_dim, 128, 'layer1_G')				
    hidden_2, _, w2, b2 = nn_layer(tf.nn.relu(hidden_1), 128, output_dim, 'layer2_G')
    return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2]
  
  #Discrimiantor NN: x -> (784,128) -> reLU -> (128,1) -> sigmoid  
def Discrim_NN(input_x, input_dim, output_dim):
    hidden_1, _, w1, b1 = nn_layer(input_x, input_dim, 128, 'layer1_D')
    hidden_2, _, w2, b2 = nn_layer(tf.nn.relu(hidden_1), 128, output_dim, 'layer2_D')
    return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2]
  	
def Discrim_NN_fixed(input_x, input_G, input_dim, output_dim):
    hidden_1, hidden_1_G, w1, b1 = nn_layer(input_x, input_dim, 128, 'layer1_D', double_input = input_G)
    hidden_2, hidden_2_G, w2, b2 = nn_layer(tf.nn.relu(hidden_1), 128, output_dim, 'layer2_D', double_input = hidden_1_G)
    return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2], tf.nn.sigmoid(hidden_2_G)
    
  
