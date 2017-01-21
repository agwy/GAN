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
         
  #----------Models-Copied from blogpost in slack, should give atleast a reasonable result------------------------------------------
  
  #Generator NN: z -> (100,128) -> reLU -> (128,784) -> Sigmoid
def Generator_NN(input, input_dim, output_dim):
    hidden_1, _, w1, b1 = nn_layer(input, input_dim, 128, 'layer1_G')
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
    
  
