import tensorflow as tf
import numpy as np
from SummaryFunctions import *

#----------------Various ways of initalising the parameters----------------------
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
    
  #Random noise - potentially swap out for gaussian later 
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
    #return np.random.normal(0,1, size=[m, n])  
    
def full_graph(NOISE_DIM):
	#IMPORTANT: Must     
    x_node = tf.placeholder(tf.float32, shape = [None,784])
    z_node = tf.placeholder(tf.float32, shape = [None,NOISE_DIM])
    
    D_W1 = tf.Variable(xavier_init([784, 128]))
    D_b1 = tf.Variable(tf.zeros(shape=[128]))
    D_mask1 = tf.Variable(tf.ones(shape = [1,128])) # For dropout
    D_beta1 = tf.zeros(shape = [1, 128]); D_scale1 = tf.ones(shape = [1, 128]) # For BN
    
    D_W2 = tf.Variable(xavier_init([128, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    theta_d = [D_W1, D_W2, D_b1, D_b2]  
    
    G_W1 = tf.Variable(xavier_init([NOISE_DIM, 128]))
    G_b1 = tf.Variable(tf.zeros(shape=[128]))
    
    G_W2 = tf.Variable(xavier_init([128, 784]))
    G_b2 = tf.Variable(tf.zeros(shape=[784]))
    theta_g = [G_W1, G_W2, G_b1, G_b2]
    
    # Build G, we have forced the variables to defined as above
    #G_h1 = tf.nn.relu(tf.matmul(z_node, G_W1) + G_b1) # <- old version
    G_h1 = build_layer(z_node, G_W1, G_b1, keep_prob = 0.8)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G = tf.nn.sigmoid(G_log_prob)    
    
    # Build D:
    keep_prob = .8
    BN = False
    if keep_prob < 1:
        mask1 = tf.nn.dropout(D_mask1, keep_prob)
        if BN: # Separate BN. TODO: still try joint
            D_pre1_real = BN_tensor(tf.matmul(x_node, D_W1) + D_b1, D_beta1, D_scale1)
            D_pre1_fake = BN_tensor(tf.matmul(G, D_W1) + D_b1, D_beta1, D_scale1)
            D_h1_real = tf.multiply(tf.nn.relu(D_pre1_real), mask1)
            D_h1_fake = tf.multiply(tf.nn.relu(D_pre1_fake), mask1)
        else:
            D_h1_real = tf.multiply(tf.nn.relu(tf.matmul(x_node, D_W1) + D_b1), mask1)
            D_h1_fake = tf.multiply(tf.nn.relu(tf.matmul(G, D_W1) + D_b1), mask1)
    
    else: # This is the old code if you're more comfortable with this
        D_h1_real = tf.nn.relu(tf.matmul(x_node, D_W1) + D_b1)
        D_h1_fake = tf.nn.relu(tf.matmul(G, D_W1) + D_b1)
        
    pre_D_real = tf.matmul(D_h1_real, D_W2) + D_b2
    D_real = tf.nn.sigmoid(pre_D_real)
                
    pre_D_fake = tf.matmul(D_h1_fake, D_W2) + D_b2
    D_fake = tf.nn.sigmoid(pre_D_fake)
    
    return G, D_real, D_fake, theta_g, theta_d, x_node, z_node, pre_D_real, pre_D_fake
	
	
def full_graph_conditional(NOISE_DIM, keep_prob_G = 0.8, keep_prob_D = 0.8,  y_in_last_layer = True):
	#IMPORTANT: Must     
    x_node = tf.placeholder(tf.float32, shape = [None,784])
    z_node = tf.placeholder(tf.float32, shape = [None,NOISE_DIM])
    y_node = tf.placeholder(tf.float32, shape = [None, 10])
    
    #D layer 1 for x-dimension
    D_W1 = tf.Variable(xavier_init([784, 128]))
    D_b1 = tf.Variable(tf.zeros(shape=[128]))
    D_mask1 = tf.Variable(tf.ones(shape = [1,128])) # For dropout
    #D layer 1 for y-dimension
    D_W1_y = tf.Variable(xavier_init([10, 128]))
    #D layer 2 (only for x-dimension)
    D_W2 = tf.Variable(xavier_init([128, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    D_W2_y = tf.Variable(xavier_init([10, 1]))
    #summarize the parameters in theta_d
    if y_in_last_layer:
        theta_d = [D_W1, D_W2, D_W2_y, D_b1, D_b2]
    else:
        theta_d = [D_W1, D_W1_y, D_W2, D_b1, D_b2]
        
    #G layer 1 for x-dimension
    G_W1 = tf.Variable(xavier_init([NOISE_DIM, 128]))
    G_b1 = tf.Variable(tf.zeros(shape=[128]))
    #G layer 1 for y-dimension
    G_W1_y = tf.Variable(xavier_init([10, 128]))
    #G layer 2 (only for x-dimension)
    G_W2 = tf.Variable(xavier_init([128, 784]))
    G_b2 = tf.Variable(tf.zeros(shape=[784]))
    #summarize the parameters in theta_g
    theta_g = [G_W1, G_W1_y, G_W2, G_b1, G_b2]
    
    #Build G
    if keep_prob_G < 1:
        G_h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(z_node, G_W1) + tf.matmul(y_node, G_W1_y) + G_b1), keep_prob_G)
    else: 
        G_h1 = tf.nn.relu(tf.matmul(z_node, G_W1) + tf.matmul(y_node, G_W1_y) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G = tf.nn.sigmoid(G_log_prob) # TODO: tanh
    
    #Build D in two parts
    if y_in_last_layer:
        preactive1_real = tf.matmul(x_node, D_W1) + D_b1
        preactive1_fake = tf.matmul(G, D_W1) + D_b1
    else: # first version
        preactive1_real = tf.matmul(x_node, D_W1) + tf.matmul(y_node, D_W1_y) + D_b1
        preactive1_fake = tf.matmul(G, D_W1) + tf.matmul(y_node, D_W1_y) + D_b1
        
    if keep_prob_D < 1:
        mask1 = tf.nn.dropout(D_mask1, keep_prob_D)
        D_h1_real = tf.multiply(tf.nn.relu(preactive1_real), mask1)
        D_h1_fake = tf.multiply(tf.nn.relu(preactive1_fake), mask1)
    else:
        D_h1_real = tf.nn.relu(preactive1_real)
        D_h1_fake = tf.nn.relu(preactive1_fake)
        
    if y_in_last_layer:
        pre_D_real = tf.matmul(D_h1_real, D_W2) + tf.matmul(y_node, D_W2_y) + D_b2
        pre_D_fake = tf.matmul(D_h1_fake, D_W2) + tf.matmul(y_node, D_W2_y) + D_b2
    else: # first version
        pre_D_real = tf.matmul(D_h1_real, D_W2) + D_b2
        pre_D_fake = tf.matmul(D_h1_fake, D_W2) + D_b2
    
    D_real = tf.nn.sigmoid(pre_D_real)
    D_fake = tf.nn.sigmoid(pre_D_fake)
    
    return G,D_real,D_fake,theta_g,theta_d,x_node,z_node, y_node, pre_D_real, pre_D_fake


#Standard implementation
def graph_objectives(D_real, D_fake):
	 #Both of these objectives need to be minimised (observe the minus sign in front)
    with tf.name_scope('loss_func'):
        obj_d= -tf.reduce_mean(tf.log(D_real)+tf.log(1-D_fake)) 
        tf.summary.scalar('d_loss', obj_d)
        obj_g= -tf.reduce_mean(tf.log(D_fake))
        tf.summary.scalar('g_loss', obj_g)
    return obj_d, obj_g

def graph_objectives_stable(pre_D_real, pre_D_fake):
    # This version is expected to be more stable.
    # The pre version means the link function for the last layer has not been applied yet.
    with tf.name_scope('loss_func'):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pre_D_real, tf.ones_like(pre_D_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pre_D_fake, tf.zeros_like(pre_D_fake)))
        D_loss = D_loss_real + D_loss_fake
        tf.summary.scalar('d_loss', D_loss)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pre_D_fake, tf.ones_like(pre_D_fake)))
        tf.summary.scalar('g_loss', G_loss)
    return D_loss, G_loss

def graph_optimizers(obj_d, obj_g, theta_d, theta_g):
	step = tf.Variable(0, trainable=False)
	with tf.name_scope('train'):
		opt_g = tf.train.AdamOptimizer().minimize(obj_g, var_list=theta_g)    
		opt_d = tf.train.AdamOptimizer().minimize(obj_d, var_list=theta_d)
	return opt_d, opt_g

##-----------------network-builder helperfunctions----------	
def build_layer(inputs, weights, biases, link_func = tf.nn.relu, keep_prob = 1, BN = False):
    # TODO: BN, dropout
    preactive = link_func(tf.matmul(inputs, weights) + biases)
    if (keep_prob < 1 and keep_prob > 0): 
        return tf.nn.dropout(preactive, keep_prob)
    else: 
        return preactive     

def BN_tensor(tensor, beta, scale):
    mean, variance = tf.nn.moments(tensor, [0])
    return tf.nn.batch_normalization(tensor, mean, variance, beta, scale, 1e-10)


#---f-gan---
#Aligned with f-GAN paper, custom activator to align with f^* domain space.  Assume output of discriminator is Real valued
#------------------GAN----------------------------
def g_f(preactivation):
	return -tf.log1p(tf.exp(tf.negative(preactivation))) 

def f_star(activated):
	return -tf.log(1-tf.exp(activated))

#------------KL---------------------------------------
def g_f_KL(preactivation):
	return preactivation 

def f_star_KL(activated):
	return tf.exp(activated-1)

#------------Reverse KL---------------------------------------
def g_f_revKL(preactivation):
	return -tf.exp(tf.negative(preactivation)) 

def f_star_revKL(activated):
	return -1-tf.log(tf.negative(activated))

#------------Pearson \chi^2---------------------------------------
def g_f_Pear(preactivation):
	return preactivation  

def f_star_Pear(activated):
	return tf.scalar_mul(0.25,tf.multiply(activated,activated)) + activated


def graph_objectives_2(D_real, D_fake):
	 #Both of these objectives need to be minimised (observe the minus sign in front)
    with tf.name_scope('loss_func'):
        obj_d= -tf.reduce_mean(g_f(D_real) - f_star(g_f(D_fake)))        			 
        tf.summary.scalar('d_loss', obj_d)
        obj_g= -tf.reduce_mean(D_fake)
        tf.summary.scalar('g_loss', obj_g)
    return obj_d, obj_g
	
	
