from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image

import argparse
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from SummaryFunctions import *
from NNBuilder import *

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None

#NOTE: Right now, this is rather limited. One could extend this function to take in
#		 an array specifying the size of a number of layers in the NN, and another one
#		 specifying which layers take which function (sigmoid, logistic, ...)
#		 same holds for the next constructor functions below
#
def graph_G_constructor(NOISE_DIM):
    with tf.variable_scope('G'):
        #z_node: The input end of the G graph receiving random noise of size
        #		  NOISE_DIM in float type, where G is built by Generator_NN
        z_node = tf.placeholder(tf.float32, shape = [None, NOISE_DIM]) 
        G, theta_g = Generator_NN(z_node,NOISE_DIM,784)
	return G, theta_g, z_node
	
def graph_D_constructor(G):
    with tf.variable_scope('D') as scope:
        x_node = tf.placeholder(tf.float32, shape = [None,784]) #Real images
        D1, theta_d, D2 = Discrim_NN_fixed(x_node, G, 784, 1)
        #image_shaped_input = tf.reshape(x_node, [-1, 28, 28, 1])
        #tf.summary.image('input', image_shaped_input, 10)
        # TODO: keep track of fixed sampled units
        # TODO: does it display the first or last 10?  
	return D1, D2, theta_d, x_node
	
def graph_objectives(D1, D2):
	 #Both of these objectives need to be minimised (observe the minus sign in front)
    with tf.name_scope('loss_func'):
        obj_d= -tf.reduce_mean(tf.log(D1)+tf.log(1-D2)) 
        tf.summary.scalar('d_loss', obj_d)
        obj_g= -tf.reduce_mean(tf.log(D2))
        tf.summary.scalar('g_loss', obj_g)
    return obj_d, obj_g

def graph_optimizers(obj_d, obj_g, theta_d, theta_g):
    with tf.name_scope('train'):
        opt_d = tf.train.AdamOptimizer(0.0001).minimize(obj_d, var_list=theta_d)
        opt_g = tf.train.AdamOptimizer(0.0001).minimize(obj_g, var_list=theta_g)
    return opt_d, opt_g

#NOTE: Trains the NN that was set up using the constructor functions from above
#		 Modifications to the above constructors should be checked for validity by
#		 running the optimization routine on them
#
def train_NN(TRAIN_ITERS, DIAGN_STEP, NOISE_DIM, M, K_G,K_D, image_count, sess, mnist, D1, D2, x_node, z_node, obj_d, obj_g, opt_d, opt_g, merged_summ, train_writer):
	
	 #Storage for objective function values
    histd, histg= np.zeros((TRAIN_ITERS*K_D)), np.zeros((TRAIN_ITERS*K_G))
    hist_pred_noise, hist_pred_data = np.zeros((TRAIN_ITERS)), np.zeros((TRAIN_ITERS))
    
    #Start trainning
    for i in range(TRAIN_ITERS):
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata() TODO: report metadata
        if i % DIAGN_STEP == 0:
        	# Compute summaries
        	#Train discriminator K_D times
        	for j in range(K_D):
        		x = mnist.train.images[np.random.choice(image_count,M),:] #Select a mini batch 
        		z = sample_Z(M,NOISE_DIM) #Noisy examples
        		summary, histd[i*K_D + j],_= sess.run([merged_summ, obj_d,opt_d], {x_node: x, z_node: z}) #update parameters in direction of gradient
        		train_writer.add_summary(summary, (K_D + K_G)*i + j)
        	
        	for j in range(K_G):
        		z = sample_Z(M, NOISE_DIM)
        		summary, histg[i*K_G + j], _ = sess.run([merged_summ, obj_g,opt_g], {x_node: np.zeros((1,784)), z_node: z}) # update generator#
        		train_writer.add_summary(summary, (K_D + K_G)*i + K_D + j) # TODO: Check if this does not cause a clash with previous add_summary
        	
        	hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[np.random.choice(image_count,100),:], z_node : np.zeros((0,NOISE_DIM))} )) # empty input
        	hist_pred_noise[i] = np.mean(sess.run([D2],{x_node : np.zeros((0,784)), z_node: sample_Z(100,NOISE_DIM)} )) # empty input
        else:
        	for j in range(K_D):
        		x = mnist.train.images[np.random.choice(image_count,M),:] #Select a mini batch 
        		z = sample_Z(M,NOISE_DIM) #Noisy examples
        		histd[i*K_D + j],_= sess.run([obj_d,opt_d], {x_node: x, z_node: z}) #update parameters in direction of gradient
        	
        	#Train generator K_G times
        	for j in range(K_G):
        		z = sample_Z(M,NOISE_DIM)
        		histg[i*K_G + j], _ = sess.run([obj_g,opt_g], {x_node: np.zeros((1,784)), z_node: z}) # update generator # TODO
        	
        	hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[np.random.choice(image_count,100),:], z_node : np.zeros((3,NOISE_DIM))} )) # TODO
        	hist_pred_noise[i] = np.mean(sess.run([D2],{x_node: np.zeros((0,784)), z_node: sample_Z(100,NOISE_DIM)} )) # empty input
        
        #Print some information to see whats happening
        if i % (TRAIN_ITERS // 10) == 0:
        	print("Iteration: ",float(i)/float(TRAIN_ITERS))
        	print("G objective (Need to maximise):",histg[i])
        	print("D objective (Need to minimise):",histd[i])
        	print("Average 100 Data into D1:",hist_pred_data[i])
        	print("avg 100 Noise into D1:",hist_pred_noise[i])
    
    return hist_pred_noise, hist_pred_data,histd, histg


#This function puts everything together 
def GAN():
	
    #STEP 0: IMPORT DATA, DEFINE ALGORITHM SPECIFICS
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)
    image_count = mnist.train.images.shape[1]

    TRAIN_ITERS=1000 #Training iterations
    NOISE_DIM = 200 #Input noise dimension
    NUM_DIAGN = 100 # Number of diagnostics to compute
    DIAGN_STEP = TRAIN_ITERS / NUM_DIAGN
    M=127 #Minibatch sizes
    K_G = 1 #Number of Generator steps for each TRAIN_ITERS
    K_D = 1 #Number of Discriminator steps for each TRAIN_ITERS
    
    #STEP 1: BUILD GRAPH
    G, theta_g, z_node = graph_G_constructor(NOISE_DIM)	#primitive function! Might need more inputs eventually
    D1, D2, theta_d, x_node = graph_D_constructor(G)		#same here
    obj_d, obj_g = graph_objectives(D1, D2)					#objectives for D and G
    opt_d, opt_g = graph_optimizers(obj_d, obj_g, theta_d, theta_g)	#optimizers for D and G 
    saver = tf.train.Saver() #For saving the fitted model TODO: how to restart a session from a saved metadata
  
    #STEP 2: INITIALIZE VARIABLES & START SESSION 
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()  
    sess.run(init)
  
    #STEP 3: MERGE SUMMARIES, CREATE LOG FILES Merge all summaries and create 
    merged_summ = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph) #files to write out to
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')					 #files to write out to
    
    #STEP 4: TRAIN NETWORK
    hist_pred_noise , hist_pred_data,histd, histg = train_NN(TRAIN_ITERS, DIAGN_STEP, NOISE_DIM, M, K_G,K_D, image_count, sess, mnist, D1, D2, x_node, z_node, obj_d, obj_g, opt_d, opt_g, merged_summ, train_writer)
    
	 
	 #STEP 5: POST-PROCESSING
	 #close the summary writers that added to the log files	
    train_writer.close()
    test_writer.close()
    
    #Save some pictures of the noise and the generated pictures
    data_noise_png(D1, D2, x_node, z_node, image_count, hist_pred_data, hist_pred_noise, mnist, sess, TRAIN_ITERS, NOISE_DIM)
    pretty_plot(G, z_node, sess, NOISE_DIM)
    Loss_function_png(histd, histg)
    
    #Save the fitted model
    saver.save(sess, 'my-model')
    
    
  
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    GAN()
    #FEW if any of these flags are used....
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                  default=False,
                  help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                  help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                  help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                  help='Keep probability for training dropout.')
    parser.add_argument('--diagn', type=bool, default=True,
                  help='Boolean indicating wheter to output diagnostics.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                  help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='logs/mnist_with_summaries',
                  help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
def test_code_bn():
    test1 = tf.placeholder(tf.float32, shape = [None, 2])
    test2 = tf.placeholder(tf.float32, shape = [None, 2])
    input_shape = tf.shape(test1)
    input_shape2 = tf.shape(test2)
    inputs = tf.concat(0,[test1, test2])
    mean, variance = tf.nn.moments(inputs, [0])
    norm_inputs = (inputs - mean) / tf.sqrt(tf.maximum(variance, 1e-12))
    #norm_inputs = tf.nn.fused_batch_norm(inputs, 1.0, 0.0)
    norm_input1 = tf.slice(norm_inputs, [0, 0], input_shape)
    norm_input2 = tf.slice(norm_inputs, [input_shape[0], 0], input_shape2) # TODO: check if this yields the correct output input_shape is the right beginning # TODO: check normalization and 
    print("test output: ", sess.run([variance, inputs, norm_inputs, norm_input1], {test1 : np.zeros((3,2)), test2 : np.array([[1e-10,1]])} ))


