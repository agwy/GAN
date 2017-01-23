from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image,ImageDraw

import argparse
import sys 
import os


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from SummaryFunctions import *
from NNBuilder import *

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None



#NOTE: Trains the NN that was set up using the constructor functions from above
#		 Modifications to the above constructors should be checked for validity by
#		 running the optimization routine on them
#
def train_NN(TRAIN_ITERS, DIAGN_STEP, NOISE_DIM, M, K_G,K_D, image_count, sess, mnist, D1, D2,G, x_node, z_node, obj_d, obj_g, opt_d, opt_g, merged_summ, train_writer):
	
	 #Storage for objective function values
    histd, histg= np.zeros((TRAIN_ITERS*K_D)), np.zeros((TRAIN_ITERS*K_G))
    hist_pred_noise, hist_pred_data = np.zeros((TRAIN_ITERS)), np.zeros((TRAIN_ITERS))
    picture_count = 0
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
        		z = sample_Z(M,NOISE_DIM)
        		summary, histg[i*K_G + j], _ = sess.run([merged_summ, obj_g,opt_g], {x_node: np.zeros((1,784)),z_node: z}) # update generator#
        		train_writer.add_summary(summary, (K_D + K_G)*i + K_D + j) # TODO: Check if this does not cause a clash with previous add_summary
        	
        	hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
        	hist_pred_noise[i] = np.mean(sess.run(D2,{x_node: np.zeros((1,784)),z_node: sample_Z(100,NOISE_DIM)} ))
        else:
        	for j in range(K_D):
        		x = mnist.train.images[np.random.choice(image_count,M),:] #Select a mini batch 
        		z = sample_Z(M,NOISE_DIM) #Noisy examples
        		histd[i*K_D + j],_= sess.run([obj_d,opt_d], {x_node: x, z_node: z}) #update parameters in direction of gradient
        	
        	#Train generator K_G times
        	for j in range(K_G):
        		z = sample_Z(M,NOISE_DIM)
        		histg[i*K_G + j], _ = sess.run([obj_g,opt_g], {x_node: np.zeros((1,784)),z_node: z}) # update generator
        	
        	hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
        	hist_pred_noise[i] = np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_DIM)} ))
        
        #Print some information to see whats happening
        if i % (TRAIN_ITERS // 10) == 0:
        	print("Iteration: ",float(i)/float(TRAIN_ITERS))
        	print("G objective (Need to minimise):",histg[i])
        	print("D objective (Need to minimise):",histd[i])
        	print("Average 100 Data into D1:",hist_pred_data[i])
        	print("avg 100 Noise into D1:",hist_pred_noise[i])
        	#Save a plot image
        	pretty_plot(G, z_node, sess, NOISE_DIM,picture_count,i)
        	picture_count += 1
        	
    
    return hist_pred_noise, hist_pred_data,histd, histg


#This function puts everything together 
def GAN():
	
    #STEP 0: IMPORT DATA, DEFINE ALGORITHM SPECIFICS
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)
    image_count = mnist.train.images.shape[1]

    TRAIN_ITERS= 100 #Training iterations
    NOISE_DIM = 100 #Input noise dimension
    NUM_DIAGN = 500 # Number of diagnostics to compute
    DIAGN_STEP = TRAIN_ITERS / NUM_DIAGN
    M= 128 #Minibatch sizes
    K_G = 1 #Number of Generator steps for each TRAIN_ITERS
    K_D = 1 #Number of Discriminator steps for each TRAIN_ITERS
    
    G,D_real,D_fake,theta_g,theta_d,x_node,z_node = full_graph(NOISE_DIM)
    obj_d, obj_g = graph_objectives(D_real, D_fake)
    opt_d, opt_g = graph_optimizers(obj_d, obj_g, theta_d, theta_g)
    
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()  
    sess.run(init)
    saver = tf.train.Saver()
    
    #STEP 3: MERGE SUMMARIES, CREATE LOG FILES Merge all summaries and create 
    merged_summ = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph) #files to write out to
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')					 #files to write out to
    
    if not os.path.exists('pictures/'):
    	os.makedirs('pictures/')
    	    
    #STEP 4: TRAIN NETWORK
    hist_pred_noise , hist_pred_data,histd, histg = train_NN(TRAIN_ITERS, DIAGN_STEP, NOISE_DIM, M, K_G,K_D, image_count, sess, mnist, 
    																			D_real, D_fake,G, x_node, z_node, obj_d, obj_g, opt_d, opt_g, merged_summ, train_writer)

	 
	 #STEP 5: POST-PROCESSING
	 #close the summary writers that added to the log files	
    train_writer.close()
    test_writer.close()
    
    #Save some pictures of the noise and the generated pictures
    data_noise_png(D_real, D_fake, x_node, z_node, image_count, hist_pred_data, hist_pred_noise, mnist, sess, TRAIN_ITERS, NOISE_DIM)
    #pretty_plot(G, z_node, sess, NOISE_DIM)
    makeAnimatedGif()
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


