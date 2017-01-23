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
from NNTrainer import *

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None

#This function puts everything together 
def GAN(conditionalBool):
	
    #STEP 0: IMPORT DATA, DEFINE ALGORITHM SPECIFICS
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)                      
    image_count = mnist.train.images.shape[1]

    TRAIN_ITERS= 100 #Training iterations
    NOISE_DIM = 100 #Input noise dimension
    NUM_DIAGN = 10 # Number of diagnostics to compute
    DIAGN_STEP = TRAIN_ITERS / NUM_DIAGN
    M= 128 #Minibatch sizes
    K_G = 1 #Number of Generator steps for each TRAIN_ITERS
    K_D = 1 #Number of Discriminator steps for each TRAIN_ITERS
    
    
    #STEP 1: CREATE NETWORK
    if conditionalBool:
        G,D_real,D_fake,theta_g,theta_d,x_node,z_node,y_node = full_graph_conditional(NOISE_DIM)
    else:
        G,D_real,D_fake,theta_g,theta_d,x_node,z_node = full_graph(NOISE_DIM)
        y_node=None        
    obj_d, obj_g = graph_objectives(D_real, D_fake)
    opt_d, opt_g = graph_optimizers(obj_d, obj_g, theta_d, theta_g)
    
    #STEP 2: RUN SESSION
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
    if conditionalBool:
        hist_pred_noise , hist_pred_data, histd, histg = train_NN_Cond(TRAIN_ITERS, DIAGN_STEP, NOISE_DIM, M, K_G,K_D, image_count, sess, mnist, 
    																			D_real, D_fake,G, x_node, z_node, y_node,obj_d, obj_g, opt_d, opt_g, merged_summ, train_writer)
    else:
        hist_pred_noise , hist_pred_data, histd, histg = train_NN(TRAIN_ITERS, DIAGN_STEP, NOISE_DIM, M, K_G,K_D, image_count, sess, mnist, 
    																			D_real, D_fake,G, x_node, z_node, obj_d, obj_g, opt_d, opt_g, merged_summ, train_writer)


	 
	 #STEP 5: POST-PROCESSING
	 #close the summary writers that added to the log files	
    train_writer.close()
    test_writer.close()
    
    #Save some pictures of the noise and the generated pictures
    data_noise_png(D_real, D_fake, x_node, z_node, image_count, hist_pred_data, hist_pred_noise, mnist, sess, TRAIN_ITERS, NOISE_DIM, y_node)
    pretty_plot(G, z_node, sess, NOISE_DIM, 0, 0, y_node, mnist)
    makeAnimatedGif()
    Loss_function_png(histd, histg)
    
    #Save the fitted model
    saver.save(sess, 'my-model')
    
    
  
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    GAN(True)
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


