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


def train():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

    TRAIN_ITERS=1000 #Training iterations
    NOISE_DIM = 100 #Input noise dimension
    NUM_DIAGN = 500 # Number of diagnostics to compute
    DIAGN_STEP = TRAIN_ITERS / NUM_DIAGN
    M=128 #Minibatch sizes
    K=1 #Number of repeated training steps for the discrimiantor

    #create and link placeholders to nextwork
  
    with tf.variable_scope('G'):
        #Noisy input of dimension: Dim
        z_node = tf.placeholder(tf.float32, shape =  [None, NOISE_DIM]) #feed in batch size
        G, theta_g = Generator_NN(z_node,NOISE_DIM,784)
  
  # OLD VERSION:
  #with tf.variable_scope('D') as scope:
  	#x_node = tf.placeholder(tf.float32, shape = [None,784]) #Real images
  	#D1,theta_d = Discrim_NN(x_node,784,1)
  	#with tf.name_scope('input_reshape'):
	
  	#Make copy of D that uses same variables but has G as input
  	#scope.reuse_variables()
  	#D2,theta_d = Discrim_NN(G,784,1)
  	
    # FIX?
    with tf.variable_scope('D') as scope:
        x_node = tf.placeholder(tf.float32, shape = [None,784]) #Real images
        D1, theta_d, D2 = Discrim_NN_fixed(x_node, G, 784, 1)
        #image_shaped_input = tf.reshape(x_node, [-1, 28, 28, 1])
        #tf.summary.image('input', image_shaped_input, 10)
        # TODO: keep track of fixed sampled units
        # TODO: does it display the first or last 10?  
  	
  
  #Both of these objectives need to be minimised (observe the minus sign in front)
    with tf.name_scope('loss_func'):
        obj_d= -tf.reduce_mean(tf.log(D1)+tf.log(1-D2)) 
        tf.summary.scalar('d_loss', obj_d)
        obj_g= -tf.reduce_mean(tf.log(D2))
        tf.summary.scalar('g_loss', obj_g)
  
    with tf.name_scope('train'):
        opt_d = tf.train.AdamOptimizer().minimize(obj_d, var_list=theta_d)
        opt_g = tf.train.AdamOptimizer().minimize(obj_g, var_list=theta_g) 
        
    saver = tf.train.Saver() #For saving the fitted model TODO: how to restart a session from a saved metadata
  
    #Initalise variables and start session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()  
    sess.run(init)
  
    # Merge all summaries and create files to write out to:
    merged_summ = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    
    image_count = mnist.train.images.shape[1]
  
    #Storage for objective function values
    histd, histg= np.zeros((TRAIN_ITERS)), np.zeros((TRAIN_ITERS))
    hist_pred_noise, hist_pred_data = np.zeros((TRAIN_ITERS)), np.zeros((TRAIN_ITERS))
  
    #Start trainning
    for i in range(TRAIN_ITERS):
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata() TODO: report metadata
        if i % DIAGN_STEP == 0:   # Compute summaries
            for j in range(K):
      	         x = mnist.train.images[np.random.choice(image_count,M),:] #Select a mini batch 
      	         z = sample_Z(M,NOISE_DIM) #Noisy examples
      	         summary, histd[i],_= sess.run([merged_summ, obj_d,opt_d], {x_node: x, z_node: z}) #update parameters in direction of gradient
      	         train_writer.add_summary(summary, (K+1)*i + j)
      		
            z = sample_Z(M,NOISE_DIM)
            summary, histg[i], _ = sess.run([merged_summ, obj_g,opt_g], {x_node: np.zeros((1,784)), z_node: z}) # update generator#
            train_writer.add_summary(summary, (K+1) * i + K) # TODO: Check if this does not cause a clash with previous add_summary
      	
            hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
            hist_pred_noise[i] = np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_DIM)} ))
        else:
            for j in range(K):
      	         x = mnist.train.images[np.random.choice(image_count, M),:] #Select a mini batch 
      	         z = sample_Z(M, NOISE_DIM) #Noisy examples
      	         histd[i],_= sess.run([obj_d,opt_d], {x_node: x, z_node: z}) #update parameters in direction of gradient
      		
            z = sample_Z(M,NOISE_DIM)
            histg[i], _ = sess.run([obj_g,opt_g], {x_node: np.zeros((1,784)), z_node: z}) # update generator#
      	
            hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
            hist_pred_noise[i] = np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_DIM)} ))
  		
         #Print some information to see whats happening
        if i % (TRAIN_ITERS // 10) == 0:		  	
             print("Iteration: ",float(i)/float(TRAIN_ITERS))
             print("G objective (Need to maximise):",histg[i])
             print("D objective (Need to minimise):",histd[i])
             print("Average 100 Data into D1:",
             hist_pred_data[i]
             )
             print("avg 100 Noise into D1:",
             hist_pred_noise[i]
             )
	    	
    train_writer.close()
    test_writer.close()
  
    #Check performance of 100 noise samples into Discriminator
    print("avg 100 Noise into D1:",
    np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_DIM)} ))
    )

    #Check performance of 100 data inputs into discriminator
    print("Average 100 Data into D1:",
    np.mean(sess.run([D1],{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
    )


    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(TRAIN_ITERS), hist_pred_data , 'b-')

    plt.subplot(212)
    plt.plot(range(TRAIN_ITERS), hist_pred_noise , 'b-')
    plt.savefig("DATA_NOISE.png",bbox_inches="tight")

    #----------------------Generate samples and plot, save to "pretty_pictures.png" --------------------------------
    samples = sess.run(G, feed_dict={z_node: sample_Z(16, NOISE_DIM)})
    fig = plot(samples)
    plt.savefig('pretty_plot.png', bbox_inches='tight')
    # TODO: need to find regions of high probability mass to generate sensible figures (interpolation pherhaps?)

    #Save the fitted model
    saver.save(sess, 'my-model')
  
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()
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


