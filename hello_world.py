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

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  
  TRAIN_ITERS=100000 #Training iterations
  NOISE_Dim = 100 #Input nosie dimension
  M=128 #Minibatch sizes

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

    
  #Basic NN layer without activation, combine these 
  def nn_layer(input_tensor, input_dim, output_dim):
    weights = tf.Variable(xavier_init([input_dim, output_dim]))
    biases = bias_variable([output_dim])
    preactivate = tf.matmul(input_tensor, weights) + biases
    return preactivate,weights,biases
         
  #----------Models-Copied from blogpost in slack, should give atleast a reasonable result------------------------------------------
  
  #Generator NN: z -> (100,128) -> reLU -> (128,784) -> Sigmoid
  def Generator_NN(input, input_dim, output_dim):
  	hidden_1,w1,b1 = nn_layer(input, input_dim,128)
  	hidden_2,w2,b2 = nn_layer(tf.nn.relu(hidden_1),128,output_dim)
  	return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2]
  
  #Discrimiantor NN: x -> (784,128) -> reLU -> (128,1) -> sigmoid  
  def Discrim_NN(input, input_dim, output_dim):
  	hidden_1,w1,b1 = nn_layer(input, input_dim,128)
  	hidden_2,w2,b2 = nn_layer(tf.nn.relu(hidden_1),128,output_dim)
  	return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2]
  
  	
  	
  #create and link placeholders to nextwork
  
  with tf.variable_scope('G'):
  	#Noisy input of dimension: NOISE_Dim
  	z_node = tf.placeholder(tf.float32, shape =  [None, NOISE_Dim]) #feed in batch size
  	G,theta_g = Generator_NN(z_node,NOISE_Dim,784)
  
  with tf.variable_scope('D') as scope:
  	x_node = tf.placeholder(tf.float32, shape = [None,784]) #Real images
  	D1,theta_d = Discrim_NN(x_node,784,1)
  	
  	#Make copy of D that uses same variables but has G as input
  	scope.reuse_variables()
  	D2,theta_d = Discrim_NN(G,784,1)
  	
  
  #Both of these objectives need to be minimised (observe the minus sign in front)
  obj_d= -tf.reduce_mean(tf.log(D1)+tf.log(1-D2)) 
  obj_g= -tf.reduce_mean(tf.log(D2)) 
  
  opt_d = tf.train.AdamOptimizer().minimize(obj_d,var_list=theta_d)
  opt_g = tf.train.AdamOptimizer().minimize(obj_g,var_list=theta_g) 
  
  
  saver = tf.train.Saver() #For saving the fitted model TODO: how to restart a session from a saved metadata
  
  #Initalise variables and start session
  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()  
  sess.run(init)
  
  #Number of repeated training steps for the discrimiantor
  k=1
  image_count = mnist.train.images.shape[1]
  
  #Storage for objective function values
  histd, histg= np.zeros((TRAIN_ITERS)), np.zeros((TRAIN_ITERS))
  
  #Start trainning
  for i in range(TRAIN_ITERS):
  	for j in range(k):
  		x = mnist.train.images[np.random.choice(image_count,M),:] #Select a mini batch 
  		z = sample_Z(M,NOISE_Dim) #Noisy examples
  		histd[i],_= sess.run([obj_d,opt_d], {x_node: x, z_node: z}) #update parameters in direction of gradient
  		
  	z = sample_Z(M,NOISE_Dim)
  	histg[i],_ = sess.run([obj_g,opt_g], {z_node: z}) # update generator#
  	
  	#Print some information to see whats happening
	if i % (TRAIN_ITERS // 10) == 0:		  	
		print("Iteration: ",float(i)/float(TRAIN_ITERS))
		print("G objective (Need to maximise):",histg[i])
		print("D objective (Need to minimise):",histd[i])
		print("Average 100 Data into D1:",
		np.mean(sess.run([D1],{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
		)
		print("avg 100 Noise into D1:",
		np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_Dim)} ))
		)
  
  
  #Check performance of 100 noise samples into Discriminator
  print("avg 100 Noise into D1:",
  np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_Dim)} ))
  )
  
  #Check performance of 100 data inputs into discriminator
  print("Average 100 Data into D1:",
  np.mean(sess.run([D1],{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
  )
  
  #Make some picture of G output
  
  #-----------------Plot function for grid of pictures-----------------------------------
  def plot(samples):
  	fig = plt.figure(figsize=(4, 4))
  	gs = gridspec.GridSpec(4, 4)
  	gs.update(wspace=0.05, hspace=0.05)
  	
  	for i, sample in enumerate(samples):
  		ax = plt.subplot(gs[i])
  		plt.axis('off')
  		ax.set_xticklabels([])
  		ax.set_yticklabels([])
  		ax.set_aspect('equal')
  		plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
  	return fig  
     	
  
  #----------------------Generate samples and plot, save to "pretty_pictures.png" --------------------------------
  samples = sess.run(G, feed_dict={z_node: sample_Z(16, NOISE_Dim)})
  fig = plot(samples)
  plt.savefig('pretty_plot.png', bbox_inches='tight')
  
  #Save the fitted model
  saver.save(sess, 'my-model')
  
def main(_):
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
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)