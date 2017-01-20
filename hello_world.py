from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image

import argparse
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)


  TRAIN_ITERS=100
  
  M=200 # minibatch size

  

  def sigmoid(x):
  	return 1 / (1 + np.exp(-x))
  	
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def nn_layer(input_tensor, input_dim, output_dim):
    weights = weight_variable([input_dim, output_dim])
    biases = bias_variable([output_dim])
    preactivate = tf.matmul(input_tensor, weights) + biases
    activations = preactivate
    return activations,weights,biases
         
      
  def simple_NN(input, input_dim, output_dim):
  	hidden_1,w1,b1 = nn_layer(input, input_dim,750)
  	hidden_2,w2,b2 = nn_layer(tf.nn.relu(hidden_1),750,output_dim)
  	return tf.nn.tanh(hidden_2),[w1,w2,b1,b2]
  	
  
  	
  	
  # Input placeholders
  with tf.variable_scope('G'):
  	#Noisy input  	
  	z_node = tf.placeholder(tf.float32, shape =  [None, 784]) #feed in batch size
  	G,theta_g = simple_NN(z_node,784,784)
   
  with tf.variable_scope('D') as scope:
  	#Will be Real images
  	x_node = tf.placeholder(tf.float32, shape = [None,784])
  	
  	#Takes input image and returns a value between 0 and 1
  	fc,theta_d = simple_NN(x_node,784,1)
  	D1 = tf.maximum(tf.minimum(fc,.99),0.01)
  	
  	#Make copy of D that uses same variables but has G as input
  	scope.reuse_variables()
  	fc,theta_d = simple_NN(G,784,1)
  	D2 = 	tf.maximum(tf.minimum(fc,.99),0.01)
  	
  		
  obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2)) #need to be minimized for D
  obj_g=tf.reduce_mean(tf.log(D2)) #Maximised for G - fooling D1!
  
  opt_g = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(-obj_g,var_list=theta_g) #Multiple objective by -1 to maximise!
  opt_d = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(obj_d,var_list=theta_d)
  
  
  saver = tf.train.Saver()
  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()  
  sess.run(init)
      
  k=1
  image_count = mnist.train.images.shape[1]
  
  #Storage for objective function values
  histd, histg= np.zeros((TRAIN_ITERS)), np.zeros((TRAIN_ITERS))
  
  
  for i in range(TRAIN_ITERS):
  	for j in range(k):
  		x = mnist.train.images[np.random.choice(image_count,M),:] #Select a mini batch 
  		z = np.random.uniform(0.0,1.0,(M,784)) #Noisy examples
  		histd[i],_= sess.run([obj_d,opt_d], {x_node: x, z_node: z})
  		
  	z = np.random.uniform(0.0,1.0,(M,784)) #Noisy sample
  	histg[i],_ = sess.run([obj_g,opt_g], {z_node: z}) # update generator#
  	
	if i % (TRAIN_ITERS // 10) == 0:		  	
		print("Iteration: ",float(i)/float(TRAIN_ITERS))
		print("G objective (Need to maximise):",histg[i])
		print("D objective (Need to minimise):",histd[i])
		
  
  
  #Feed noise into D and see what comes out:
  #Should be a probablity between 0-1 on its whether it believes the image is noise or not
  print("Noise into D1:",
  sess.run([D1],{x_node: np.random.uniform(0.0,1.0,(1,784))} )
  )
  
  #Feeding noise into G to see what comes out ! 
  my_array = np.array(sess.run([G],{z_node: np.random.uniform(0.0,1.0,(1,784))})).reshape(28,28)
  plt.imshow(my_array,interpolation='nearest')
  plt.gray()
  plt.show()

  
  saver.save(sess, 'my-model')
  
def main(_):
  train()


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