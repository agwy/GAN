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

  
  TRAIN_ITERS=1000 #Training iterations
  NOISE_Dim = 100 #Input noise dimension
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
    return preactivate,weights,biases, preactivate2
         
  #----------Models-Copied from blogpost in slack, should give atleast a reasonable result------------------------------------------
  
  #Generator NN: z -> (100,128) -> reLU -> (128,784) -> Sigmoid
  def Generator_NN(input, input_dim, output_dim):
  	hidden_1,w1,b1, _ = nn_layer(input, input_dim, 128, 'layer1_G')
  	hidden_2,w2,b2, _ = nn_layer(tf.nn.relu(hidden_1), 128, output_dim, 'layer2_G')
  	return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2]
  
  #Discrimiantor NN: x -> (784,128) -> reLU -> (128,1) -> sigmoid  
  def Discrim_NN(input_x, input_dim, output_dim):
  	hidden_1,w1,b1, _ = nn_layer(input_x, input_dim, 128, 'layer1_D')
  	hidden_2,w2,b2, _ = nn_layer(tf.nn.relu(hidden_1), 128, output_dim, 'layer2_D')
  	return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2]
  	
  def Discrim_NN_fixed(input_x, input_G, input_dim, output_dim):
    hidden_1,w1,b1, hidden_1_G = nn_layer(input_x, input_dim, 128, 'layer1_D', double_input = input_G)
    hidden_2,w2,b2, hidden_2_G = nn_layer(tf.nn.relu(hidden_1), 128, output_dim, 'layer2_D', double_input = hidden_1_G)
    return tf.nn.sigmoid(hidden_2),[w1,w2,b1,b2], tf.nn.sigmoid(hidden_2_G)
    
  
  	
  	
  #create and link placeholders to nextwork
  
  with tf.variable_scope('G'):
  	#Noisy input of dimension: NOISE_Dim
  	z_node = tf.placeholder(tf.float32, shape =  [None, NOISE_Dim]) #feed in batch size
  	G,theta_g = Generator_NN(z_node,NOISE_Dim,784)
  
  # OLD VERSION:
  #with tf.variable_scope('D') as scope:
  	#x_node = tf.placeholder(tf.float32, shape = [None,784]) #Real images
  	#D1,theta_d = Discrim_NN(x_node,784,1)
  	#with tf.name_scope('input_reshape'):
      #image_shaped_input = tf.reshape(x_node, [-1, 28, 28, 1])
      #tf.summary.image('input', image_shaped_input, 10)
      # TODO: keep track of fixed sampled units
      # TODO: does it display the first or last 10?  	
  	#Make copy of D that uses same variables but has G as input
  	#scope.reuse_variables()
  	#D2,theta_d = Discrim_NN(G,784,1)
  	
  # FIX?
  with tf.variable_scope('D') as scope:
    x_node = tf.placeholder(tf.float32, shape = [None,784]) #Real images
    D1, theta_d, D2 = Discrim_NN_fixed(x_node,G,784,1)
  	
  
  #Both of these objectives need to be minimised (observe the minus sign in front)
  with tf.name_scope('loss_func'):
      obj_d= -tf.reduce_mean(tf.log(D1)+tf.log(1-D2)) 
      tf.summary.scalar('d_loss', obj_d)
      obj_g= -tf.reduce_mean(tf.log(D2))
      tf.summary.scalar('g_loss', obj_g)
  
  with tf.name_scope('train'):
      opt_d = tf.train.AdamOptimizer().minimize(obj_d,var_list=theta_d)
      opt_g = tf.train.AdamOptimizer().minimize(obj_g,var_list=theta_g) 
        
  saver = tf.train.Saver() #For saving the fitted model TODO: how to restart a session from a saved metadata
  
  #Initalise variables and start session
  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()  
  sess.run(init)
  
   # Merge all summaries and create files to write out to:
  merged_summ = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  
  #Number of repeated training steps for the discrimiantor
  k=1
  image_count = mnist.train.images.shape[1]
  
  #Storage for objective function values
  histd, histg= np.zeros((TRAIN_ITERS)), np.zeros((TRAIN_ITERS))
  hist_pred_noise, hist_pred_data = np.zeros((TRAIN_ITERS)), np.zeros((TRAIN_ITERS))
  
  #Start trainning
  for i in range(TRAIN_ITERS):
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata() TODO: report metadata every 200 steps
  	for j in range(k):
  		x = mnist.train.images[np.random.choice(image_count,M),:] #Select a mini batch 
  		z = sample_Z(M,NOISE_Dim) #Noisy examples
  		summary, histd[i],_= sess.run([merged_summ, obj_d,opt_d], {x_node: x, z_node: z}) #update parameters in direction of gradient
  		train_writer.add_summary(summary, (k+1)*i + j)
  		
  	z = sample_Z(M,NOISE_Dim)
  	summary, histg[i], _ = sess.run([merged_summ, obj_g,opt_g], {x_node: np.zeros((1,784)), z_node: z}) # update generator#
  	train_writer.add_summary(summary, (k+1) * i + k) # TODO: Check if this does not cause a clash with previous add_summary
  	
  	hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
  	hist_pred_noise[i] = np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_Dim)} ))
  		
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
  np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_Dim)} ))
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
  # TODO: need to find regions of high probability mass to generate sensible figures (interpolation pherhaps?)
  
  #Save the fitted model
  saver.save(sess, 'my-model')

def variable_summaries(var, extended = False):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        if extended:
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.histogram('histogram', var)
  
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
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
