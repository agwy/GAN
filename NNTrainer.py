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
def train_NN_Cond(TRAIN_ITERS, DIAGN_STEP, NOISE_DIM, M, K_G,K_D, image_count, sess, mnist, D1, D2,G, x_node, z_node, y_node, obj_d, obj_g, opt_d, opt_g, merged_summ, train_writer, filedir=""):
	
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
        		
        		random_batch_indices = np.random.choice(image_count,M) #Select a mini batch 
        		x = mnist.train.images[random_batch_indices,:] 			 #get the images
        		y = mnist.train.labels[random_batch_indices,:] 		    #get the labels
        		z = sample_Z(M,NOISE_DIM) 										 #Noisy examples
        		
        		summary, histd[i*K_D + j],_= sess.run([merged_summ, obj_d,opt_d], {x_node: x, z_node: z, y_node: y}) #update parameters in direction of gradient
        		train_writer.add_summary(summary, (K_D + K_G)*i + j)
        	
        	for j in range(K_G):
        		z = sample_Z(M,NOISE_DIM)
        		y = mnist.train.labels[np.random.choice(10,M, replace=True),:] 		#sample from MNIST labels randomly
        		summary, histg[i*K_G + j], _ = sess.run([merged_summ, obj_g,opt_g], {x_node: np.zeros((1,784)),z_node: z, y_node: y}) # update generator#
        		train_writer.add_summary(summary, (K_D + K_G)*i + K_D + j) # TODO: Check if this does not cause a clash with previous add_summary
        	
        	random_batch_indices = np.random.choice(image_count,100) #Select a mini batch 
        	hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[random_batch_indices,:], y_node: mnist.train.labels[random_batch_indices,:]} ))
        	hist_pred_noise[i] = np.mean(sess.run(D2,{x_node: np.zeros((1,784)),z_node: sample_Z(100,NOISE_DIM), y_node: mnist.train.labels[random_batch_indices,:]} ))
        else:
        	for j in range(K_D):
        		
        		random_batch_indices = np.random.choice(image_count,M) #Select a mini batch 
        		x = mnist.train.images[random_batch_indices,:] 			 #get the images
        		y = mnist.train.labels[random_batch_indices,:] 		    #get the labels
        		z = sample_Z(M,NOISE_DIM) 										 #Noisy examples
        		
        		histd[i*K_D + j],_= sess.run([obj_d,opt_d], {x_node: x, z_node: z, y_node: y}) #update parameters in direction of gradient
        	
        	#Train generator K_G times
        	for j in range(K_G):
        		z = sample_Z(M,NOISE_DIM)
        		y = mnist.train.labels[np.random.choice(10,M, replace=True),:] 		#sample from MNIST labels randomly
        		histg[i*K_G + j], _ = sess.run([obj_g,opt_g], {x_node: np.zeros((1,784)),z_node: z, y_node: y}) # update generator
        		
        	random_batch_indices = np.random.choice(image_count,100) #Select a mini batch 
        	hist_pred_data[i] = np.mean(sess.run(D1,{x_node: mnist.train.images[random_batch_indices,:], y_node: mnist.train.labels[random_batch_indices,:]} ))
        	hist_pred_noise[i] = np.mean(sess.run([D2],{z_node: sample_Z(100,NOISE_DIM), y_node: mnist.train.labels[random_batch_indices,:]} ))
        
        #Print some information to see whats happening
        if i % (TRAIN_ITERS // 10) == 0:
        	print("Iteration: ",float(i)/float(TRAIN_ITERS))
        	print("G objective (Need to minimise):",histg[i])
        	print("D objective (Need to minimise):",histd[i])
        	print("Average 100 Data into D1:",hist_pred_data[i])
        	print("avg 100 Noise into D1:",hist_pred_noise[i])
        	#Save a plot image
        	pretty_plot(G, z_node, sess, NOISE_DIM,picture_count,i,filedir)
        	picture_count += 1
        	
    
    return hist_pred_noise, hist_pred_data,histd, histg


#NOTE: Trains the NN that was set up using the constructor functions from above
#		 Modifications to the above constructors should be checked for validity by
#		 running the optimization routine on them
#
def train_NN(TRAIN_ITERS, DIAGN_STEP, NOISE_DIM, M, K_G,K_D, image_count, sess, mnist, D1, D2,G, x_node, z_node, obj_d, obj_g, opt_d, opt_g, merged_summ, train_writer, filedir = ""):
	
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
        	pretty_plot(G, z_node, sess, NOISE_DIM,picture_count,i,filedir)
        	picture_count += 1
        	
    
    return hist_pred_noise, hist_pred_data,histd, histg

