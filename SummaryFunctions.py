import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def sample_Z_2(m, n):
    return np.random.uniform(-1., 1., size=[m, n])	  

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
        
        
def data_noise_png(D1, D2, x_node, z_node, image_count, hist_pred_data, hist_pred_noise, mnist, sess, TRAIN_ITERS, NOISE_Dim):
    #Check performance of 100 noise samples into Discriminator
    print("avg 100 Noise into D1:",
    np.mean(sess.run([D2],{z_node: sample_Z_2(100,NOISE_Dim)} ))
    )
    #Check performance of 100 data inputs into discriminator
    print("Average 100 Data into D1:",
    np.mean(sess.run([D1],{x_node: mnist.train.images[np.random.choice(image_count,100),:]} ))
    )
  	 #Generate the plot of DATA_NOISE and save to hard drive
    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(TRAIN_ITERS), hist_pred_data , 'b-')
    plt.subplot(212)
    plt.plot(range(TRAIN_ITERS), hist_pred_noise , 'b-')
    plt.savefig("DATA_NOISE.png",bbox_inches="tight")
     
def pretty_plot(G, z_node, sess, NOISE_Dim):
    #----------------------Generate samples and plot, save to "pretty_pictures.png" --------------------------------
    samples = sess.run(G, feed_dict={z_node: sample_Z_2(16, NOISE_Dim)})
    fig = plot(samples)
    plt.savefig('pretty_plot.png', bbox_inches='tight')
    # TODO: need to find regions of high probability mass to generate sensible figures (interpolation pherhaps?)

        
        
        
        