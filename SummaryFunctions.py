
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