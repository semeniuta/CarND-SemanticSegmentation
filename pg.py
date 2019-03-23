# Playground

import os
import tensorflow as tf
import helper
from functions import load_vgg, load_vgg_graph

data_dir = '/data'
saved_model_dir = '/data/vgg/'

if __name__ == '__main__':
    
    print('TensorFlow version:', tf.__version__)
    
    image_shape = (160, 576)
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
    
    with tf.Session() as sess:
        
        #g = load_vgg_graph(sess, saved_model_dir)
        tensors = load_vgg(sess, saved_model_dir)
        
        t_out = tensors[-1]
        
        
    
