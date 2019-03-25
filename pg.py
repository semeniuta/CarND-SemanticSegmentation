# Playground

import os
import tensorflow as tf
from glob import glob
import helper
from functions import load_vgg, load_vgg_graph, layers

data_dir = '/data'
saved_model_dir = '/data/vgg/'
image_dir = os.path.join(data_dir, 'data_road/training/image_2')
image_mask = os.path.join(image_dir, '*.png')

image_shape = (160, 576)
get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)


def get_small_batch(batch_size=2):
    gen = get_batches_fn(batch_size)
    return next(gen)
    
    
if __name__ == '__main__':
    
    print('TensorFlow version:', tf.__version__)
    
    n_images = len(glob(image_mask))
    print('Number of images:', n_images)
    
    small_batch_images, small_batch_gt = get_small_batch()
     
    with tf.Session() as sess:
        
        #g = load_vgg_graph(sess, saved_model_dir)
        tensors = load_vgg(sess, saved_model_dir)
        
        names = ('t_im', 't_keep', 't_out3', 't_out4', 't_out7')
        for name, t in zip(names, tensors):
            print(name, t)
        
        t_im, t_keep, t_out3, t_out4, t_out7 = tensors
        
        t_last = layers(t_out3, t_out4, t_out7, n_classes=2)
        
        t_gt = tf.placeholder(tf.float32, (None, None, None, 2))
        
        fd = {
            t_im: small_batch_images,
            t_gt: small_batch_gt,
            t_keep: 0.5
        }
        
        res = sess.run([t_im, t_gt, t_out3, t_out4, t_out7], feed_dict=fd)
        
        print('Shapes when fed with actual images:')
        for el in res:
            print(el.shape)
        
        
    
