# Playground

import os
import sys
import tensorflow as tf
from glob import glob
import helper
from functions import load_vgg, load_vgg_graph, layers, add_regularization

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
    
    num_classes = 2
     
    with tf.Session() as sess:
        
        #g = load_vgg_graph(sess, saved_model_dir)
        tensors = load_vgg(sess, saved_model_dir)
        
        names = ('t_im', 't_keep', 't_out3', 't_out4', 't_out7')
        for name, t in zip(names, tensors):
            print(name, t)
        
        t_im, t_keep, t_out3, t_out4, t_out7 = tensors
        
        t_last = layers(t_out3, t_out4, t_out7, n_classes=num_classes)
        
        t_gt = tf.placeholder(tf.float32, (None, None, None, num_classes))
        
        t_logits = tf.reshape(t_last, (-1, num_classes), name='logits') 
    
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=t_gt,
            logits=t_logits,
            name='ce_loss'
        ) 
        
        loss_op = tf.reduce_mean(ce_loss, name='loss_op')
        
        new_loss = add_regularization(sess, loss_op, beta=1e-2)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(new_loss, name='train_op')
        
        #sys.exit(0)
        
        for batch_im, batch_gt in get_batches_fn(2):
        
            fd = {
                t_im: batch_im,
                t_gt: batch_gt,
                t_keep: 0.5
            }

            sess.run(tf.global_variables_initializer())

            sess.run(train_op, feed_dict=fd)

            res = sess.run([t_im, t_gt, t_out3, t_out4, t_out7, t_last, t_logits, ce_loss, loss_op], feed_dict=fd)

            print('Shapes when fed with actual images:')
            for el in res:
                print(el.shape)

            print(res[-1])
        
        
    
