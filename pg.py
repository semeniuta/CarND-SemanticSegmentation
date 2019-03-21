# Playground

import tensorflow as tf
from functions import load_vgg

saved_model_dir = '/data/vgg/'

if __name__ == '__main__':
    
    print('TensorFlow version:', tf.__version__)
    
    with tf.Session() as sess:
        tensors = load_vgg(sess, saved_model_dir)
        
        
    
