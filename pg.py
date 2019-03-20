# Playground

import tensorflow as tf

saved_model_dir = '/data/vgg/'

if __name__ == '__main__':
    
    print('TensorFlow version:', tf.__version__)
    
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['vgg16'], saved_model_dir)
        graph = tf.get_default_graph()
    
        #print(graph.get_operations())