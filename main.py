#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

from functions import load_vgg, layers, optimize, train_nn

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

tests.test_load_vgg(load_vgg, tf)
tests.test_layers(layers)
tests.test_optimize(optimize)
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = '/data'
    saved_model_dir = os.path.join(data_dir, 'vgg')
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        
        tensors = load_vgg(sess, saved_model_dir)
        t_im, t_keep, t_out3, t_out4, t_out7 = tensors
        
        t_last = layers(t_out3, t_out4, t_out7, n_classes=2)
        
        # TODO: Train NN using the train_nn function
        
        epochs = 5
        batch_size = 8
        
        t_gt = tf.placeholder(tf.float32, (None, None, None, 2), name='ground_truth')
        t_rate = tf.placeholder(tf.float32, (), name='learning_rate')
        num_classes = 2
        
        logits, train_op, ce_loss = optimize(t_last, t_gt, t_rate, num_classes)
        
        sess.run(tf.global_variables_initializer())
        
        train_nn(
            sess,
            epochs,
            batch_size,
            get_batches_fn,
            train_op,
            ce_loss,
            t_im,
            t_gt, 
            t_keep, 
            t_rate
        )
        
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
