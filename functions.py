import tensorflow as tf
import json
import os
import shutil

def load_vgg_graph(sess, vgg_path):

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    return tf.get_default_graph()

    
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    graph = load_vgg_graph(sess, vgg_path)
        
    names = (
        vgg_input_tensor_name, 
        vgg_keep_prob_tensor_name, 
        vgg_layer3_out_tensor_name, 
        vgg_layer4_out_tensor_name, 
        vgg_layer7_out_tensor_name
    )
    
    tensors = tuple(graph.get_tensor_by_name(name) for name in names)
    return tensors


def upsample_layer(t_out, num_classes, kernel_size, strides, l2_const=1e-3, **kwargs):
    
    t_upsample = tf.layers.conv2d_transpose(
        t_out, 
        num_classes, 
        kernel_size, 
        strides, 
        padding='same', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_const),
        **kwargs
    )
    
    return t_upsample


def layers(t_out3, t_out4, t_out7, n_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param t_out3: TF Tensor for VGG Layer 3 output
    :param t_out4: TF Tensor for VGG Layer 4 output
    :param t_out7: TF Tensor for VGG Layer 7 output
    :param n_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    t_up74 = upsample_layer(t_out7, num_classes=512, kernel_size=4, strides=2, name='upsample74')
    t_skip4 = tf.add(t_up74, t_out4, name='skip4')
    
    t_up43 = upsample_layer(t_up74, num_classes=256, kernel_size=4, strides=2, name='upsample43')
    t_skip3 = tf.add(t_up43, t_out3, name='skip3')
    
    t_last = upsample_layer(t_up43, num_classes=n_classes, kernel_size=16, strides=8, name='last')
    
    print('t_up74', t_up74)
    print('t_skip4', t_skip4)
    print('t_up43', t_up43)
    print('t_skip3', t_skip3)
    print('t_last', t_last)
    
    return t_last


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits') 
    
    ce_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=correct_label,
            logits=logits,
        ),
        name='ce_loss'
    )
    
    #loss_op = tf.reduce_mean(ce_loss, name='loss_op')
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(ce_loss, name='train_op')
    
    return logits, train_op, ce_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, keep_prob_val=0.5, rate_val=1e-3):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    for i in range(epochs):
        print('Epoch', i + 1)
            
        for batch_im, batch_gt in get_batches_fn(batch_size):
            
            fd = {
                input_image: batch_im,
                correct_label: batch_gt,
                keep_prob: keep_prob_val,
                learning_rate: rate_val
            }

            #_, loss = sess.run([train_op, cross_entropy_loss], feed_dict=fd)
            sess.run(train_op, feed_dict=fd)
            loss = sess.run(cross_entropy_loss, feed_dict=fd)
            
            print('Loss:', loss)
    

def save_model_sm(sess, savedmodel_dir, hyper, t_im, t_gt, t_keep, logits, ce_loss):
    
    # Implementation based on example at
    # https://medium.com/@jsflo.dev/saving-and-loading-a-tensorflow-model-using-the-savedmodel-api-17645576527
    
    if os.path.exists(savedmodel_dir):
        shutil.rmtree(savedmodel_dir)
    
    """
    # doesn't exist in TF 1.3
    tf.saved_model.simple_save(
        sess,
        savedmodel_dir,
        inputs={"t_im": t_im, "t_gt": t_gt, "t_keep": t_keep},
        outputs={"logits": logits, "ce_loss": ce_loss}
    )
    """
    
    psd = tf.saved_model.signature_def_utils.predict_signature_def
    
    builder = tf.saved_model.builder.SavedModelBuilder(savedmodel_dir)

    signature = psd(inputs={"t_im": t_im, "t_gt": t_gt, "t_keep": t_keep},
                            outputs={"logits": logits, "ce_loss": ce_loss})
    # using custom tag instead of: tags=[tag_constants.SERVING]
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=["fcn"],
                                         signature_def_map={'predict': signature})
    builder.save()
    
    with open(os.path.join(savedmodel_dir, 'hyper.json'), 'w') as jf:  
        json.dump(hyper, jf)

        
def save_model_saver(sess, saver_dir, cpkt_name, hyper):
   
    saver = tf.train.Saver()

    saver_path = os.path.join(saver_dir, cpkt_name)
    saver.save(sess, saver_path)
    
    with open(os.path.join(saver_dir, 'hyper.json'), 'w') as jf:  
        json.dump(hyper, jf)
    
    
