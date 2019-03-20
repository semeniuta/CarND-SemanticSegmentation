# Playground

import tensorflow as tf

saved_model_dir = '/data/vgg/'

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
        
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
        graph = tf.get_default_graph()
        
    names = (
        vgg_input_tensor_name, 
        vgg_keep_prob_tensor_name, 
        vgg_layer3_out_tensor_name, 
        vgg_layer4_out_tensor_name, 
        vgg_layer7_out_tensor_name
    )
    
    tensors = tuple(graph.get_tensor_by_name(name) for name in names)
    return tensors

if __name__ == '__main__':
    
    print('TensorFlow version:', tf.__version__)
    
    
    
    """
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['vgg16'], saved_model_dir)
        graph = tf.get_default_graph()

        #print(graph.get_operations())
        
    """
