import tensorflow as tf
# initializer = tf.contrib.layers.xavier_initializer()
# initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
initializer = tf.random_normal_initializer()
i = 0
def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''

    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)
    num_output1 = 3
    num_output2 = 3
    num_output3 = 3
    num_output4 = 2
    # num_output4 = 2
    # num_output4 = 2
    # TODO

    with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
        fc1 = FC_Block("FC_1", "BN_1", "ReLU_1", x, num_output1)
        fc1 = FC_Block("FC_2", "BN_2", "ReLU_2", fc1, num_output2)
        fc1 = FC_Block("FC_3", "BN_3", "ReLU_3", fc1, num_output3)
        fc1 = FC_Block("FC_4", "BN_4", "ReLU_4", fc1, num_output4)

        return fc1

    pass

def FC_Block(fc_name, bn_name, relu_name, input, num_output):
    global i
    print(i)
    i+=1
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)
    fc = tf.layers.dense(input,units=num_output,kernel_initializer=initializer,name=fc_name)
    bn = tf.layers.batch_normalization(fc,name=bn_name, trainable = is_training)
    relu = tf.nn.relu(bn,name=relu_name)
    # relu = tf.sigmoid(bn, name=relu_name)
    return relu

