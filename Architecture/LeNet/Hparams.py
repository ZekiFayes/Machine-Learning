import tensorflow as tf


param = {
    'lr': 0.001,
    'batch_size': 100,
    'num_steps': 1000,
    'display_step': 100,
    'num_inputs': 28,
    'num_channels': 1,
    'num_classes': 10,
    'con1_deep': 32,
    'con1_size': 5,
    'con2_deep': 64,
    'con2_size': 5,
    'Resize': 3136,
    'FC_size': 512
}

weights = {
    'l1': tf.Variable(tf.truncated_normal([param['con1_size'],
                                           param['con1_size'],
                                           param['num_channels'],
                                           param['con1_deep']], mean=0, stddev=0.1)),
    'l2': tf.Variable(tf.truncated_normal([param['con2_size'],
                                           param['con2_size'],
                                           param['con1_deep'],
                                           param['con2_deep']], mean=0, stddev=0.1)),

    'l3': tf.Variable(tf.truncated_normal([param['Resize'],
                                           param['FC_size']], mean=0, stddev=0.01)),
    'l4': tf.Variable(tf.truncated_normal([param['FC_size'],
                                           param['num_classes']], mean=0, stddev=0.01))
}

biases = {
    'l1': tf.Variable(tf.zeros([param['con1_deep']])),
    'l2': tf.Variable(tf.zeros([param['con2_deep']])),
    'l3': tf.Variable(tf.zeros([param['FC_size']])),
    'l4': tf.Variable(tf.zeros([param['num_classes']]))
}


