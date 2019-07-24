import tensorflow as tf


param = {
    'lr': 0.001,
    'batch_size': 60,
    'num_steps': 100,
    'display_step': 10,
    'num_inputs': 28,
    'num_channels': 1,
    'num_classes': 10,
    'conv1_size': 3,
    'conv2_size': 1,
    'conv1_deep': 64,
    'conv2_deep': 64,
    'conv3_deep': 128,
    'conv4_deep': 128,
    'conv5_deep': 256,
    'conv6_deep': 256,
    'conv7_deep': 256,
    'conv8_deep': 512,
    'conv9_deep': 512,
    'conv10_deep': 512,
    'conv11_deep': 512,
    'conv12_deep': 512,
    'conv13_deep': 512,
    'resize': 512,
    'fc1_size': 512,
    'fc2_size': 512,
}

weights = {
    'l1': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['num_channels'],
                                           param['conv1_deep']], mean=0, stddev=0.1)),

    'l2': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv1_deep'],
                                           param['conv2_deep']], mean=0, stddev=0.1)),

    'l3': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv2_deep'],
                                           param['conv3_deep']], mean=0, stddev=0.1)),

    'l4': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv3_deep'],
                                           param['conv4_deep']], mean=0, stddev=0.1)),

    'l5': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv4_deep'],
                                           param['conv5_deep']], mean=0, stddev=0.1)),

    'l6': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv5_deep'],
                                           param['conv6_deep']], mean=0, stddev=0.1)),

    'l7': tf.Variable(tf.truncated_normal([param['conv2_size'],
                                           param['conv2_size'],
                                           param['conv6_deep'],
                                           param['conv7_deep']], mean=0, stddev=0.1)),

    'l8': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv7_deep'],
                                           param['conv8_deep']], mean=0, stddev=0.1)),

    'l9': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv8_deep'],
                                           param['conv9_deep']], mean=0, stddev=0.1)),

    'l10': tf.Variable(tf.truncated_normal([param['conv2_size'],
                                           param['conv2_size'],
                                           param['conv9_deep'],
                                           param['conv10_deep']], mean=0, stddev=0.1)),

    'l11': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv10_deep'],
                                           param['conv11_deep']], mean=0, stddev=0.1)),

    'l12': tf.Variable(tf.truncated_normal([param['conv1_size'],
                                           param['conv1_size'],
                                           param['conv11_deep'],
                                           param['conv12_deep']], mean=0, stddev=0.1)),

    'l13': tf.Variable(tf.truncated_normal([param['conv2_size'],
                                           param['conv2_size'],
                                           param['conv12_deep'],
                                           param['conv13_deep']], mean=0, stddev=0.1)),

    'l14': tf.Variable(tf.truncated_normal([param['resize'],
                                           param['fc1_size']], mean=0, stddev=0.01)),
    'l15': tf.Variable(tf.truncated_normal([param['fc1_size'],
                                           param['fc2_size']], mean=0, stddev=0.01)),
    'l16': tf.Variable(tf.truncated_normal([param['fc2_size'],
                                           param['num_classes']], mean=0, stddev=0.01))
}

biases = {
    'l1': tf.Variable(tf.zeros([param['conv1_deep']])),
    'l2': tf.Variable(tf.zeros([param['conv2_deep']])),
    'l3': tf.Variable(tf.zeros([param['conv3_deep']])),
    'l4': tf.Variable(tf.zeros([param['conv4_deep']])),
    'l5': tf.Variable(tf.zeros([param['conv5_deep']])),
    'l6': tf.Variable(tf.zeros([param['conv6_deep']])),
    'l7': tf.Variable(tf.zeros([param['conv7_deep']])),
    'l8': tf.Variable(tf.zeros([param['conv8_deep']])),
    'l9': tf.Variable(tf.zeros([param['conv9_deep']])),
    'l10': tf.Variable(tf.zeros([param['conv10_deep']])),
    'l11': tf.Variable(tf.zeros([param['conv11_deep']])),
    'l12': tf.Variable(tf.zeros([param['conv12_deep']])),
    'l13': tf.Variable(tf.zeros([param['conv13_deep']])),
    'l14': tf.Variable(tf.zeros([param['fc1_size']])),
    'l15': tf.Variable(tf.zeros([param['fc2_size']])),
    'l16': tf.Variable(tf.zeros([param['num_classes']]))
}


