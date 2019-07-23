import tensorflow as tf


param = {
    'learning_rate': 0.005,
    'num_steps': 10000,
    'batch_size': 100,
    'display_step': 1000,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden': 200,
}


u = {
    't1': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01))
}

v = {
    't1': tf.Variable(tf.random_normal([param['hidden'], param['num_classes']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden'], param['num_classes']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden'], param['num_classes']], mean=0, stddev=0.01))
}

w = {
    't1': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01))
}

bs = {
    't1': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01))
}

bo = {
    't1': tf.Variable(tf.random_normal([param['num_classes']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['num_classes']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['num_classes']], mean=0, stddev=0.01))
}

