import tensorflow as tf


param = {
    'lr': 0.005,
    'batch_size': 100,
    'num_steps': 10000,
    'display_step': 1000,
    'num_inputs': 784,
    'num_classes': 10,
    'hidden': 200,

}

Wf = {
    't1': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01))
}

Uf = {
    't1': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01))
}

Bf = {
    't1': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01))
}

Wi = {
    't1': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
}

Ui = {
    't1': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01))
}

Bi = {
    't1': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01))
}

Wc = {
    't1': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
}

Uc = {
    't1': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01))
}

Bc = {
    't1': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01))
}

Wo = {
    't1': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden'], param['hidden']], mean=0, stddev=0.01))
}

Uo = {
    't1': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['num_inputs'], param['hidden']], mean=0, stddev=0.01))
}

Bo = {
    't1': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden']], mean=0, stddev=0.01)),
}

V = {
    't1': tf.Variable(tf.random_normal([param['hidden'], param['num_classes']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['hidden'], param['num_classes']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['hidden'], param['num_classes']], mean=0, stddev=0.01))
}

Bs = {
    't1': tf.Variable(tf.random_normal([param['num_classes']], mean=0, stddev=0.01)),
    't2': tf.Variable(tf.random_normal([param['num_classes']], mean=0, stddev=0.01)),
    't3': tf.Variable(tf.random_normal([param['num_classes']], mean=0, stddev=0.01))
}
