import tensorflow as tf
from scipy.special import binom
from itertools import combinations
import numpy as np


def penal(x, gamma, type='L2'):
    if type == 'L2':
        return gamma * tf.square(tf.nn.relu(x))
    elif type == 'exp':
        return 1/gamma * tf.exp(gamma * x)
    else:
        print('potential ERROR: penalization method not implemented')
        return 0


def penal_der(x, gamma, type='L2'):
    if type == 'L2':
        return 2 * gamma * tf.nn.relu(x)
    elif type == 'exp':
        return tf.exp(gamma * x)
    else:
        print('potential ERROR: penalization method not implemented')
        return 0


def map_feat(x, feat, center=(0.0,), scale=1.0, d=(0,)):
    if len(d) == 1:
        if scale < 0.0001:
            scale = 1
        if feat == 'identity':
            return (x[:, d[0]:d[0]+1] - center[0]) / scale
        if feat == 'square':
            return ((x[:, d[0]:d[0]+1] - center[0]) / scale) ** 2
        if feat == 'sign':
            return tf.sign(x[:, d[0]:d[0]+1] - center[0])
        if feat == 'relu':
            return tf.nn.relu(x[:, d[0]:d[0]+1] - center[0]) / scale
        if feat == 'pow4':
            return ((x[:, d[0]:d[0]+1] - center[0]) / scale) ** 4
        else:
            print('potential ERROR: feature not supplied')
            return 0
    if len(d) == 2:
        center = [0.0] * len(d)
        if feat == 'max':
            return tf.maximum(x[:, d[0]:d[0]+1] - center[0], x[:, d[1]:d[1]+1] - center[1]) / scale
        if feat == 'prod':
            return (x[:, d[0]:d[0]+1] - center[0]) * (x[:, d[1]:d[1]+1] - center[1]) / scale
        if feat == 'signdiff':
            return tf.sign((x[:, d[0]:d[0]+1] - center[0]) - (x[:, d[1]:d[1]+1] - center[1]))
        else:
            print('potential ERROR: feature not supplied')
            return 0
    else:
        print('potential ERROR: feature not supplied')
        return 0


def layer(x, layernum, input_dim, output_dim, activation='ReLu', type_num=32, outputlin=0):
    if type_num == 32:
        ua_w = tf.get_variable('ua_w' + str(layernum), shape=[input_dim, output_dim],
                               initializer=tf.initializers.glorot_normal(), dtype=tf.float32)
        ua_b = tf.get_variable('ua_b' + str(layernum), shape=[output_dim],
                               initializer=tf.initializers.glorot_normal(),
                               dtype=tf.float32)
    else:
        ua_w = tf.get_variable('ua_w' + str(layernum), shape=[input_dim, output_dim],
                               initializer=tf.initializers.glorot_normal(), dtype=tf.float64)
        ua_b = tf.get_variable('ua_b' + str(layernum), shape=[output_dim],
                               initializer=tf.initializers.glorot_normal(),
                               dtype=tf.float64)
    if outputlin:
        z = tf.matmul(x, ua_w)
    else:
        z = tf.matmul(x, ua_w) + ua_b
    if activation == 'identity':
        return z
    if activation == 'ReLu':
        return tf.nn.relu(z)
    if activation == 'tanh':
        return tf.nn.tanh(z)
    if activation == 'leakyReLu':
        return tf.nn.leaky_relu(z)
    if activation == 'softplus':
        return tf.nn.softplus(z)
    else:
        return z


def pos_layer(x, layernum, input_dim, output_dim, activation='ReLu', type_num=32):
    if type_num == 32:
        ua_w = tf.get_variable('ua_w' + str(layernum), shape=[input_dim, output_dim],
                               initializer=tf.initializers.glorot_normal(), dtype=tf.float32)
        ua_b = tf.get_variable('ua_b' + str(layernum), shape=[output_dim],
                               initializer=tf.initializers.glorot_normal(),
                               dtype=tf.float32)
    else:
        ua_w = tf.get_variable('ua_w' + str(layernum), shape=[input_dim, output_dim],
                               initializer=tf.initializers.glorot_normal(), dtype=tf.float64)
        ua_b = tf.get_variable('ua_b' + str(layernum), shape=[output_dim],
                               initializer=tf.initializers.glorot_normal(),
                               dtype=tf.float64)
    z = tf.matmul(x, tf.exp(ua_w)) + tf.exp(ua_b)
    if activation == 'ReLu':
        return tf.nn.relu(z)
    if activation == 'tanh':
        return tf.nn.tanh(z)
    if activation == 'leakyReLu':
        return tf.nn.leaky_relu(z)
    if activation == 'softplus':
        return tf.nn.softplus(z)
    else:
        return z


def numpy_univ_approx(n_layers, w_list, b_list, activation='ReLu'):
    if activation == 'ReLu':
        def np_activation(x):
            return np.maximum(x, 0)

    def net_rebuild(x):
        a = np.matmul(x, w_list[0]) + b_list[0]
        for i in range(1, n_layers):
            a = np_activation(a)
            a = np.matmul(a, w_list[i]) + b_list[i]
        return a

    return net_rebuild

def univ_approx(x, name, n_layers=4, hidden_dim=32, input_dim=1, output_dim=1, activation='ReLu', positive=0,
                outputlin=0, type_num=32):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if positive == 0:
            if n_layers == 1:
                return layer(x, 0, input_dim, output_dim, activation='', type_num=type_num)
            else:
                a = layer(x, 0, input_dim, hidden_dim, activation=activation, type_num=type_num)
                for i in range(1, n_layers - 1):
                    a = layer(a, i, hidden_dim, hidden_dim, activation=activation, type_num=type_num)
                a = layer(a, n_layers - 1, hidden_dim, output_dim, activation='', outputlin=outputlin,
                          type_num=type_num)
                return a
        else:
            if n_layers == 1:
                return pos_layer(x, 0, input_dim, output_dim, activation='', type_num=type_num)
            else:
                a = layer(x, 0, input_dim, hidden_dim, type_num=type_num)
                for i in range(1, n_layers - 1):
                    a = layer(a, i, hidden_dim, hidden_dim, activation=activation, type_num=type_num)
                a = pos_layer(a, n_layers - 1, hidden_dim, output_dim, activation='', type_num=type_num)
                return a


def ff_net(x, name, feats=((1, 'identity'),), n_layers=4, hidden_dim=32, input_dim=1, output_dim=1, activation='ReLu',
           positive=0, outputlin=0, type_num=32):
    # x should be [A, input_dim] tf array, where A is arbitrary
    # returns tf array of shape [A, output_dim]

    count = 0
    # extract all features from x
    for tup in list(combinations(range(input_dim), feats[0][0])):
        if count == 0:
            inp_x = map_feat(x, feats[0][1], d=tup)
        else:
            inp_x = tf.concat([inp_x, map_feat(x, feats[0][1], d=tup)], axis=1)
        count += 1

    for ic in range(1, len(feats)):
        for tup in list(combinations(range(input_dim), feats[ic][0])):
            inp_x = tf.concat([inp_x, map_feat(x, feats[ic][1], d=tup)], axis=1)
    # return feed forward network applied to features
    inp_x_dim = inp_x.shape[1]
    return univ_approx(inp_x, name=name, n_layers=n_layers, hidden_dim=hidden_dim, input_dim=inp_x_dim,
                       output_dim=output_dim, activation=activation, positive=positive, outputlin=outputlin,
                       type_num=type_num)





# # Tests:
# import numpy as np
# FEATURES = [(1, 'identity'), (1, 'square'), (2, 'prod')]
# d = 10
# X = tf.placeholder(dtype=tf.float32, shape=[None, d])
# Y = ff_net(X, 'Hallo', feats=FEATURES, input_dim=d)
# print(Y.shape)
#
# batch = 2 ** 5
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     yval = sess.run(Y, feed_dict={X: np.random.random_sample([batch, d])})
#     print(yval)

