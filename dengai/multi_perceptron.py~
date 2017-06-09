import tensorflow as tf 

def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2, w_3, keep_prob):

    h = tf.nn.tanh(tf.nn.dropout(tf.matmul(X, w_1), keep_prob))
    out = tf.nn.tanh(tf.nn.dropout(tf.matmul(h, w_2), keep_prob))  
    yhat = tf.matmul(out, w_3)
    return yhat
