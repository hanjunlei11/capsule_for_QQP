import tensorflow as tf
from config import *
from tool import *
kernel = tf.Variable([[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]],trainable=True, name='word_embedding')
c = tf.norm(kernel,axis=-1)
c1 = tf.sqrt(tf.reduce_sum(tf.square(kernel),axis=-1))
print(c)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(c))
    print(sess.run(c1))

