import tensorflow as tf
from config import *
from tool import *
import random
# s1 = tf.get_variable(dtype=tf.float32, shape=[20,10,200], name='s1')
# s2 = tf.get_variable(dtype=tf.float32, shape=[20,20,200], name='s2')
# matrix_1 = tf.matmul(s1, tf.transpose(s2, perm=[0, 2, 1]))
# matrix_2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(s1), axis=-1)), axis=-1)
# matrix_3 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(s2), axis=-1)), axis=1)
# cosin_matrix = tf.div(matrix_1, tf.matmul(matrix_2, matrix_3))
#
# softmax_s1 = tf.nn.softmax(tf.reduce_mean(cosin_matrix,axis=-1,keep_dims=True), dim=-1)
# cosin_matrix_s2 = tf.transpose(cosin_matrix, perm=[0, 2, 1])
# softmax_s2 = tf.nn.softmax(tf.reduce_mean(cosin_matrix_s2,axis=-1,keep_dims=True), dim=-1)
# a_s2 = tf.multiply(softmax_s2, s2)
# a_s1 = tf.multiply(softmax_s1, s1)
s1 = np.array([1,2,3,4,5,6,7,8])
int_ran = random.sample([i for i in range(len(s1))],3)
print(int_ran)
print(s1[int_ran])
# print(a_s2)
# print(a_s1)

# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     sess.run(matrix_1)

