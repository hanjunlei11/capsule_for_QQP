import tensorflow as tf
from config import *
from function import *
from tool import *
import random
s1 = tf.get_variable(dtype=tf.float32,shape=[70,26,16,100],name='s1')
# cell_f_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=5,reuse=tf.AUTO_REUSE,activation=tf.keras.layers.LeakyReLU())
# cell_b_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=5,reuse=tf.AUTO_REUSE,activation=tf.keras.layers.LeakyReLU())
# cell_z_f = cell_f_1.zero_state(batch_size=1,dtype=tf.float32)
# cell_z_b = cell_b_1.zero_state(batch_size=1,dtype=tf.float32)
# lstm_output_s1, _ = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,inputs=s1,dtype=tf.float32,initial_state_fw=cell_z_f,initial_state_bw=cell_z_b)
# tv = tf.trainable_variables()
# K = tf.keras.layers.LeakyReLU()
k = conv2D(s1,kernel_shape=[1,5,100,100],strides=[1,1,1,1],trainning=True,padding='VALID',kernel_name='conv1')
print(k)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(k)

