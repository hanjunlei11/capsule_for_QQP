import tensorflow as tf
from config import *
from tool import *
from function import *

class Model():
    def __init__(self,embadding_re):
        self.batch_size = batch_size
        self.batch_len  = batch_len
        self.embedding_size = embadding_size
        self.voca_size = vocab_size
        self.learnning_rate = learning_rate
        self.char_size = char_size
        self.embedding_char_size = embadding_char_size
        self.word_length = word_length

        with tf.name_scope('input'):
            self.s1 = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.batch_len),name='s1_input')
            self.s2 = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.batch_len),name='s2_input')
            self.input_char_s1 = tf.placeholder(dtype=tf.int32,shape=(self.batch_size, self.batch_len, self.word_length),name='char_s1')
            self.input_char_s2 = tf.placeholder(dtype=tf.int32,shape=(self.batch_size, self.batch_len, self.word_length),name='char_s2')
            self.label = tf.placeholder(dtype=tf.int64,shape=(None),name='label')
            self.keep_rate = tf.placeholder(dtype=tf.float32,shape=(None),name='keep_rate')
            # self.embedding_keep_rate = tf.placeholder(dtype=tf.float32,shape=(None),name='embedding_keeprate')
            self.is_training = tf.placeholder(dtype=tf.bool,shape=(None),name='training')

        with tf.name_scope('embedding'):
            embedding_table = tf.Variable(embadding_re,trainable=False, name='word_embedding',dtype=tf.float32)
            self.s1_matrix_tr = tf.nn.embedding_lookup(embedding_table, self.s1)
            self.s2_matrix_tr = tf.nn.embedding_lookup(embedding_table, self.s2)
            embedding = tf.Variable(tf.random_uniform([self.voca_size, self.embedding_size]),trainable=True,dtype=tf.float32,name='embeding')
            self.s1_matrix = tf.nn.embedding_lookup(embedding, self.s1)
            self.s2_matrix = tf.nn.embedding_lookup(embedding, self.s2)

        with tf.name_scope('char_embedding'):
            embedding_char = tf.Variable(tf.random_uniform([self.char_size, self.embedding_char_size], dtype=tf.float32), trainable=True,
                name='char_embedding')
            s1_char_embedding = tf.nn.embedding_lookup(embedding_char, self.input_char_s1)
            s2_char_embedding = tf.nn.embedding_lookup(embedding_char, self.input_char_s2)
            # self.s1_char_conv_out = tf.nn.dropout(s1_char_embedding, keep_prob=self.embedding_keep_rate)
            # self.s2_char_conv_out = tf.nn.dropout(s2_char_embedding, keep_prob=self.embedding_keep_rate)
            self.s1_char_conv_out = conv2D(inputs=s1_char_embedding, kernel_shape=[1, 5, 100, 100],
                                                strides=[1, 1, 1, 1],trainning=self.is_training, padding='VALID', kernel_name='char_kernel1')
            self.s1_char_conv_out = tf.layers.max_pooling2d(inputs=self.s1_char_conv_out, pool_size=[1, 12],
                                                            strides=[1, 1])
            self.s1_char_conv_out = tf.reduce_mean(self.s1_char_conv_out, axis=2)
            self.s2_char_conv_out = conv2D(inputs=s2_char_embedding, kernel_shape=[1, 5, 100, 100],
                                                strides=[1, 1, 1, 1],trainning=self.is_training, padding='VALID', kernel_name='char_kernel2')
            self.s2_char_conv_out = tf.layers.max_pooling2d(inputs=self.s2_char_conv_out, pool_size=[1, 12],
                                                            strides=[1, 1])
            self.s2_char_conv_out = tf.reduce_mean(self.s2_char_conv_out, axis=2)
            self.s1_concat = tf.concat([self.s1_matrix_tr, self.s1_matrix,self.s1_char_conv_out], axis=-1)
            self.s2_concat = tf.concat([self.s2_matrix_tr, self.s2_matrix,self.s2_char_conv_out], axis=-1)

        with tf.name_scope("decoder"):
            self.encoder_1_s1, self.encoder_1_s2 = Dynamic_LSTM(self.s1_concat,self.s2_concat,keep_rate=self.keep_rate,training=self.is_training,name='decoder1')
            self.encoder_2_s1, self.encoder_2_s2 = Dynamic_LSTM(self.encoder_1_s1,self.encoder_1_s2, keep_rate=self.keep_rate,training=self.is_training,name='decoder2')
            self.encoder_3_s1, self.encoder_3_s2 = Dynamic_LSTM(self.encoder_2_s1,self.encoder_2_s2, keep_rate=self.keep_rate,training=self.is_training, name='decoder3')

        with tf.name_scope('matching_layer'):
            for i in range(self.encoder_3_s1.shape[1]):
                temp = tf.slice(self.encoder_3_s1, [0, i, 0], [-1, 1, -1])
                for j in range(self.encoder_3_s2.shape[1]):
                    temp1 = tf.slice(self.encoder_3_s2, [0, j, 0], [-1, 1, -1])
                    temp2 = tf.multiply(temp, temp1)
                    if j == 0:
                        self.temp4 = temp2
                    else:
                        self.temp4 = tf.concat([self.temp4, temp2], axis=1)
                self.temp4 = tf.expand_dims(self.temp4, axis=1)
                if i == 0:
                    self.temp5 = self.temp4
                else:
                    self.temp5 = tf.concat([self.temp5,self.temp4], axis=1)
            self.temp5 = tf.layers.batch_normalization(inputs=self.temp5, training=self.is_training)

        with tf.name_scope('Primary_Capsule'):
            self.caps1 = PrimaryCaps(self.temp5,vec_len=8,num_outputs=16,kernel_size=[5,5],stride=[4,4],training=self.is_training)

        with tf.name_scope('caps_chenge'):
            self.caps2 = chenge(self.caps1, output_dim=18, output_num=26,training=self.is_training,name='caps_chenge1')
            self.caps2 = squash(self.caps2,axis=-1)
            self.caps2 = routing(self.caps2,is_training=self.is_training)
            self.caps3 = chenge(self.caps2, output_dim=10, output_num=10,training=self.is_training,name='caps_chenge2')
            self.caps3 = squash(self.caps3,axis=-1)
            self.caps3 = routing(self.caps3,is_training=self.is_training)
            self.caps4 = chenge(self.caps3,output_dim=10, output_num=2,training=self.is_training,name='caps_chenge3')
            self.caps4 = squash(self.caps4,axis=-1)
            self.caps4 = routing(self.caps4, is_training=self.is_training)


        with tf.name_scope('reconstruction'):
            self.re_struct = chenge(self.caps4,output_dim=13,output_num=13,training=self.is_training,name='reconstruction1')
            self.re_struct = tf.reduce_mean(tf.reduce_sum(self.re_struct,axis=1),axis=-1)
            self.re_struct = squash(self.re_struct,axis=-1)
            self.re_struct = chenge(self.re_struct, output_dim=batch_len, output_num=batch_len,training=self.is_training, name='reconstruction2')
            self.re_struct = tf.reduce_mean(tf.reduce_sum(self.re_struct, axis=1), axis=-1)
            self.re_struct = squash(self.re_struct,axis=-1)
            self.re_struct = tf.expand_dims(self.re_struct,axis=-1)
            self.re_struct = fully_conacation(self.re_struct,haddin_size=hidden_size)
            self.restruct_loss = tf.losses.mean_squared_error(self.re_struct,self.temp5)
            tf.add_to_collection('losses',self.restruct_loss)
            tf.summary.scalar('restrcut_loss',self.restruct_loss)

        with tf.name_scope('loss'):
            self.prediction = tf.norm(self.caps4,axis=-1)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,logits=self.prediction),axis=-1)
            tf.add_to_collection('losses',self.loss)
            tf.summary.scalar('margin_loss', self.loss)
            self.all_loss = tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('all_loss',self.all_loss)

        with tf.name_scope('acc'):
            self.max_index = tf.argmax(self.prediction, axis=1)
            cast_value = tf.cast(tf.equal(self.max_index, self.label), dtype=tf.float32)
            self.acc = tf.reduce_mean(cast_value, axis=-1)
            tf.summary.scalar('acc', self.acc)

# s1_word_train,s1_word_test,s2_word_train,s2_word_test,vector_lines,label_train,label_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test = read_file(s1path='./s1.txt',s2path='./s2.txt',labelpath='./label.txt',re_vector='./vector.txt')
#
# model = Model(vector_lines)
# init_op=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     sess.run(model.similar_s12)

