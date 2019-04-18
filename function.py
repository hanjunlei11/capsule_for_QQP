import tensorflow as tf
from config import *
from tool import *


def conv2D(inputs, kernel_shape, strides, padding, kernel_name, activation='relu', dropuot_rate=None):
    kernel = tf.get_variable(dtype=tf.float32, shape=kernel_shape, name=kernel_name)
    conv_output = tf.nn.conv2d(input=inputs, filter=kernel, strides=strides, padding=padding)
    # conv_output = tf.layers.batch_normalization(inputs=conv_output, training=self.is_traning)
    if activation == 'relu':
        conv_output = tf.nn.relu(conv_output)
    if activation == 'leaky_relu':
        conv_output = tf.nn.leaky_relu(conv_output)
    if dropuot_rate is not None:
        conv_output = tf.nn.dropout(conv_output, keep_prob=dropuot_rate)
    return conv_output

def PrimaryCaps(input,vec_len,num_outputs,kernel_size,stride,training):
    '''
    :param input:
    :param vec_len:
    :param num_outputs:
    :param kernel_size:
    :param stride:
    :return: [batch_size,-1,vec_len]
    '''
    capsules = []
    for i in range(vec_len):
        # 所有Capsule的一个分量，其维度为: [batch_size, 6, 6, 32]，即6×6×1×32
        with tf.variable_scope('ConvUnit_' + str(i)):
            caps_i = tf.contrib.layers.conv2d(input, num_outputs,
                                              kernel_size, stride,
                                              padding="VALID")

            # 将一般卷积的结果张量拉平，并为添加到列表中
            caps_i = tf.reshape(caps_i,shape=[batch_size,-1,1,1])
            caps_i = tf.reduce_mean(caps_i,axis=-1)
            capsules.append(caps_i)

    # 合并为PrimaryCaps的输出张量，即6×6×32个长度为8的向量
    capsules = tf.concat(capsules, axis=-1)
    # 将每个Capsule 向量投入非线性函数squash进行缩放与激活
    capsules = tf.layers.batch_normalization(inputs=capsules, training=training)
    capsules = squash(capsules,2)
    return capsules

def squash(capsules,axis):
    '''
    :param capsules: Input tensor. Shape is [batch, num_channels, num_atoms] for a
      fully connected capsule layer or [batch, num_channels, num_atoms, height, width] for a convolutional
      capsule layer.
    ：实现了非线性压缩功能
    :return:与输入形状相同
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(capsules), axis,keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm+10e-7)
    vec_squashed = scalar_factor * capsules  # element-wise
    return vec_squashed

    # with tf.name_scope('norm_non_linearity'):
    #     norm = tf.norm(capsules, axis=axis, keep_dims=True)
    #     norm_squared = norm * norm
    #     return (capsules / (norm+10e-7)) * (norm_squared / (1 + norm_squared))


def routing(input,is_training):
    '''
    :param input: 输入，[batch_size,caps1_n_caps,caps2_n_caps,caps2_n_num,1]
    :param num_rout: 迭代次数
    :return: 输出，[batch_size,caps2_n_caps,caps2_n_num]
    '''
    batch_size = input.shape[0].value
    caps1_n_caps = input.shape[1].value
    caps2_n_caps = input.shape[2].value
    b_ij = tf.zeros([1, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32)
    b_ij_tiled = tf.tile(tf.Variable(b_ij,trainable=True),multiples=[batch_size,1,1,1,1])
    input_stop = tf.stop_gradient(input,name='gradient_stop')
    #first
    c_ij = tf.nn.softmax(b_ij_tiled,dim=2)
    s = tf.multiply(c_ij, input_stop)
    s = tf.reduce_sum(s,axis=1,keep_dims=True)
    v = squash(s,axis=-2)

    v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1])
    agreement = tf.matmul(input_stop, v_tiled, transpose_a=True)
    b_ij_tiled = tf.add(agreement, b_ij_tiled)
    c_ij = tf.nn.softmax(b_ij_tiled, dim=2)
    s = tf.multiply(c_ij, input_stop)
    s = tf.reduce_sum(s, axis=1, keep_dims=True)
    v = squash(s, axis=-2)

    v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1])
    agreement = tf.matmul(input_stop, v_tiled, transpose_a=True)
    b_ij_tiled = tf.add(agreement, b_ij_tiled)
    c_ij = tf.nn.softmax(b_ij_tiled, dim=2)
    s = tf.multiply(c_ij, input_stop)
    s = tf.reduce_sum(s, axis=1, keep_dims=True)
    v = squash(s, axis=-2)

    v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1])
    agreement = tf.matmul(input, v_tiled, transpose_a=True)
    b_ij_tiled = tf.add(agreement, b_ij_tiled)
    c_ij = tf.nn.softmax(b_ij_tiled, dim=2)
    s = tf.multiply(c_ij, input)
    s = tf.reduce_sum(s, axis=1, keep_dims=True)
    v = squash(s, axis=-2)
    v = tf.reduce_mean(v,axis=1)
    v = tf.reduce_mean(v,axis=-1)
    v = tf.layers.batch_normalization(inputs=v, training=is_training)

    # for i in range(num_rout):
    #     if i == num_rout-1:
    #         v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1])
    #         agreement = tf.matmul(input, v_tiled, transpose_a=True)
    #         b_ij_tiled = tf.add(agreement, b_ij_tiled)
    #         c_ij = tf.nn.softmax(b_ij_tiled, dim=2)
    #         s = tf.multiply(c_ij, input)
    #         s = tf.reduce_sum(s, axis=1, keep_dims=True)
    #         v = squash(s, axis=-2)
    #     else:
    #         v_tiled = tf.tile(v,[1,caps1_n_caps,1,1,1])
    #         agreement = tf.matmul(input_stop, v_tiled,transpose_a=True)
    #         b_ij_tiled = tf.add(agreement,b_ij_tiled)
    #         c_ij = tf.nn.softmax(b_ij_tiled,dim=2)
    #         s = tf.multiply(c_ij, input_stop)
    #         s = tf.reduce_sum(s, axis=1, keep_dims=True)
    #         v = squash(s, axis=-2)
    #[batch_size,1,caps2_n_caps,caps2_n_num,1]
    return v

def chenge(input,output_dim,output_num,training,name):
    '''
    :param input:shape[batch_size,caps1_n_caps,caps1_n_dims]
    :param output_dim:
    :param output_num:
    :return:[batch_size,caps1_n_caps,caps2_n_caps,caps2_n_dims,1]
    '''
    caps1_n_caps = input.shape[1].value
    caps1_n_dims = input.shape[2].value
    caps2_n_caps = output_num
    caps2_n_dims = output_dim
    init_sigma = 0.01
    W_init = tf.random_normal(shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),stddev=init_sigma, dtype=tf.float32)
    W = tf.Variable(W_init,trainable=True,name='W'+name,dtype=tf.float32)
    W_tited = tf.tile(W,multiples=[batch_size,1,1,1,1])
    caps_1 = input
    caps_1 = tf.expand_dims(caps_1,axis=-1)
    caps_1 = tf.expand_dims(caps_1,axis=2)
    caps1_tiled = tf.tile(caps_1,multiples= [1, 1, caps2_n_caps, 1, 1])
    caps2 = tf.matmul(W_tited, caps1_tiled)
    # caps_2 = tf.reduce_sum(caps2,axis=1)
    # caps_2 = tf.reduce_mean(caps_2,axis=-1)
    caps_2 = tf.layers.batch_normalization(inputs=caps2, training=training)
    return caps_2

def Dynamic_LSTM(input_s1,input_s2,keep_rate,training,name):
    with tf.variable_scope("lst_"+str(name)+"_1"):
        cell_f_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,reuse=tf.AUTO_REUSE)
        cell_b_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,reuse=tf.AUTO_REUSE)
        lstm_output_s1, _ = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,
                                                                   inputs=input_s1,
                                                                   dtype=tf.float32)
        lstm_fw_s1, lstm_bw_s1 = lstm_output_s1
        lstm_output_s1 = tf.concat([lstm_fw_s1, lstm_bw_s1], axis=-1)
    # with tf.variable_scope("lst_"+str(name)+"_2"):
        lstm_output_s2, _ = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,
                                                                   inputs=input_s2,
                                                                   dtype=tf.float32)
        lstm_fw_s2, lstm_bw_s2 = lstm_output_s2
        lstm_output_s2 = tf.concat([lstm_fw_s2, lstm_bw_s2], axis=-1)
        lstm_output_s1 = tf.layers.batch_normalization(inputs=lstm_output_s1, training=training)
        lstm_output_s2 = tf.layers.batch_normalization(inputs=lstm_output_s2, training=training)
    attention_s1,attention_s2 = co_attention(lstm_output_s1,lstm_output_s2)
    concat_s1 = tf.concat([input_s1, lstm_output_s1], axis=-1)
    concat_s2 = tf.concat([input_s2, lstm_output_s2,attention_s2], axis=-1)
    auto_encoder_1 = fully_conacation(concat_s1, hidden_size, keep_rate=keep_rate)
    auto_encoder_2 = fully_conacation(concat_s2, hidden_size, keep_rate=keep_rate)
    decoder_s1_1 = fully_conacation(auto_encoder_1, haddin_size=concat_s1.shape[-1], keep_rate=keep_rate)
    decoder_s2_1 = fully_conacation(auto_encoder_2, haddin_size=concat_s2.shape[-1], keep_rate=keep_rate)
    loss_encoder_s1 = tf.losses.mean_squared_error(concat_s1, decoder_s1_1)
    loss_encoder_s2 = tf.losses.mean_squared_error(concat_s2, decoder_s2_1)
    tf.add_to_collection('losses', loss_encoder_s1)
    tf.summary.scalar('s1_auto_loss'+name,loss_encoder_s1)
    tf.add_to_collection('losses', loss_encoder_s2)
    tf.summary.scalar('s2_auto_loss'+name, loss_encoder_s2)

    return auto_encoder_1,auto_encoder_2

def co_attention(s1,s2):
    # attention构造
    # 计算cosin匹配矩阵
    matrix_1 = tf.matmul(s1, tf.transpose(s2, perm=[0, 2, 1]))
    matrix_2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(s1), axis=-1)), axis=-1)
    matrix_3 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(s2), axis=-1)), axis=1)
    cosin_matrix_s1 = tf.div(matrix_1, tf.matmul(matrix_2, matrix_3))
    # 计算相似矩阵权重
    cosin_matrix_s1 = tf.nn.softmax(cosin_matrix_s1, dim=-1)
    cosin_matrix_s2 = tf.transpose(cosin_matrix_s1, perm=[0, 2, 1])
    cosin_matrix_s2 = tf.nn.softmax(cosin_matrix_s2, dim=-1)
    a_s2 = tf.matmul(cosin_matrix_s1, s2)
    a_s1 = tf.matmul(cosin_matrix_s2, s1)

    return a_s1, a_s2

def fully_conacation(input,haddin_size,training=True,keep_rate=1.0,activation='relu'):
    dense_out = tf.layers.dense(inputs=input, units=haddin_size)
    dense_out = tf.layers.batch_normalization(inputs=dense_out, training=training)
    if activation == 'relu':
        dense_relu = tf.nn.relu(dense_out)
        dense_relu = tf.nn.dropout(dense_relu,keep_prob=keep_rate)
        return dense_relu
    elif activation == 'leaky_relu':
        dense_relu = tf.nn.leaky_relu(dense_out)
        dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
        return dense_relu
    elif activation == 'sigmoid':
        dense_relu = tf.nn.sigmoid(dense_out)
        dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
        return dense_relu
    elif activation == 'None':
        dense_relu = tf.nn.dropout(dense_out, keep_prob=keep_rate)
        return dense_relu

def margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.
    Each wrong logit costs its distance to margin. For negative logits margin is
    0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.
    Args:
    labels: tensor, one hot encoding of ground truth.
    raw_logits: tensor, model predictions in range [0, 1]
    margin: scalar, the margin after subtracting 0.5 from raw_logits.
    downweight: scalar, the factor for negative cost.
    Returns:
    A tensor with cost for each data point of shape [batch_size].
    """

    m_plus =0.9
    m_minus =0.1
    lambda_ =0.5
    T = tf.one_hot(labels, depth=2, name="T")
    v_norm = raw_logits
    FP_raw = tf.square(tf.maximum(0., m_plus - v_norm), name="FP_raw")
    FN_raw = tf.square(tf.maximum(0., v_norm - m_minus), name="FN_raw")
    L = tf.add(T * FP_raw, lambda_ * (1.0 - T) * FN_raw, name="L")
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

    # label = tf.one_hot(labels,depth=2)
    # logits = raw_logits - 0.5
    # positive_cost = label * tf.cast(tf.less(logits, margin),tf.float32) * tf.pow(logits - margin, 2)
    # negative_cost = (1 - label) * tf.cast(tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    # loss_tensor = 0.5 * positive_cost + downweight * 0.5 * negative_cost
    # loss = tf.reduce_mean(tf.reduce_sum(loss_tensor,axis=-1))
    return margin_loss


def drop_out():
    pass