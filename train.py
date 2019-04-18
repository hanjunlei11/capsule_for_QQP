from model import *
from tool import *
from config import *
import tensorflow as tf
s1_word_train,s1_word_test,s2_word_train,s2_word_test,vector_lines,label_train,label_test= read_file(s1path='./data/s1.txt',s2path='./data/s2.txt',labelpath='./data/label.txt',re_vector='./data/vector.txt')
Model = Model(vector_lines)
print('1、构造模型完成')
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Model.all_loss)
print('2、load data完成')
saver = tf.train.Saver(max_to_keep=3)
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init_op)
    # ckpt = tf.train.get_checkpoint_state('./ckpt/')
    # saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    print('3、初始化完成')
    print('4、开始训练')
    max_acc=0
    for i in range(100000):
        batch_s1, batch_s2, batch_label,random_int = get_batch(batch_size=batch_size, s1=s1_word_train, s2=s2_word_train,label=label_train)
        feed_dic = {Model.s1:batch_s1,Model.s2:batch_s2,Model.label:batch_label,Model.keep_rate:0.8,Model.is_training:True}
        _,rs,loss, acc,chack_point=sess.run([opt_op,merged,Model.all_loss,Model.acc,Model.s1_matrix_tr],feed_dict=feed_dic)
        # print(chack)
        writer.add_summary(rs, i)
        print(i+1,'次训练 ','loss: ','%.7f'%loss,'acc: ','%.7f'%acc,'max:','%.7f'%max_acc)
        if (i+1)%100==0:
            all_acc = 0
            all_loss = 0
            for j in range(20):
                batch_s1, batch_s2, batch_label,random_int = get_batch(batch_size=batch_size, s1=s1_word_test, s2=s2_word_test, label=label_test)
                feed_dic = {Model.s1: batch_s1, Model.s2: batch_s2, Model.label: batch_label, Model.keep_rate: 1.0,Model.is_training:False}
                loss, acc= sess.run([Model.all_loss, Model.acc],feed_dic)
                all_acc+=acc
                all_loss+=loss
                print('test losses: ','%.7f'%loss,' ','accuracy: ','%.7f'%acc)
            all_acc = all_acc/20.0
            all_loss = all_loss/20.0
            if all_acc > max_acc:
                max_acc = all_acc
                saver.save(sess, save_path='ckpt_retrain/model.ckpt', global_step=i + 1)
            print('第',int((i+1)/100),'次测试 ','losses: ','%.7f'%all_loss, 'accuracy: ', '%.7f'%all_acc)