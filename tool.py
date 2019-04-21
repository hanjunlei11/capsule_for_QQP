import numpy as np
import re
from config import *
# -*- coding: utf-8 -*-
import csv
import collections
import random

# file = open('./test.txt','w+',encoding='utf-8')

def get_data(filepath):
    with open(filepath,'r',encoding='utf-8',errors='ignore') as data,open('./data/s1.txt','w+', encoding='ANSI',errors='ignore') as s1, open('./data/s2.txt', 'w+', encoding='ANSI',errors='ignore') as s2, open('./data/label.txt', 'w+', encoding='ANSI',errors='ignore') as label, open('./data/word2index.txt','w+',encoding='ANSI',errors='ignore') as word_index,open('./data/vector.txt','w+',encoding='utf-8',errors='ignore') as vector,open('./data/out_of_vecab.txt','w+',encoding='utf-8',errors='ignore') as out_vecab:
        data_lines = csv.reader(data)
        # stop_words = stopwords.readlines()
        r1 = "[\�\!\_,$\"%^*(+]+|[+！\”\“，\‘。？、~@#￥%&*（）;\’?:`>><()]+"
        word2index = collections.OrderedDict()
        embedding = collections.OrderedDict()
        out_of_vocab = collections.OrderedDict()
        embedding_table = load_vector('./data/glove.840B.300d.txt')
        s = 0
        for i in data_lines:
            s+=1
            print('第',s,'行')
            ss1 = i[3].lower()
            ss2 = i[4].lower()
            # ss1 = ss1.replace('\'','')
            # ss2 = ss2.replace('\'','')
            ss1 = re.sub(r1,'',ss1)
            ss2 = re.sub(r1,'',ss2)
            ss1 = re.split(pattern=r'[./\s]', string=ss1)
            ss2 = re.split(pattern=r'[./\s]',string=ss2)
            ss1 = ['[BIN]']+ss1
            ss2 = ['[SEP]']+ss2
            if len(ss1)<=2:
                continue
            if len(ss2)<=2:
                continue
            if len(i[5]) == 0:
                continue
            else:
                label.write(i[5]+'\n')
            for j in range(len(ss1)):
                if ss1[j] not in embedding_table:
                    if ss1[j] not in out_of_vocab:
                        out_of_vocab[ss1[j]] = len(out_of_vocab)
                    embedding[ss1[j]] = np.random.uniform(-0.25000,0.25000,300)
                else:
                    embedding[ss1[j]] = embedding_table[ss1[j]]
            for j in range(len(ss2)):
                if ss2[j] not in embedding_table:
                    if ss2[j] not in out_of_vocab:
                        out_of_vocab[ss2[j]] = len(out_of_vocab)
                    embedding[ss2[j]] = np.random.uniform(-0.25000,0.25000,300)
                else:
                    embedding[ss2[j]] = embedding_table[ss2[j]]

            for item in embedding.items():
                if item[0] not in word2index:
                    word2index[item[0]] = len(word2index)
            for j in range(len(ss1)):
                if ss1[j] in word2index:
                    ss1[j] = str(word2index[ss1[j]])
            for j in range(len(ss2)):
                if ss2[j] in word2index:
                    ss2[j] = str(word2index[ss2[j]])
            s1.write(' '.join(ss1) + '\n')
            s2.write(' '.join(ss2) + '\n')

        print(len(out_of_vocab))
        print(len(word2index))
        for k, v in embedding.items():
            vector.write(' '.join([str(i) for i in v])+'\n')
        for item in word2index.items():
            word_index.write(item[0]+' '+str(item[1])+'\n')
        for item in out_of_vocab.items():
            out_vecab.write(item[0]+' '+str(item[1])+'\n')

def get_epoch(batch_size,s1,s2,label):
    data_s1 = s1
    data_s2 = s2
    data_label = label
    epoch_s1 = []
    epoch_s2 = []
    epoch_label = []
    list_int = [i for i in range(len(data_label))]
    while len(list_int)>batch_size:
        random_int = random.sample(list_int, batch_size)
        batch_s1 = np.asarray(data_s1)[random_int]
        batch_s2 = np.asarray(data_s2)[random_int]
        batch_label = np.asarray(data_label)[random_int]
        random_int = list(random_int)
        random_int.sort(reverse=True)
        for i in random_int:
            list_int.remove(i)
        epoch_s1.append(batch_s1)
        epoch_s2.append(batch_s2)
        epoch_label.append(batch_label)
    return epoch_s1,epoch_s2,epoch_label,len(epoch_s1)

def get_batch(s1, s2, label, i):
    batch_s1 = s1[i]
    batch_s2 = s2[i]
    batch_label = label[i]
    return batch_s1, batch_s2, batch_label

def read_file(s1path, s2path, labelpath,re_vector):
    f_s1 = open(s1path, 'r', encoding='utf-8')
    f_s2 = open(s2path, 'r', encoding='utf-8')
    f_quality = open(labelpath, 'r', encoding='utf-8')
    f_vector = open(re_vector,'r',encoding='utf-8')
    s1_lines = []
    # s1_len = []
    # s2_len = []
    for line in f_s1.readlines():
        temp = [77]*batch_len
        line = line.strip('\n').split()
        # if len(line)>batch_len:
        #     s1_len.append(batch_len)
        # else:
        #     s1_len.append(len(line))
        for i in range(len(line)):
            if i < batch_len:
                temp[i] = int(line[i])
        s1_lines.append(temp)
    s2_lines = []
    for line in f_s2.readlines():
        temp = [77] * batch_len
        line = line.strip('\n').split()
        # if len(line)>batch_len:
        #     s2_len.append(batch_len)
        # else:
        #     s2_len.append(len(line))
        for i in range(len(line)):
            if i < batch_len:
                temp[i] = int(line[i])
        s2_lines.append(temp)
    quality_lines = []
    for line in f_quality.readlines():
        line = line.strip('\n')
        quality_lines.append(int(line))
    vector_lines = []
    for line in f_vector.readlines():
        temp2 = [float(i) for i in line.strip().split(' ')]
        if len(temp2)==embadding_size:
            vector_lines.append(temp2)
    s1_word_train = s1_lines[0:int(len(s1_lines)*0.8)]
    s1_word_test = s1_lines[int(len(s1_lines)*0.8):]
    s2_word_train = s2_lines[0:int(len(s2_lines) * 0.8)]
    s2_word_test = s2_lines[int(len(s2_lines) * 0.8):]
    label_train = quality_lines[0:int(len(quality_lines)*0.8)]
    label_test = quality_lines[int(len(quality_lines)*0.8):]
    # s1_len_train = s1_len[0:int(len(s1_len)*0.8)]
    # s1_len_test = s1_len[int(len(s1_len)*0.8):]
    # s2_len_train = s2_len[0:int(len(s2_len) * 0.8)]
    # s2_len_test = s2_len[int(len(s2_len) * 0.8):]
    return s1_word_train,s1_word_test,s2_word_train,s2_word_test,vector_lines,label_train,label_test

def load_vector(filepath):
    with open(filepath,'r',encoding='utf-8') as glove_vector:
        embadding_table = collections.OrderedDict()
        count = 0
        for line in glove_vector.readlines():
            print(count)
            count+=1
            temp = []
            line = line.split(' ')
            for j in range(len(line)):
                if j != 0:
                        temp.append(float(line[j]))
            if line[0] not in embadding_table:
                embadding_table[line[0]] = temp
    return embadding_table

def get_char(s1_path,s2_path):
    with open(s1_path,'r',encoding='utf-8',errors='ignore') as f1,open(s2_path,'r',encoding='utf-8',errors='ignore') as f2,open('./char2index.txt','w+',encoding='utf-8',errors='ignore') as char2index:
        data_lines_s1 = f1.readlines()
        data_lines_s2 = f2.readlines()
        s1 = []
        s2 = []
        char_index = {}
        char_index[','] = 0
        for i in range(len(data_lines_s1)):
            ss1 = data_lines_s1[i]
            ss2 = data_lines_s2[i]
            ss1 = ss1.strip().split()
            ss2 = ss2.strip().split()
            s1_word = []
            s2_word = []
            temp = [',']*batch_len
            for j in range(len(ss1)):
                if j < batch_len:
                    temp[j] = ss1[j]
            for j in temp:
                temp_char = [0]*word_length
                word_char = []
                for k in j:
                    word_char.append(k)
                    if k not in char_index:
                        char_index[k] = len(char_index)
                for k in range(len(word_char)):
                    if k < word_length:
                        word_char[k] = char_index[word_char[k]]
                        temp_char[k] = word_char[k]
                s1_word.append(temp_char)
            s1.append(s1_word)

            temp = [','] * batch_len
            for j in range(len(ss2)):
                if j < batch_len:
                    temp[j] = ss2[j]
            for j in temp:
                temp_char = [0] * word_length
                word_char = []
                for k in j:
                    word_char.append(k)
                    if k not in char_index:
                        char_index[k] = len(char_index)
                for k in range(len(word_char)):
                    if k < word_length:
                        word_char[k] = char_index[word_char[k]]
                        temp_char[k] = word_char[k]
                s2_word.append(temp_char)
            s2.append(s2_word)
        for item in char_index.items():
            char2index.write(item[0] + ' ' + str(item[1]) + '\n')
        s1_char_train = s1[0:int(len(s1)*0.8)]
        s1_char_test = s1[int(len(s1)*0.8):]
        s2_char_train = s2[0:int(len(s1) * 0.8)]
        s2_char_test = s2[int(len(s1) * 0.8):]
        return s1_char_train,s1_char_test, s2_char_train,s2_char_test

# def get_epoch():
# s1_char_train,s1_char_test, s2_char_train,s2_char_test = get_char('./first questions.csv')
# s1_char_train,s1_char_test, s2_char_train,s2_char_test = get_char('./QQP/s1.txt','./QQP/s2.txt')
# get_data('./data/first questions.csv')
# s1_word_train,s1_word_test,s2_word_train,s2_word_test,vector_lines,label_train,label_test= read_file(s1path='./data/s1.txt',s2path='./data/s2.txt',labelpath='./data/label.txt',re_vector='./data/vector.txt')
# print(np.shape(vector_lines))