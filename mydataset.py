#对原始数据进行划分，得到标记文件

import os
import shutil
import collections
import math
import csv
import random

root_dir = '/home/ding/DATA/dataset/'
valid_ratio = 0.1
batch_size = 4

data_train_dir = os.path.join(root_dir,'train')

#随机字典
def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

#建立CSV标记文件
def create_csv_labels(data_train_dir,csv_dir = os.path.join(data_train_dir,'../trainLabels.csv')):
    for root,dir,files in os.walk(data_train_dir):
        tokens = [name.rstrip().split('.') for name in files]
        data_label_list = [label for label,id,suffix in tokens]
        data_id_list = [(label+'.'+ id) for label,id,suffix in tokens]
        csv_dict = dict(zip(data_id_list,data_label_list))
        csv_dict = random_dic(csv_dict)
        #print(csv_dict.items())
        with open(csv_dir,'w') as f:
            csv_write = csv.writer(f)
            csv_head = ["id","label"]
            csv_write.writerow(csv_head)
            for id,label in csv_dict.items():
                #print(id,label)
                csv_write.writerow([id,label])
#        for name in files:
#            print(name.rstrip().split('.',1))
labels_map = {
    0: "cat",
    1: "dog",
}

#建立CSV标记文件V2
def create_csv_labels_v2(data_train_dir,csv_dir = os.path.join(data_train_dir,'../trainLabels1.csv')):
    for root,dir,files in os.walk(data_train_dir):
        tokens = [name.rstrip().split('.') for name in files]
        data_label_list = [label for label,id,suffix in tokens]
        data_id_list = [(label+'.'+ id+'.'+suffix) for label,id,suffix in tokens]
        csv_dict = dict(zip(data_id_list,data_label_list))
        csv_dict = random_dic(csv_dict)
        #print(csv_dict.items())
        with open(csv_dir,'w') as f:
            csv_write = csv.writer(f)
            csv_head = ["id","label"]
            csv_write.writerow(csv_head)
            for id,label in csv_dict.items():
                #print(id,label)
                csv_write.writerow([id,label])
#        for name in files:
#            print(name.rstrip().split('.',1))

#读取标记字典
def read_csv_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


def copyfile(filename,target_dir):
    os.makedirs(target_dir,exist_ok = True)
    shutil.copy(filename,target_dir)

#划分训练 校验集
def reorg_train_valid(data_dir, labels, valid_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        l = train_file.split('.')
        label = labels[l[0]+'.'+l[1]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(
            fname,
            os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(
                fname,
                os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label

#划分测试集
def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(
            os.path.join(data_dir, 'test', test_file),
            os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_catsanddog_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

valid_ratio = 0.1
#create_csv_labels_v2(data_train_dir)
#reorg_catsanddog_data(root_dir,valid_ratio)


