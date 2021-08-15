#python split_data.py --img_dir=/mnt/sdc/craig/Guizhou_NIR_dataset_kfr --save_dir=/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_tmp/ --train_num=926


import argparse
import os
import shutil
import random
from random import shuffle
random.seed(0)

def argparse_fun():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_num',type=int, default= 926)
    parser.add_argument('--img_dir',type=str, default='/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304')
    parser.add_argument('--save_dir',type=str, default='/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval')
    args = parser.parse_args()
    return args

def move_data_fun(img_dir_list_,data_tye,args):
    for img_dir_i in img_dir_list_:
        src_dir = os.path.join(args.img_dir, img_dir_i)
        dst_dir = os.path.join(args.save_dir, data_tye, img_dir_i)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        for file_i in os.listdir(src_dir):
            src_path = os.path.join(src_dir, file_i)
            dst_path = os.path.join(dst_dir, file_i)
            shutil.copyfile(src_path, dst_path)

def split_data_fun(args):
    img_dir_list = os.listdir(args.img_dir)
    img_dir_list_shuffle = sorted(img_dir_list, key=lambda k: random.random())
    train_num = args.train_num
    img_dir_list_train = img_dir_list_shuffle[:train_num]
    img_dir_list_val = img_dir_list_shuffle[train_num:]
    #print(len(img_dir_list_shuffle),len(img_dir_list_train),len(img_dir_list_val))
    move_data_fun(img_dir_list_train,'train',args)
    move_data_fun(img_dir_list_val,'val',args)


if __name__ == '__main__':

    args = argparse_fun()
    split_data_fun(args)
