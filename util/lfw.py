"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import numpy as np
import sys

from sklearn.model_selection import KFold
from scipy import interpolate
from sklearn import metrics
from scipy.optimize import brentq
from datetime import datetime

import cv2
import pickle
import joblib
from tqdm import tqdm
from os.path import basename
import onnxruntime
import shutil

import torch, torchvision
import imgaug as ia
import imgaug.augmenters as iaa


ia.seed(1)

val_seq_mix = iaa.Sequential([

    iaa.OneOf([
                    iaa.Multiply((0.8, 1.2), per_channel=False),
                    iaa.Multiply((0.4, 0.6), per_channel=False),
                    iaa.Multiply((1.4, 1.6), per_channel=False),
                ]),
], random_order=True) # apply augmenters in random order

val_seq = iaa.Sequential([

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        #iaa.GaussianBlur(sigma=(0, 0.5))
        iaa.GaussianBlur(sigma=(0, 1))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.6, 1.4)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=False),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.6, 1.4), per_channel=False),
], random_order=True) # apply augmenters in random order


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

mix_transform = [torchvision.transforms.ColorJitter(brightness=0.2), torchvision.transforms.ColorJitter(brightness=(0.3, 0.6)), torchvision.transforms.ColorJitter(brightness=(1.3, 1.6))]

affine_transform = [ torchvision.transforms.RandomAffine(5, translate=None, scale=None, shear=10, resample=False, fillcolor=0)]
image_transforms = {

    'kneron-gray-mix': torchvision.transforms.Compose([
        #torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomChoice(mix_transform),
        #torchvision.transforms.RandomAffine(5, translate=None, scale=None, shear=10, resample=False, fillcolor=0),
    ]),
}
data_transforms = {
    'PIL-kneron': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[1.0, 1.0, 1.0], mean=[0.5, 0.5, 0.5])
    ]),
    'kneron-PIL': torchvision.transforms.Compose([
        torchvision.transforms.Normalize(std=[1.0, 1.0, 1.0], mean=[-0.5, -0.5, -0.5]),
        torchvision.transforms.ToPILImage()
    ]),
    'PIL-tf': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
    ]),
    'CV-kneron': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[256., 256., 256.], mean=[128, 128, 128]),
        #torchvision.transforms.RandomErasing(),
        #torchvision.transforms.Normalize(std=[255., 255., 255.], mean=[0, 0, 0]),
        #torchvision.transforms.Normalize(std=[1.0, 1.0, 1.0], mean=[0.5, 0.5, 0.5])
    ]),
    'CV-tf': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[127.5, 127.5, 127.5], mean=[127.5, 127.5, 127.5])
    ])
}

#from .utils import normalization
def normalization(X):
    return X / 256. - 0.5

preprocess = normalization
far_val = 1e-3


def evaluate(embeddings, actual_issame, visfpfn, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, fnfpids = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), far_val, visfpfn=visfpfn , nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far, fnfpids


def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


#'''
def load_folder(file_data_list, file_issame_list, flip=False, image_size=(112,112)):
    '''
    file_data_list = '/mnt/sdd/johnnysun/data/FR/gz_val_data_list.pickle'
    file_issame_list = '/mnt/sdd/johnnysun/data/FR/gz_val_issame_list.pickle'
    with open(file_data_list,'rb') as f:
        data_list = pickle.load(f)

    with open(file_issame_list,'rb') as f:
        issame_list = pickle.load(f)
    '''
    '''
    pair_folder_name = '/mnt/sdd/johnnysun/data/FR/henan_gz_val'
    #
    same_txtfile_name = 'validate_same_pairs_hn_gz_50'
    diff_txtfile_name = 'validate_diff_pairs_hn_gz_50'


    flipflag = [0] # if flip use [0,1]
    image_list = []
    with open( os.path.join(pair_folder_name, same_txtfile_name+'.txt' ),'r') as fp:
        for line in fp.readlines():
            line = line.split('\t')
            image_list.append([line[0], line[1], 1])
            #print(image_list)

    with open(os.path.join(pair_folder_name, diff_txtfile_name+'.txt' ),'r') as fp:
        for line in fp.readlines():
            line = line.split('\t')
            image_list.append([line[0], line[1], 0])

    #print("image_list: ", len(image_list))
    #print("image_list[6000]: ", image_list[6000])
    np.random.shuffle(image_list)

    data_list = []
    issame_list = []
    for flip in flipflag:
        data = np.empty((len(image_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(image_list)):
        img1_path = image_list[i][0]
        img2_path = image_list[i][1]
        #print(img1_path)
        #print(img2_path)
        issame = image_list[i][2]
        #print(issame)
        issame_list.append(issame)
        #sys.exit(1)
        #print ("img1_path: ", img1_path)
        #print ("img2_path: ", img2_path)
        #print ("issame: ", issame)
        #img0 = cv2.imdecode()
        img1_path = img1_path.rstrip()
        img2_path = img2_path.rstrip()
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        #print("img1_path: ", img1_path)
        #print("img2_path: ", img2_path)
        #print("img1_path: ", os.path.isfile(img1_path))
        #print("img2_path: ", os.path.isfile(img2_path.rstrip()))

        if img1.shape[1]!=image_size[0]:
            img1 = cv2.resize(img1, image_size)
        if img2.shape[1]!=image_size[0]:
            img2 = cv2.resize(img2, image_size)

        for flip in flipflag:
            if flip==1:
                img1 = np.flip(img1, axis=1)
                img2 = np.flip(img2, axis=1)
            data_list[flip][2*i+0] = img1
            data_list[flip][2*i+1] = img2
        if i%1000==0:
            print('loading folder', i)

    #print(folder_name.split('/')[-1], data_list[0].shape)
    #"""db
    print( 'total len of data_list:', len(data_list[0]))

    return (data_list, issame_list), image_list
    '''


    #file_data_list = '/mnt/sdd/johnnysun/data/FR/henan_gz_val/henan_gz_val_data_50.pkl'
    #file_issame_list = '/mnt/sdd/johnnysun/data/FR/henan_gz_val/henan_gz_val_issame_50.pkl'
    with open(file_data_list,'rb') as f:
        data_list = pickle.load(f)

    with open(file_issame_list,'rb') as f:
        issame_list = pickle.load(f)
    if flip:
        print( 'Apply Flip Images.')
        #print( data_list[0].shape)
        #print( type(issame_list) )
        amt, h,w,c =  data_list[0].shape
        data_list_flip = np.zeros( (amt*2, h,w,c) )
        progress = tqdm(total=amt)
        for i, data in enumerate(data_list[0]):
            #print(data.shape)
            progress.update(1)
            data_list_flip[i] = data
            flipdata = np.flip(data, axis=1)
            data_list_flip[i+amt] = flipdata
        #flip_merge = np.concatenate( (data_list[0], data_list_flip))
        #print(flip_merge.shape)
        issame_list = issame_list*2
        #sys.exit(1)
        data_list = [data_list_flip]


    print( f'Val set:{basename(file_data_list)}, pair amount: {len(data_list[0])//2}, {len(issame_list)}')
    return (data_list, issame_list)

#'''
'''
def load_folder(img_folder_name, pair_folder_name,image_size=(112,112)):
    #bins, issame_list = pickle.load(open(bins_filename, 'rb'))
    #bins, issame_list = pickle.load(open(bins_filename, 'rb', encoding='latin1'))
    image_list = []
    with open( os.path.join(pair_folder_name, "validate_same_pairs_tpe_street.txt"),'r') as fp:
        for line in fp.readlines():
            line = line.split('\t')
            image_list.append([line[0], line[1], 1])


    with open(os.path.join(pair_folder_name, "validate_diff_pairs_tpe_street.txt"),'r') as fp:
        for line in fp.readlines():
            line = line.split('\t')
            image_list.append([line[0], line[1], 0])

    #print("image_list: ", len(image_list))
    #print("image_list[6000]: ", image_list[6000])
    #exit(0)
    np.random.shuffle(image_list)

    data_list = []
    issame_list = []
    for flip in [0,1]:
        data = np.empty((len(image_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(image_list)):
        img1_path = os.path.join(img_folder_name, image_list[i][0])
        img2_path = os.path.join(img_folder_name, image_list[i][1])
        issame = image_list[i][2]
        issame_list.append(issame)
        #print ("img1_path: ", img1_path)
        #print ("img2_path: ", img2_path)
        #print ("issame: ", issame)
        #img0 = cv2.imdecode()
        img1_path = img1_path.rstrip()
        img2_path = img2_path.rstrip()
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        #print("img1_path: ", img1_path)
        #print("img2_path: ", img2_path)
        #print("img1_path: ", os.path.isfile(img1_path))
        #print("img2_path: ", os.path.isfile(img2_path.rstrip()))

        if img1.shape[1]!=image_size[0]:
            img1 = cv2.resize(img1, image_size)
        if img2.shape[1]!=image_size[0]:
            img2 = cv2.resize(img2, image_size)
        for flip in [0,1]:
            if flip==1:
                img1 = np.flip(img1, axis=1)
                img2 = np.flip(img2, axis=1)
            data_list[flip][2*i+0] = img1
            data_list[flip][2*i+1] = img2
        if i%1000==0:
            print('loading folder', i)

    #print(folder_name.split('/')[-1], data_list[0].shape)
    """
    file_data_list = '/mnt/storage1/craig/kneron_mask_fr/tpe_street_val_data_list.pickle'
    file_issame_list = '/mnt/storage1/craig/kneron_mask_fr/tpe_street_val_issame_list.pickle'
    with open(file_data_list,'wb') as f:
        #data_list = pickle.load(f)
        pickle.dump(data_list, f,protocol = 4)

    with open(file_issame_list,'wb') as f:
        #issame_list = pickle.load(f)
        pickle.dump(issame_list, f,protocol = 4)
    print("write complete")
    """
    return (data_list, issame_list)
'''

def load_bin(bins_filename,image_size=(112,112), to_grayscale=False):
    #bins, issame_list = pickle.load(open(bins_filename, 'rb'))
    #bins, issame_list = pickle.load(open(bins_filename, 'rb', encoding='latin1'))
    with open(bins_filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        bins, issame_list = u.load()

    data_list = []
    for flip in [0,1]:
        data = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = cv2.imdecode(np.frombuffer(_bin, dtype=np.uint8), flags=1)[...,::-1]
        if to_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.shape[1]!=image_size[0]:
            img = cv2.resize(img, image_size)

        for flip in [0,1]:
            if flip==1:
                img = np.flip(img, axis=1)
            data_list[flip][i] = img
        if i%1000==0:
            print('loading bin', i)
    print(bins_filename.split('/')[-1], data_list[0].shape)
    return (data_list, issame_list)


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        #print("thresholds[best_threshold_index]: ", thresholds[best_threshold_index])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, visfpfn, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    #
    thres = np.zeros(nrof_folds)
    ta = np.zeros(nrof_folds)
    fa = np.zeros(nrof_folds)
    nsame = np.zeros(nrof_folds)
    ndiff = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    #print('distance:', dist[:20])
    #print('negative distance:', dist[-20:])
    #print (embeddings1[0][:10])
    #print (embeddings1[-1][:10])
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx],_,_,_,_,_ = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx], thres[fold_idx],ta[fold_idx],fa[fold_idx],nsame[fold_idx],ndiff[fold_idx] \
                    = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    print('Test_set len:', len(test_set), 'nrof_folds:',nrof_folds)
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    # print fp/fn
    threshold_m = np.mean(thres)
    true_accept_m = np.sum(ta)
    false_accept_m = np.sum(fa)
    n_same_m = np.sum(nsame)
    n_diff_m = np.sum(ndiff)
    #
    print( f'Threshold:{threshold_m:.4f}, True_accept:{true_accept_m}/n_same:{n_same_m}, False_accept:{false_accept_m}/n_diff:{n_diff_m}')

    # visualized fp/fn
    fnfpids = []
    if visfpfn:
        print(f'Save visfpfn in {visfpfn}')
        fn_pair_id, fp_pair_id = calculate_val_far_fpfn_index( threshold_m, dist[:nrof_pairs], actual_issame[:nrof_pairs] )
        fnfpids = [fn_pair_id, fp_pair_id]


    return val_mean, val_std, far_mean, fnfpids


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    # fn
    fn_pair_id = np.logical_and( np.logical_not(predict_issame), actual_issame )
    # fp
    fp_pair_id = np.logical_and( np.logical_not(predict_issame), np.logical_not(actual_issame) )

    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    #print(threshold, predict_issame, true_accept, false_accept, n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far, threshold, true_accept, false_accept, n_same, n_diff


def calculate_val_far_fpfn_index(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    # fn
    fn_pair_id = np.logical_and( np.logical_not(predict_issame), actual_issame )
    fn_pair_id = np.argwhere(fn_pair_id>0).reshape(-1)
    # fp
    fp_pair_id = np.logical_and( predict_issame, np.logical_not(actual_issame) )
    fp_pair_id = np.argwhere(fp_pair_id>0).reshape(-1)
    #fp_pair_id = np.argwhere(np.logical_not(actual_issame)>0).reshape(-1)
    print(threshold)
    print( 'fn')
    print( fn_pair_id)
    print( dist[fn_pair_id])
    print( actual_issame[fn_pair_id])
    print( 'fp')
    print( fp_pair_id)
    print( dist[fp_pair_id])
    print( actual_issame[fp_pair_id])

    return fn_pair_id, fp_pair_id


def evaluate_keras(model, data_list, actual_issame, nrof_folds, log_dir=None, step=0,
                   img_size=(112,112,3),  batch_size=32, name='', flip=True, **kwargs):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print ('\nRunning forward pass on {} images: '.format(name),end='')
    nrof_images = len(actual_issame)*2

    if flip:
        print (' with adding flipped image')
        embeddings_list = []
        for i in range(len(data_list)):
            data = data_list[i]
            data = preprocess(data)
            emb = model.predict(data, batch_size=batch_size, verbose=1)
            emb = emb / (np.linalg.norm(emb, axis=-1)+1e-10).reshape(-1,1)
            embeddings_list.append(emb)
        embeddings = embeddings_list[0] + embeddings_list[1]
    else:
        data = data_list[0]
        data = preprocess(data)
        embeddings = model.predict(data, batch_size=batch_size, verbose=1)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)+1e-10).reshape(-1,1)

    embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)).reshape(-1,1)

    print('duration:', '%.3f' % (time.time()-start_time), 'seconds, embedding shape:', embeddings.shape)
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, actual_issame, nrof_folds=nrof_folds)
    print('Accuracy: %1.5f+-%1.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.5f' % auc)
    # remove EER due to bug: A value in x_new is above the interpolation range. #Long#
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.5f' % eer)
    if log_dir:
        with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
            f.write('%s %d\t%.5f\t%.5f\t%.5f\n' % (name, step, np.mean(accuracy), val, far))
    return np.mean(accuracy), val

def draw_fnfpids(fnfpids, idata, visfpfn, samplenumb=20):
    fn_ids = fnfpids[0]
    fp_ids = fnfpids[1]

    if os.path.exists( visfpfn):
        shutil.rmtree( visfpfn )
    os.makedirs( visfpfn+'/fn')
    os.makedirs( visfpfn+'/fp')

    for idx, fnid in enumerate(fn_ids):
        if idx==samplenumb:
            break
        data1, data2 = idata[fnid*2], idata[fnid*2+1]
        cv2.imwrite( f'{visfpfn}/fn/{idx}_1.png' ,data1 )
        cv2.imwrite( f'{visfpfn}/fn/{idx}_2.png' ,data2 )

    for idx, fpid in enumerate(fp_ids):
        if idx==samplenumb:
            break
        data1, data2 = idata[fpid*2], idata[fpid*2+1]
        cv2.imwrite( f'{visfpfn}/fp/{idx}_1.png' ,data1 )
        cv2.imwrite( f'{visfpfn}/fp/{idx}_2.png' ,data2 )
    return


def evaluate_pytorch(model, data_list, actual_issame, nrof_folds, log_dir=None, log_file=None, step=0,
                   img_size=(112,112,3), batch_size=200, name='', visfpfn=False, flip=True, samplenumb=20,
                   data_transform_id='CV-kneron', **kwargs):

    model = model.to(device)
    model.eval()

    start_time = time.time()
    # Run forward pass to calculate embeddings
    print ('\nRunning forward pass on {} images: '.format(name), end='')
    nrof_images = len(actual_issame)*2

    if flip:
        print (' with adding flipped image')
        embeddings_list = []
        for i in range(len(data_list)):
            idata = data_list[i]
            num_batches = int(len(idata)/batch_size)
            data = torch.zeros([batch_size, 3, img_size[0], img_size[1]], dtype=torch.float, device=device)
            for b in range(num_batches):
                for img in range(batch_size):
                    #buffer = np.float32(idata[img+b*batch_size])
                    #images_aug = image_transforms['kneron-gray-mix'](buffer)
                    #images_aug = val_seq_mix(image=buffer)
                    #data[img] = data_transforms[data_transform_id](images_aug)
                    data[img] = data_transforms[data_transform_id](idata[img+b*batch_size])

                with torch.no_grad():
                    #backbone_output = backbone(data)
                    #output = head(backbone_output)
                    output = model(data)
                    emb_in_batch = output.cpu().numpy()

                if b == 0:
                    emb_per_list = emb_in_batch.copy()
                else:
                    emb_per_list = np.concatenate((emb_per_list, emb_in_batch))

            #emb_per_list = emb_per_list / (np.linalg.norm(emb_per_list, axis=-1)+1e-10).reshape(-1,1)
            embeddings_list.append(emb_per_list)

        # add addtional individual normalized (may lower accuracy but increase verification)
        # embeddings = (embeddings_list[0] / (np.linalg.norm(embeddings_list[0], axis=-1)).reshape(-1,1)) + \
        #        (embeddings_list[1] / (np.linalg.norm(embeddings_list[1], axis=-1)).reshape(-1,1))
        embeddings = embeddings_list[0] + embeddings_list[1]
    else:
        print ()
        idata = data_list[0]
        print( 'idata:', idata.shape)
        num_batches = int(len(idata)/batch_size)
        for b in range(num_batches):
            data = torch.zeros([batch_size, 3, img_size[0], img_size[1]], dtype=torch.float, device=device)
            for img in range(batch_size):
                data[img] = data_transforms[data_transform_id]( idata[img+b*batch_size] )

            with torch.no_grad():
                #backbone_output = backbone(data)
                #output = head(backbone_output)
                output = model(data)
                emb_in_batch = output.cpu().numpy()

            emb_in_batch = emb_in_batch / (np.linalg.norm(emb_in_batch, axis=-1)+1e-10).reshape(-1,1)
            if b == 0:
                embeddings = emb_in_batch.copy()
            else:
                embeddings = np.concatenate((embeddings, emb_in_batch))
        print( 'embeddings shapes', embeddings.shape )
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)+1e-10).reshape(-1,1)

    embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)).reshape(-1,1)

    print('Duration:', '%.3f' % (time.time()-start_time), 'seconds, embedding shape:', embeddings.shape)
    tpr, fpr, accuracy, val, val_std, far, fnfpids = evaluate(embeddings, actual_issame, visfpfn=visfpfn, nrof_folds=nrof_folds)
    # fnfpids
    if visfpfn:
        draw_fnfpids( fnfpids, idata, visfpfn, samplenumb=samplenumb)


    print('Accuracy: %1.5f+-%1.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.5f' % auc)
    # remove EER due to bug: A value in x_new is above the interpolation range. #Long#
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.5f' % eer)

    log_str = '%s \tACC: %.5f\tVAL: %.5f\t@FAR: %.5f\tEpoch: %d\t%s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), np.mean(accuracy), val, far, step, name)
    print(log_str)
    if log_dir and log_file:
        with open(os.path.join(log_dir, log_file),'at') as f:
            #f.write('%s \tACC: %.5f\tVAL: %.5f\t@FAR: %.5f\tEpoch: %d\t%s \n' % (str(datetime.now()), np.mean(accuracy), val, far, step, name))
            f.write(log_str+'\n')
    print ()
    return np.mean(accuracy), val, fnfpids


def evaluate_onnx(model, data_list, actual_issame, nrof_folds, log_dir=None, log_file=None, step=0,
                   img_size=(112,112,3),  batch_size=100, name='', flip=True,
                   data_transform_id='CV-kneron', **kwargs):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print ('\nRunning forward pass on {} images: '.format(name), end='')
    nrof_images = len(actual_issame)*2

    if flip:
        print (' with adding flipped image')
        embeddings_list = []
        for i in range(len(data_list)):
            idata = data_list[i]
            num_batches = int(len(idata)/batch_size)
            data = np.zeros([batch_size, 3, img_size[0], img_size[1]], dtype=np.float)
            for b in range(num_batches):
                for img in range(batch_size):
                    data[img] = data_transforms[data_transform_id](idata[img+b*batch_size])
                #output = model(data)
                output = model.run(data)
                #emb_in_batch = output.cpu().numpy()
                emb_in_batch = np.array(output)
                print(emb_in_batch.shape)

                if b == 0:
                    emb_per_list = emb_in_batch.copy()
                else:
                    emb_per_list = np.concatenate((emb_per_list, emb_in_batch))
            print(emb_per_list.shape)
            #emb_per_list = emb_per_list / (np.linalg.norm(emb_per_list, axis=-1)+1e-10).reshape(-1,1)
            embeddings_list.append(emb_per_list)

        # add addtional individual normalized (may lower accuracy but increase verification)
        # embeddings = (embeddings_list[0] / (np.linalg.norm(embeddings_list[0], axis=-1)).reshape(-1,1)) + \
        #        (embeddings_list[1] / (np.linalg.norm(embeddings_list[1], axis=-1)).reshape(-1,1))
        embeddings = embeddings_list[0] + embeddings_list[1]
    else:
        print ()
        idata = data_list[0]
        #print(idata)
        num_batches = int(len(idata)/batch_size)
        embeddings = np.zeros( [ len(idata), 256]  )
        for b in tqdm(range(num_batches)):
            #data = np.zeros([batch_size, 3, img_size[0], img_size[1]], dtype=np.float)

            #print(data_transforms[data_transform_id])
            data = idata[b]
            #print(data.shape)
            data = np.array(data, dtype=np.float32)
            data = cv2.cvtColor( data, cv2.COLOR_BGR2RGB )
            output = model.run(data)
            #emb_in_batch = output.cpu().numpy()
            emb_in_batch = np.array(output)
            #print(emb_in_batch.shape)
            #sys.exit(1)

            emb_in_batch = emb_in_batch / (np.linalg.norm(emb_in_batch, axis=-1)+1e-10).reshape(-1)
            embeddings[b] = emb_in_batch
            #if b == 0:
            #    embeddings = emb_in_batch.copy()
            #else:
            #    embeddings = np.concatenate((embeddings, emb_in_batch))
            #print(embeddings.shape)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)+1e-10).reshape(-1,1)

    embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1)).reshape(-1,1)

    print('Duration:', '%.3f' % (time.time()-start_time), 'seconds, embedding shape:', embeddings.shape)
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, actual_issame, nrof_folds=nrof_folds)
    print('Accuracy: %1.5f+-%1.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.5f' % auc)
    # remove EER due to bug: A value in x_new is above the interpolation range. #Long#
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.5f' % eer)

    log_str = '%s \tACC: %.5f\tVAL: %.5f\t@FAR: %.5f\tEpoch: %d\t%s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), np.mean(accuracy), val, far, step, name)
    print(log_str)
    if log_dir and log_file:
        with open(os.path.join(log_dir, log_file),'at') as f:
            #f.write('%s \tACC: %.5f\tVAL: %.5f\t@FAR: %.5f\tEpoch: %d\t%s \n' % (str(datetime.now()), np.mean(accuracy), val, far, step, name))
            f.write(log_str+'\n')

    return np.mean(accuracy), val
