from ..model.resnet import ResNet50Backbone
from ..eval.eval import load_val_dataset, eval_model
import argparse
import os
import torch
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--network', default='r50', help='specify network')
    parser.add_argument('--inp-size', type=int, default=112, help='input image size')
    parser.add_argument('--block-layout', type=str, default='8 28 6', help='feature block layout')
    #parser.add_argument('--block-size', type=str, default='32 384 1152 2144', help='feature block size')
    parser.add_argument('--block-size', type=str, default='32 32 64 128 256', help='feature block size')
    parser.add_argument('--se-ratio', type=int, default=0, help='SE reduction ratio')
    parser.add_argument('--head', type=str, default='fc', help='head fc or varg')
    #parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--emb-size', type=int, default=256, help='embedding length')
    #parser.add_argument('--do-rate', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--do-rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
    parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
    #parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
    parser.add_argument('--focal-loss', type=bool, default=True, help='focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    #parser.add_argument('--checkpoint', type=str, default='None', help='checkpoint')
    parser.add_argument('--checkpoint', type=str, default='/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/checkpoint/kface_fr_gray-r50_nb-E256-av0.9980_0.9730_0.9786_0.9967_0.9260_0.9154.pth', help='checkpoint')
    parser.add_argument('--nir-head', type=str, default='/mnt/storage1/craig/kneron_fr/0314_10_label_v23/checkpoint-r50-I112-E512-e0099-av0.9987_0.9990.tar', help='nir_head')
    #parser.add_argument('--nir-head', type=str, default='/mnt/storage1/craig/kneron_fr/0305_10_label_emb_128/checkpoint-r50-I112-E128-e0099-av0.9995_0.9996.tar', help='nir_head')

    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')
    parser.add_argument('--warmup', type=int, default=0, help='warmup training epochs without validation')
    parser.add_argument('--cooldown', type=int, default=0, help='keep training with repeating the last few epochs')
    parser.add_argument('--max-cool', type=int, default=5, help='Maxium cooling down without improvment')
    parser.add_argument('--end-epoch', type=int, default=100, help='training epoch size.')
    parser.add_argument('--gpus', type=str, default='0', help='running on GPUs ID')
    #parser.add_argument('--log-dir', type=str, default=None, help='Checkpoint/log root directory')
    parser.add_argument('--log-dir', type=str, default='/mnt/storage1/craig/test_kfr', help='Checkpoint/log root directory')
    #parser.add_argument('-gr', '--grayscale', help='Use grayscale input.', action='store_true')
    parser.add_argument('-nf', '--no-flip', help='No face flip in evaluation.', action='store_false')
    parser.add_argument('-pre', '--pre-norm', help='Preprocessing normalization id.', default='CV-kneron')

    args = parser.parse_args()
    args.block_layout = [int(n) for n in args.block_layout.split()]
    args.block_size = [int(n) for n in args.block_size.split()]
    args.gpus = [int(n) for n in args.gpus.split()]

    return args


def load_folder_inf(img_folder_name, same_pairs_path, diff_pairs_path,image_size=(112,112)):
    image_list = []
    with open(same_pairs_path,'r') as fp:
        for line in fp.readlines():
            line = line.split('\t')
            image_list.append([line[0], line[1], 1])

    with open(diff_pairs_path,'r') as fp:
        for line in fp.readlines():
            line = line.split('\t')
            image_list.append([line[0], line[1], 0])

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

        img1_path = img1_path.rstrip()
        img2_path = img2_path.rstrip()
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

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
    return (data_list, issame_list)

def load_val_dataset_inf(val_folder, same_pairs_path, diff_pairs_path, input_shape=(112,112,3), nrof_folds=10, to_grayscale=False):
    # load testset
    names = [item for item in val_folder]
    val_dataset = []
    val_labels = []

    for item in val_folder:
        #val_data, actual_issame = load_bin(item, image_size=input_shape[:-1], to_grayscale=to_grayscale)
        val_data, actual_issame = load_folder_inf(item, same_pairs_path, diff_pairs_path, image_size=input_shape[:-1])
        val_dataset.append(val_data)
        val_labels.append(actual_issame)

    return val_dataset, val_labels, names


if __name__ == '__main__':

    same_pairs_path = "/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/pair_gen/validate_same_pairs_casia.txt"
    diff_pairs_path = "/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/pair_gen/validate_diff_pairs_casia.txt"
    VALID_LIST = [
        '/mnt/sdc/craig/CASIA_NIR/NIR_kfr_0304',
        ]

    EVAL_LOG_FILE = 'train.log'
    epoch = 0
    args = parse_args()

    # Case1: Load backbone from originalrgb's backbone
    #'''
    model = torch.load(args.checkpoint)
    backbone = ResNet50Backbone(model)
    backbone.eval()
    #'''
    # Case2: Load backbone and head from our checkpoint
    checkpoint_state = torch.load(args.nir_head)
    #backbone = checkpoint_state['backbone']
    print('load backbone')


    head = checkpoint_state['head']
    print('load head')
    backbone.eval()
    head.eval()

    print('load val_dataset')
    val_datasets, val_labels, names = load_val_dataset_inf(VALID_LIST, same_pairs_path, diff_pairs_path, to_grayscale=True)

    #https://www.tutorialsteacher.com/python/python-read-write-file

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    print('eval model')
    accs, vals = eval_model(backbone, head, val_datasets, val_labels, names,
                    log_dir=args.log_dir, log_file=EVAL_LOG_FILE, step=epoch, flip=args.no_flip, data_transform_id=args.pre_norm)
    #print('accs:',accs,'vals:',vals)
