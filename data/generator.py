import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import BatchSampler, WeightedRandomSampler
from torch.utils.data import DataLoader

import numpy as np
import os, sys
import glob
import random

import os
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import sys
import pickle
from os.path import splitext, basename, dirname
import json

from time import time
from tqdm import tqdm
from ..conf.config import ATTRIBUTE_THRES
print("ATTRIBUTE_THRES")
print(ATTRIBUTE_THRES)
ia.seed(1)

seq = iaa.Sequential([

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
        #iaa.GaussianBlur(sigma=(0, 1))
    ),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel
    #. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=False),
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=False),
], random_order=True) # apply augmenters in random order


seq_indoor = iaa.Sequential([

    iaa.OneOf([
                    iaa.Multiply((0.8, 1.2), per_channel=False),
                    iaa.Multiply((0.4, 0.6), per_channel=False),
                ]),
], random_order=True) # apply augmenters in random order

seq_outdoor = iaa.Sequential([

    iaa.OneOf([
                    iaa.Multiply((0.8, 1.2), per_channel=False),
                    iaa.Multiply((1.4, 1.6), per_channel=False),
                ]),
], random_order=True) # apply augmenters in random order

class ImageDataSet(Dataset):
    image_list = []
    label_list = []

    def __init__(self, image_root=None, image_list=None, image_transform_indoor=None, image_transform_outdoor=None, image_transform= None, data_transform=None, min_images_per_label=10, with_glass = False, with_lighting = False, with_mi_augment = False, upsample = False, upsample_child = False, occlusion_ratio=1.0, attribute_filter=False):
        self.image_transform_indoor = image_transform_indoor
        self.image_transform_outdoor = image_transform_outdoor
        self.image_transform = image_transform
        self.data_transform = data_transform
        self.min_images_per_label = min_images_per_label
        self.image_root = image_root

        self.image_list = image_list
        self.with_glass = with_glass
        self.with_lighting = with_lighting
        self.with_mi_augment = with_mi_augment
        self.upsample = upsample
        self.upsample_child = upsample_child
        self.occlusion_ratio = occlusion_ratio
        self.attribute_filter = attribute_filter


        # Open and load text file including the whole training data
        if self.image_list == None:
            # load from image_root
            print('Loading data from directory:\n', self.image_root)
            self.image_list, self.label_list, self.images_per_label, self.upsample_index, self.upsample_index_child = self._get_dataset_from_dir(self.image_root, self.min_images_per_label)
        else:
            if self.image_root == None:
                self.image_root = os.path.dirname(self.image_list)
            print('Loading data from list:', self.image_list)

            self.image_list, self.label_list, self.images_per_label = self._get_dataset_by_csv(
                    self.image_list, self.image_root, min_images_per_label=self.min_images_per_label)

        #print ("self.label_list: ", self.label_list)
        #exit(0)
        self.min_num_images = min(self.images_per_label.values())
        self.max_num_images = max(self.images_per_label.values())
        self.num_labels = len(self.images_per_label)
        print('- number of labels:', self.num_labels)
        print('- total number of images:', self.__len__())
        print('- number of images per label, min/max: {}/{}'.format(self.min_num_images, self.max_num_images))

    # Override to give PyTorch access to any image on the dataset
    '''
    def __getitem__batch(self, index):
        imgs = []
        labels = []
        for i in index:
            img = Image.open(self.image_list[i])
            img = img.convert('RGB')

            if self.image_transform:
                img = self.image_transform(img)

            # Convert image and label to torch tensors
            img = np.asarray(img)

            if self.data_transform:
                img = self.data_transform(img)

            imgs.append(img)
            labels.append(self.label_list[i])

        return torch.stack(imgs), labels
    '''



    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        img = img.convert('RGB')
        #print("origin_mode:", img.mode)
        #print("origin_value:",img.getpixel((30,30)))
        #if self.image_transform:
        if 'indoor' in self.image_list[index]:
            img = self.image_transform_indoor(img)
        elif 'outdoor' in self.image_list[index]:
            img = self.image_transform_outdoor(img)
        else:
            #raise NotImplementedError
            img = self.image_transform(img)

        # Convert image and label to torch tensors
        #img2 = np.asarray(img)
        img = np.asarray(img, dtype='f')

        #images_aug = seq(image=img)
        '''
        img = np.asarray(img, dtype='f')
        if 'indoor' in self.image_list[index]:
            images_aug = seq_indoor(image=img)
        elif 'outdoor' in self.image_list[index]:
            images_aug = seq_outdoor(image=img)
        else:
            raise NotImplementedError
        '''
        #images_aug = seq(image=img)
        #images_aug = img
        #print("img.dtype: ", img.dtype)
        #print("img.shape: ", img.shape) #112,112,3
        #print("img[30][30]", img[30][30])
        #print("img2[30][30]", img2[30][30])
        #exit(0)
        if self.data_transform:
            img = self.data_transform(img)
            #img = self.data_transform(img)

        label = self.label_list[index]
        return img, label


    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.image_list)



    def _get_dataset_from_dir(self, img_root_list, min_images_per_label=10):
        image_list = []
        label_list = []
        images_per_label = {}
        upsample_index = {}
        upsample_index_child = {}
        label_seq = 0
        missglass_counter = 0
        for img_root in img_root_list:
            print ("=============================================")
            print("img_root: ", img_root)
            #counter +=1
            #if counter == 2:
            #    exit(0)
            #if 'ms1m-retinaface-t1-112x112' in img_root:
            if self.with_glass and 'insightface' in img_root:
                print ("with glasses")
                path_exp = os.path.expanduser(img_root)
                labels = sorted(os.listdir(path_exp))
                labels = [item for item in labels if '.txt' not in item]
                labels = np.asarray(labels, dtype=str)
                labels = np.sort(labels)
                print("len(labels):",  len(labels))
                #glass_path = "/datadisk/johnnyalg2/data/FR/FRdata_v30_3/insightface/ms1m-retinaface-t1-112x112_glass_yuanlin_rgb_strict"
                glass_path = "/data1/johnnyalg1/data/FR/insightface/ms1m-retinaface-t1-112x112_glass_yuanlin_rgb_strict"
                if not os.path.exists(glass_path):
                    raise ValueError("glass_path not exist")
                path_exp_glass = os.path.expanduser(glass_path)

                #if 'occ_all' in img_root:
                if self.with_lighting:
                    lighting_path = "/datadisk/johnnyalg2/data/FR/FRdata_v30_3/insightface/FR_insightface_lighting_sample_bad_angle"
                    path_exp_lighting = os.path.expanduser(lighting_path)
                    if not os.path.exists(lighting_path):
                        raise ValueError("lighting_path not exist")

                if self.occlusion_ratio != 1:
                    if 'realign' not in img_root:
                        raise ValueError("060921realign path not exist")
                if self.attribute_filter:
                    print("with attribute_filter")

                for label in labels:
                    facedir = os.path.join(path_exp, str(label))
                    facedir_glass = os.path.join(path_exp_glass, str(label))
                    image_paths = glob.glob(os.path.join(facedir, '*'))
                    #print(image_paths)

                    ''' Occlusion '''
                    if self.occlusion_ratio != 1:
                        occfilt_paths = []
                        for imitem in image_paths:
                            occresult = int( (splitext( basename(imitem) )[0])[-2:] )*0.01
                            if occresult < self.occlusion_ratio:
                                occfilt_paths.append(imitem)
                        image_paths = occfilt_paths
                    ''''''

                    ''' attribute filter '''
                    if self.attribute_filter:
                        #print('is filtering')
                        attribute_json_path = '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign_060921realign_json'
                        att_filtered_path = []
                        for imitem in image_paths:
                            bname = basename(splitext(imitem)[0])
                            filterdir = f'{attribute_json_path}/{bname}.json'
                            if not os.path.exists(filterdir):
                                continue
                            with open( filterdir, 'r') as json_file:
                                attribute_object = json.load(json_file)
                            val = attribute_object['attribute']['badpose_hor_ver']
                            if ATTRIBUTE_THRES['badpose_hor_ver'][0][0] > val[0] or val[0] > ATTRIBUTE_THRES['badpose_hor_ver'][0][1] or \
                                    ATTRIBUTE_THRES['badpose_hor_ver'][1][0] > val[1] or val[0] > ATTRIBUTE_THRES['badpose_hor_ver'][1][1] or \
                                    val[0] == 0 or val[1] == 0 :
                                continue
                            val = attribute_object['attribute']['badangle_hor_ver']
                            if ATTRIBUTE_THRES['badangle_hor_ver'][0] < val[0] or ATTRIBUTE_THRES['badangle_hor_ver'][1] < val[1] \
                                    or val[0] == 0 or val[1] == 0 :
                                continue
                            val = attribute_object['attribute']['pose_quality_yrp']
                            if ATTRIBUTE_THRES['pose_quality_yrp'][0] < val[0] or ATTRIBUTE_THRES['pose_quality_yrp'][1] < val[1] or ATTRIBUTE_THRES['pose_quality_yrp'][2] < val[2] \
                                     -ATTRIBUTE_THRES['pose_quality_yrp'][0] > val[0] or -ATTRIBUTE_THRES['pose_quality_yrp'][1] > val[1] or -ATTRIBUTE_THRES['pose_quality_yrp'][2] > val[2]:
                                continue
                            val = attribute_object['attribute']['pose_quality_facequality']
                            if ATTRIBUTE_THRES['pose_quality_facequality'] > val:
                                continue
                            val = attribute_object['attribute']['occ_cls_eye_nose_chin']
                            if ATTRIBUTE_THRES['occ_cls_eye_nose_chin'][0] > val[0] or ATTRIBUTE_THRES['occ_cls_eye_nose_chin'][1] > val[1] or ATTRIBUTE_THRES['occ_cls_eye_nose_chin'][2] > val[2] or ATTRIBUTE_THRES['occ_cls_eye_nose_chin'][3] > val[3]:
                                continue
                            val = attribute_object['attribute']['ocr_filter_eye_nose_chin']
                            if ATTRIBUTE_THRES['ocr_filter_eye_nose_chin'][0] > val[0] or ATTRIBUTE_THRES['ocr_filter_eye_nose_chin'][1] > val[1] or ATTRIBUTE_THRES['ocr_filter_eye_nose_chin'][2] > val[2]:
                                continue
                            val = attribute_object['attribute']['occ_ratio']
                            if ATTRIBUTE_THRES['occ_ratio'] < val:
                                continue
                            #elif att == 'glass':
                            #continue
                            val = attribute_object['attribute']['sunglass']
                            if ATTRIBUTE_THRES['sunglass'] != val:
                                continue
                            val = attribute_object['attribute']['has_mask']
                            if ATTRIBUTE_THRES['has_mask'] != val:
                                continue
                            att_filtered_path.append(imitem)
                            #print(attribute_object['attribute'])

                        image_paths = att_filtered_path
                    ''' '''

                    #''' get glass image'''
                    image_paths_glass = []
                    #print('pick miss glass')
                    for imgp in image_paths:
                        foldername = basename( dirname(imgp) )
                        bname = basename( splitext(imgp)[0] )
                        glass_file = f'{glass_path}/{foldername}/{bname}.png'
                        if os.path.exists(glass_file):
                            image_paths_glass.append(glass_file)
                        else:
                            #print( 'glass Not exists', glass_file)
                            missglass_counter += 1
                    #image_paths_glass = glob.glob(os.path.join(facedir_glass, '*'))

                    original_images_in_label = len(image_paths)
                    image_paths.extend(image_paths_glass)
                    if self.with_lighting:
                        facedir_lighting = os.path.join(path_exp_lighting, str(label))
                        image_paths_lighting = glob.glob(os.path.join(facedir_lighting, '*'))
                        image_paths.extend(image_paths_lighting)

                    images_in_label = len(image_paths)
                    if original_images_in_label >= min_images_per_label: #keep the class number same with no-glass version
                        images_per_label.update({label_seq:images_in_label})
                        if self.upsample or self.upsample_child:
                            upsample_index.update({label_seq:False})
                            upsample_index_child.update({label_seq:False})
                        for path in image_paths:
                            image_list.append(path)
                            label_list.append(label_seq)
                        label_seq += 1
                print("label_seq:",  label_seq)
                print("missglass_counter", missglass_counter)

                #'''
            elif 'FR_glint' in img_root and 'celebrity_112' not in img_root:
                print ("Glint: add 110k enhancement")
                path_exp = os.path.expanduser(img_root)
                labels = sorted(os.listdir(path_exp))
                labels = [item for item in labels if '.txt' not in item]
                labels = np.asarray(labels, dtype=str)
                labels = np.sort(labels)
                print("len(labels):",  len(labels))
                enhance_path = "/mnt/sdd/craig/face_recognition/glint/110k_enhancement_bad_angle"
                #glass_path = "/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_glass_yuanlin_rgb_strict_bad_angle"
                path_exp_enhance = os.path.expanduser(enhance_path)

                for label in labels:
                    facedir = os.path.join(path_exp, str(label))
                    image_paths = glob.glob(os.path.join(facedir, '*'))
                    facedir_enhance = os.path.join(path_exp_enhance, str(label))
                    if os.path.isdir(facedir_enhance):
                        image_paths_enhance= glob.glob(os.path.join(facedir_enhance, '*'))
                        image_paths.extend(image_paths_enhance)
                    original_images_in_label = len(image_paths)

                    images_in_label = len(image_paths)

                    if original_images_in_label >= min_images_per_label: #keep the class number same with no-glass version
                        images_per_label.update({label_seq:images_in_label})
                        if self.upsample or self.upsample_child:
                            upsample_index.update({label_seq:False})
                            upsample_index_child.update({label_seq:False})
                        for path in image_paths:
                            image_list.append(path)
                            label_list.append(label_seq)
                        label_seq += 1
                print("label_seq:",  label_seq)
                #'''
            elif 'WebFace260M' in img_root:
                if not os.path.exists(img_root):
                    raise ValueError("WebFace260M not exist")
                path_exp = os.path.expanduser(img_root)
                labels = sorted(os.listdir(path_exp))
                labels = np.asarray(labels, dtype=str)
                labels = np.sort(labels)
                print("len(labels):",  len(labels))

                for label in labels:
                    facedir = os.path.join(path_exp, str(label))
                    image_paths = glob.glob(os.path.join(facedir, '*'))

                    original_images_in_label = len(image_paths)
                    images_in_label = len(image_paths)

                    if original_images_in_label >= min_images_per_label: #keep the class number same with no-glass version
                        images_per_label.update({label_seq:images_in_label})
                        if self.upsample or self.upsample_child:
                            upsample_index.update({label_seq:False})
                            upsample_index_child.update({label_seq:False})
                        for path in image_paths:
                            image_list.append(path)
                            label_list.append(label_seq)
                        label_seq += 1
                print("label_seq:",  label_seq)

            else:
                print ("without glasses")
                path_exp = os.path.expanduser(img_root)
                labels = sorted(os.listdir(path_exp))
                labels = [item for item in labels if '.txt' not in item]
                # may sorted as str
                labels = np.asarray(labels, dtype=str)
                #labels = np.asarray(labels, dtype=int)
                labels = np.sort(labels)
                print("len(labels):",  len(labels))

                #untrained_celeb_list = []
                #untrained_celeb_list = pickle.load( open( "untrained_celeb_list.pkl", "rb" ) )
                #print(len(untrained_celeb_list))
                #print(untrained_celeb_list[0])
                #exit(0)
                #pickle.dump(untrained_celeb_list, open( "untrained_celeb_list.pkl", "wb" ) )

                if 'mi8'in img_root and self.with_mi_augment:
                    mi_augment_path = "/mnt/sdc/johnnysun/data/FR/FR_Asian_Raw_Data_clean_light_augmented"
                    mi_8_folder = os.path.basename(img_root)

                    mi_augment_path= os.path.join(mi_augment_path, mi_8_folder)
                    if not os.path.exists(mi_augment_path):
                        raise ValueError("mi_augment_path not exist")
                    #print ("original mi folder: ", mi_8_folder)
                    print ("mi-8 added path: ", mi_augment_path)
                    path_exp_mi_augment= os.path.expanduser(mi_augment_path)


                for label in labels:
                    facedir = os.path.join(path_exp, str(label))
                    image_paths = glob.glob(os.path.join(facedir, '*'))
                    images_in_label = len(image_paths)


                    if 'mi8'in img_root and self.with_mi_augment:
                        facedir_mi_augment= os.path.join(path_exp_mi_augment, str(label))
                        image_paths_mi_augment = glob.glob(os.path.join(facedir_mi_augment, '*'))
                        image_paths.extend(image_paths_mi_augment)

                    if images_in_label >= min_images_per_label:
                        images_per_label.update({label_seq:images_in_label})
                        if self.upsample or self.upsample_child:
                            if 'mi8'in img_root:
                                upsample_index.update({label_seq:True})
                                if self.upsample_child:
                                    image_name = os.path.basename(image_paths[0])
                                    split_image_name = image_name.split('_') #age
                                    if int(split_image_name[3]) <16:
                                        upsample_index_child.update({label_seq:True})
                                    else:
                                        upsample_index_child.update({label_seq:False})
                            else:
                                upsample_index.update({label_seq:False})
                                upsample_index_child.update({label_seq:False})
                        for path in image_paths:
                            image_list.append(path)
                            label_list.append(label_seq)
                        label_seq += 1

                    #else:
                    #    untrained_celeb_list.append(label)
                        #exit(0)
                '''
                elif (images_in_label) >= 5 and 'insightface' in img_root:
                    images_per_label.update({label_seq:images_in_label})
                    for path in image_paths: #enqueue twice
                        image_list.append(path)
                        image_list.append(path)
                        label_list.append(label_seq)
                        label_list.append(label_seq)
                    label_seq += 1
                elif (images_in_label) >= 3 and 'vloggerface' in img_root:
                    images_per_label.update({label_seq:images_in_label})
                    for path in image_paths: #enqueue twice
                        image_list.append(path)
                        image_list.append(path)
                        label_list.append(label_seq)
                        label_list.append(label_seq)
                    label_seq += 1
                '''
                print("label_seq:",  label_seq)
                #pickle.dump(untrained_celeb_list, open( "untrained_celeb_list.pkl", "wb" ) )
                #exit(0)
        return image_list, label_list, images_per_label, upsample_index, upsample_index_child
    """
    def _get_dataset_from_dir(self, img_root_list, min_images_per_label=10):

        image_list = []
        label_list = []
        images_per_label = {}
        label_seq = 0
        for img_root in img_root_list:
            print("img_root: ", img_root)
            path_exp = os.path.expanduser(img_root)
            labels = sorted(os.listdir(path_exp))
            labels = [item for item in labels if '.txt' not in item]
            # may sorted as str
            labels = np.asarray(labels, dtype=str)
            #labels = np.asarray(labels, dtype=int)
            labels = np.sort(labels)
            print("len(labels):",  len(labels))
            for label in labels:
                facedir = os.path.join(path_exp, str(label))
                image_paths = glob.glob(os.path.join(facedir, '*'))
                images_in_label = len(image_paths)
                if images_in_label >= min_images_per_label:
                    images_per_label.update({label_seq:images_in_label})
                    for path in image_paths:
                        image_list.append(path)
                        label_list.append(label_seq)
                    label_seq += 1
            print("label_seq:",  label_seq)
        return image_list, label_list, images_per_label
        # all the images, their responding lable, 0-1025, images per_label
    """
    def _get_dataset_from_dir_noseq(self, path, min_images_per_label=10):
        path_exp = os.path.expanduser(path)
        labels = sorted(os.listdir(path_exp))
        labels = [item for item in labels if '.txt' not in item]
        labels = np.asarray(labels, dtype=str)
        labels = np.sort(labels)

        image_list = []
        label_list = []
        images_per_label = {}
        for label in labels:
            facedir = os.path.join(path_exp, str(label))
            image_paths = glob.glob(os.path.join(facedir, '*'))
            images_in_label = len(image_paths)
            if images_in_label >= min_images_per_label:
                images_per_label.update({label:images_in_label})
                for path in image_paths:
                    image_list.append(path)
                    label_list.append(label)

        return image_list, label_list, images_per_label


    def _get_dataset_by_csv(self, img_list_csv, root_dir,  min_images_per_label=10, delim=' '):
        """ Load dataset from a CSV file with (sorted by id): image_path id  ...
        """

        image_ids = np.genfromtxt(img_list_csv, dtype=str, delimiter=delim, usecols=(0,1))

        image_list_orig = image_ids[:,0]
        # for str or int labels
        label_list_orig = image_ids[:,1]
        #label_list_orig = image_ids[:,1].astype(int)

        # get label count from label list
        labels_orig, labels_count = np.unique(label_list_orig, return_counts=True)

        label_seq = 0
        label_map = {}
        images_per_label = {}
        for i in range(len(labels_orig)):
            images_in_label = labels_count[i]
            if images_in_label >= min_images_per_label:
                label_map.update({labels_orig[i]:label_seq})
                images_per_label.update({label_seq:images_in_label})
                label_seq += 1
            else:
                label_map.update({labels_orig[i]:-1})

        image_list = []
        label_list = []
        for i in range(len(label_list_orig)):
            label_new = label_map[label_list_orig[i]]
            if label_new >= 0:
                label_list.append(label_new)
                image_list.append(os.path.join(root_dir,image_list_orig[i]))

        return image_list, label_list, images_per_label


def agefilter():
    path = '/jason/insightface/'
    IDs = os.listdir(path)
    for ID in IDs:
        ID_path = os.path.join(path,ID)
        imgs = os.listdir(ID_path)
        if not(os.path.isfile(ID_path+'/head.json')):
            continue
        with open(ID_path+'/head.json') as f:
            ID_info = json.load(f)
            Age_Mean =  int(ID_info['Age_Mean'])
            Age_Std = int(ID_info['Age_Std'])
        valid_img = []
        for img in imgs:
            if 'head' in img:
                continue
            img_path = os.path.join(ID_path,img)
            with open(img_path) as file:
                img_info = json.load(file)
            if Age_Std<=6:
                valid_img.append(img_path.split('.')[0])
                #valid_img.append(img_info['Age'])
                print('Age_std<6,all use')
            elif Age_Mean<=30:
                if Age_Mean-7<=int(img_info['Age'])<=Age_Mean+7:
                    valid_img.append(img_path.split('.')[0])
                    #valid_img.append(img_info['Age'])
                    print(f'{Age_Mean}-7<=Age<={Age_Mean}+7')
            elif Age_Mean>30:
                if Age_Mean-10<=int(img_info['Age'])<=Age_Mean+10:
                    valid_img.append(img_path.split('.')[0])
                    #valid_img.append(img_info['Age'])
                    print(f'{Age_Mean}-10<=Age<={Age_Mean}+10')
        print(sorted(valid_img))  #all_vali_img_list


class ImageDataSampler(BatchSampler):
    def __init__(self, dataset=None, batch_size=1, drop_last=True,
            images_per_label=1):
        self.images_per_label = images_per_label
        self.batch_size = batch_size
        self.dataset = dataset

        self.weighted_random_sampler = self._get_balanced_sampler()
        super().__init__(self.weighted_random_sampler, batch_size=batch_size, drop_last=drop_last)


    def __len__(self):
        return self.num_samples


    def __iter__(self):
        __get_sample_weight()

    def _get_sample_weight(self):
        self.per_label_sample_size = self.images_per_label
        #self.per_label_sample_size = min(self.images_per_label, self.dataset.min_num_images)
        self.num_samples = self.per_label_sample_size * self.dataset.num_labels
        self.weights = np.arange(self.dataset.__len__(), dtype=float)
        for i in range(self.dataset.__len__()):
            label = self.dataset.label_list[i]
            images_per_label = self.dataset.images_per_label[label]
            self.weights[i] = 1./images_per_label
        return WeightedRandomSampler(self.weights, num_samples=self.num_samples, replacement=True)


class BalancedImageDataSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, images_per_label=1, upsample = False, upsample_child = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.images_per_label = images_per_label
        self.sample_size_per_label = min(self.images_per_label, self.dataset.min_num_images)
        self.num_samples = self.sample_size_per_label * self.dataset.num_labels
        self.upsample = upsample
        self.upsample_child = upsample_child
        #self._resample()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self._resample()
        return iter(self.indices)
        """
        if self.sampled+self.batch_size > self.num_samples:
            self._resample()
        print('__iter__:', self.sampled, self.indices[self.sampled:self.sampled+10])
        it = iter(self.indices[self.sampled:])
        self.sampled += self.batch_size
        return it
        """

    def _resample(self):
        """
        Re-populate self.indices with balanced selection of dataset indices
        """

        self.indices = np.arange(0)
        image_index = 0

        labels = self.dataset.images_per_label.keys()
        number_of_total_indice = 0
        for l in labels:
            image_per_label = self.dataset.images_per_label[l]
            images_in_label = np.arange(image_index, image_index+image_per_label)

            if (self.upsample) or self.upsample_child: #mi8 up-sample
                if self.dataset.upsample_index[l] == True:
                    if self.upsample_child:
                        if self.dataset.upsample_index_child[l] == True:
                            child_sample = min(image_per_label,150)
                            number_of_total_indice += child_sample
                        else:
                            number_of_total_indice += 50
                    else:
                        number_of_total_indice += 50
                else:
                    #sample_indices = images_in_label[:self.sample_size_per_label]
                    number_of_total_indice += self.sample_size_per_label
            else:
                #sample_indices = images_in_label[:self.sample_size_per_label]
                number_of_total_indice += self.sample_size_per_label
            image_index += image_per_label

        ''''''
        self.indices = np.arange(number_of_total_indice)
        image_index = 0
        indice_curr_ptr = 0
        for l in labels:
            #t1 = time()
            image_per_label = self.dataset.images_per_label[l]
            images_in_label = np.arange(image_index, image_index+image_per_label)
            np.random.shuffle(images_in_label)

            if (self.upsample) or self.upsample_child: #mi8 up-sample
                if self.dataset.upsample_index[l] == True:
                    if self.upsample_child:
                        if self.dataset.upsample_index_child[l] == True:
                            child_sample = min(image_per_label,150)
                            self.indices[indice_curr_ptr:indice_curr_ptr+child_sample] = images_in_label[:child_sample]
                            indice_curr_ptr += child_sample
                        else:
                            self.indices[indice_curr_ptr:indice_curr_ptr+50] = images_in_label[:50]
                            indice_curr_ptr += 50
                    else:
                        self.indices[indice_curr_ptr:indice_curr_ptr+50] = images_in_label[:50]
                        indice_curr_ptr += 50
                else:
                    self.indices[indice_curr_ptr:indice_curr_ptr+self.sample_size_per_label] = images_in_label[:self.sample_size_per_label]
                    indice_curr_ptr += self.sample_size_per_label
            else:
                self.indices[indice_curr_ptr:indice_curr_ptr+self.sample_size_per_label] = images_in_label[:self.sample_size_per_label]
                indice_curr_ptr += self.sample_size_per_label

            #self.indices = np.concatenate((self.indices, sample_indices))
            #print( 't1',time()-t1) # 0.0001645
            image_index += image_per_label
        #at = time()
        np.random.shuffle(self.indices)
        #print( 'at', time()-at)



class ImageDataGenerator(DataLoader):
    def __init__(self, dataset=None, image_root=None, image_list=None,
            image_transform_indoor=None, image_transform_outdoor=None, image_transform=None, data_transform=None,
            shuffle=True, images_per_label=10, batch_size=64, num_workers=10, glass = False, lighting = False, mi_augment = False, upsample = False, upsample_child = False, occlusion_ratio=1.0, attribute_filter=False, **kwargs):

        if dataset:
            self.dataset = dataset
        else:
            self.dataset =ImageDataSet(image_root=image_root, image_list=image_list,
                    image_transform_indoor=image_transform_indoor, image_transform_outdoor=image_transform_outdoor, image_transform=image_transform, data_transform=data_transform,
                    min_images_per_label=images_per_label, with_glass = glass, with_lighting = lighting, with_mi_augment = mi_augment, upsample = upsample, upsample_child = upsample_child, occlusion_ratio=occlusion_ratio, attribute_filter=attribute_filter)

        self.sampler = BalancedImageDataSampler(self.dataset, batch_size=batch_size,
                images_per_label=images_per_label, upsample = upsample, upsample_child = upsample_child)

        super().__init__(self.dataset, sampler=self.sampler,
                batch_size=batch_size, drop_last=True, pin_memory=True, num_workers=num_workers)


class ImageData(Dataset):
    image_list = []
    label_list = []

    def __init__(self, folder_dataset, image_transform=None, data_transform=None):
        self.image_transform = image_transform
        self.data_transform = data_transform
        # Open and load text file including the whole training data
        with open(folder_dataset+'vgg2.txt') as f:
            for line in f:
                # Image path
                self.image_list.append(folder_dataset + line.split()[0])
                # Steering wheel label
                self.label_list.append(line.split()[1])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        img = img.convert('RGB')
        if self.image_transform:
            img = self.image_transform(img)

        # Convert image and label to torch tensors
        img = np.asarray(img)

        if self.data_transform:
            img = self.data_transform(img)

        label = self.label_list[index]
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.image_list)
