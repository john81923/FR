import os
import torch

# Sets device for model and PyTorch tensors
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
NUM_WORKERS = 20  # for data-loading; right now, only 1 works with h5py
GRAD_CLIP = 5.  # clip gradients at an absolute value of
PRINT_FREQ = 100  # print training/validation stats  every __ batches

# Training dataset
# IMAGE_ROOT = None
# IMAGE_ROOT = '/opt/train_data/fr_data/vgg2/vgg2_112x112'
# IMAGE_ROOT = '/opt/train_data/fr_data/faces_glint/all_112'
# IMAGE_ROOT = '/opt/train_data/fr_data/faces_glint/train_kface/msra'
# IMAGE_ROOT = '/opt/train_data/fr_data/insightface/ms1m-retinaface-t1-112x112'
# IMAGE_LIST = None
# IMAGE_LIST = '/opt/train_data/fr_data/vgg2/vgg2.txt'
#IMAGE_ROOT = '/opt/train_data/fr_data/insightface/ms1m-retinaface-t1-112x112'
'''
IMAGE_ROOT = [
        #'/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        #'/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112',
        #'/mnt/sdc/craig/CASIA_NIR/NIR_kfr_0304',
        #'/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle/', #insight strictfix (bad_angle)  (--glass), Yuanlin
        '/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix', #insight strictfix  (--glass), Yuanlin
        #'/mnt/sdd/craig/insightface/ms1m-retinaface-t1-112x112_glass_mixed_rgb_badpose_fix', #insight strictfix, Lucas
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        #'/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/val',
        '/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_train',
        '/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_val',
        '/mnt/sdc/craig/Asia_face_dataset/FR_TPE_Street_Video_Labeled_Dataset_badpose_cleaned',
        ]
'''
#'''



#'/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix', #insight strictfix  (--glass), Yuanlin
#'/mnt/sdd/craig/face_recognition/insightface/FR_insightface/ms1m-retinaface-t1-112x112', #insight strictfix  (--glass), Yuanlin
#'/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
#'/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_train',
##'/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_val',
#'/mnt/sdc/craig/Asia_face_dataset/FR_TPE_Street_Video_Labeled_Dataset_badpose_cleaned',


#mix nir # used in v12-16_3 (mi-8) 2020/12/10 + try RGB/NIR separated

#RGB
'''
IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign_bad_angle',
        '/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data_realign_bad_pose',
        '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_C_rgb',
        ]
'''
#NIR
'''
IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign_bad_angle',
        #'/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data_realign_bad_pose',
        '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_C_nir',
        ]
'''

# realigned, RGB/NIR separated
'''
IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle/', #insight strictfix (bad_angle)  (--glass), Yuanlin
        '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        #'/mnt/sdd/craig/face_recognition/AsianTestDatasets_badpose_train_realign_bad_angle',
        #'/mnt/sdd/craig/face_recognition/AsianTestDatasets_badpose_val_realign_bad_angle',
        #'/mnt/sdd/craig/face_recognition/FR_TPE_badpose_cleaned_realign_bad_angle',
        #'/mnt/sdd/craig/face_recognition/Guizhou_NIR_dataset_kfr_0304_trainval_realign_bad_angle/train',
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        #'/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_train',
        #'/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_val',
        #'/mnt/sdc/craig/Asia_face_dataset/FR_TPE_Street_Video_Labeled_Dataset_badpose_cleaned',
        # RGB/NIR separated full
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_nir',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_nir',
        ]
'''

# Glink 360
'''
IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign',
        #'/mnt/sdd/craig/face_recognition/glint/FR_glint_bad_angle/', #insight strictfix  (--glass), Yuanlin
        '/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign_bad_angle', #insight strictfix  (--glass), Yuanlin
        #'/mnt/sdd/craig/face_recognition/glint/FR_glint_bad_angle/', #insight strictfix  (--glass), Yuanlin
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_C_nir150',
        ]
'''

#mix nir # used in v12-16_3 (mi-8) 2020/11/13 + vloggerface + try rgb only
'''
IMAGE_ROOT = [
        '/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data_realign_bad_angle', #vlogger
        '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_rgb',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_rgb',
        ]
'''

#mix nir # used in v12-16_3 (mi-8) 2020/11/13 + vloggerface + celeb > 10 samples, 2020/01/01

'''
IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign_bad_angle_bad_pose', #celeb
        '/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data_realign_bad_pose', #vlog
        '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_C_nir150',
        ]
'''

# Johnny Note : Training command
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /data2/johnnyalg1/model/FR/0609_resnet_mi_v30_3_ckpt302 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /data2/johnnyalg1/model/FR/0612_resnet_mi_v30_2/checkpoint-r50-I112-E256-e0059-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample'
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /data2/johnnyalg1/model/FR/0609_resnet_mi_v30_3_ckpt303 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /data2/johnnyalg1/model/FR/0612_resnet_mi_v30_3/checkpoint-r50-I112-E256-e0059-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample'
#
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /data1/johnnyalg1/model/FR/0617_resnet_mi_v30_3_ckpt303 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /data1/johnnyalg1/model/FR/0612_resnet_mi_v30_3/checkpoint-r50-I112-E256-e0059-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample'
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /data1/johnnyalg1/model/FR/0617_resnet_mi_v30_3_ckpt302 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /data1/johnnyalg1/model/FR/0612_resnet_mi_v30_2/checkpoint-r50-I112-E256-e0059-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample'
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /data1/johnnyalg1/model/FR/0617_resnet_mi_v30_3_ckpt302 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /data1/johnnyalg1/model/FR/0617_resnet_mi_v30_3_ckpt303/checkpoint-r50-I112-E256-e0069-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment '

# Johnny reproduce
'''
IMAGE_ROOT = [
        '/data1/johnnyalg1/data/FR/vloggerface/align_data_realign_bad_pose', #vlog
        '/data1/johnnyalg1/data/FR/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        '/data1/johnnyalg1/data/FR/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/data1/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200608_nir150_new_copy',
        '/data1/johnnyalg1/data/FR/mi8/mi8_mdong_112x112_20200608_nir150_new_copy',
        '/data1/johnnyalg1/data/FR/mi8/20200611_nir150_new_copy',
        '/data1/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_A_nir150',
        '/data1/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_B_nir150',
        '/data1/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_C_nir150',
        '/data1/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_D_nir150',
        '/data1/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_E_nir150',
        '/data1/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_F_nir150',
        '/data1/johnnyalg1/data/FR/mi8/mi8_mdong_112x112_20200803_A_nir150',
        '/data1/johnnyalg1/data/FR/mi8/mi8_mdong_112x112_20200803_B_nir150',
        '/data1/johnnyalg1/data/FR/mi8/mi8_mdong_112x112_20200803_C_nir150',
        ]

'''
# 0612 Johnny realigned
#'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /datadisk/johnnyalg2/model/FR/model/FR/0617_resnet_mi_v31_1_ckpt302 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /datadisk/johnnyalg2/model/FR/0612_resnet_mi_v30_2/checkpoint-r50-I112-E256-e0059-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample --occlusion_ratio 0.2'
#new fc metrics
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /datadisk/johnnyalg2/model/FR/0617_resnet_mi_v31_1_ckpt302 --batch-size 512 --emb-size 256 --no-flip --finetune --freeze_backbone --new_metric_fc --checkpoint /datadisk/johnnyalg2/model/FR/0612_resnet_mi_v30_2/checkpoint-r50-I112-E256-e0059-av0.9998_1.0000.tar --glass --end-epoch 11 --do-rate 0.2 --mi_augment --upsample --occlusion_ratio 0.2'
'CUDA_VISIBLE_DEVICES=1 python -m kfr.bin.ktrain --log-dir /datadisk/johnnyalg2/model/FR/0617_resnet_mi_v31_1_ckpt303 --batch-size 512 --emb-size 256 --no-flip --finetune --freeze_backbone --new_metric_fc --checkpoint /datadisk/johnnyalg2/model/FR/0612_resnet_mi_v30_3/checkpoint-r50-I112-E256-e0059-av0.9998_1.0000.tar --glass --end-epoch 11 --do-rate 0.2 --mi_augment --upsample --occlusion_ratio 0.2'
#finetune
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /datadisk/johnnyalg2/model/FR/0617_resnet_mi_v31_2_ckpt303 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /datadisk/johnnyalg2/model/FR/0617_resnet_mi_v31_1_ckpt303/checkpoint-r50-I112-E256-e0010-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample --occlusion_ratio 0.2'
'CUDA_VISIBLE_DEVICES=1 python -m kfr.bin.ktrain --log-dir /datadisk/johnnyalg2/model/FR/0617_resnet_mi_v31_2_ckpt302 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /datadisk/johnnyalg2/model/FR/0617_resnet_mi_v31_1_ckpt302/checkpoint-r50-I112-E256-e0010-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample --occlusion_ratio 0.2'
# occ 0.15
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /datadisk/johnnyalg2/model/FR/0617_resnet_mi_v31_2_ckpt303_occ015/ --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /datadisk/johnnyalg2/model/FR/0617_resnet_mi_v31_1_ckpt303_occ015/checkpoint-r50-I112-E256-e0010-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample --occlusion_ratio 0.15'
# 32_2
'CUDA_VISIBLE_DEVICES=0 python -m kfr.bin.ktrain --log-dir /datadisk/johnnyalg2/model/FR/0706_resnet_mi_v32_2_ckpt303 --batch-size 512 --emb-size 256 --no-flip --finetune --checkpoint /datadisk/johnnyalg2/model/FR/0706_resnet_mi_v32_1_ckpt303/checkpoint-r50-I112-E256-e0012-av0.9998_1.0000.tar --glass --end-epoch 70 --do-rate 0.1 --mi_augment --upsample --attribute_filter'
'''
IMAGE_ROOT = [
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/vloggerface/align_data_realign_bad_pose_060921realign', #vlog
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign_060921realign', #insight strictfix  (--glass), Yuanlin
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/Guizhou_NIR_dataset_kfr_0304_trainval/train_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200608_nir150_new_copy_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_mdong_112x112_20200608_nir150_new_copy_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/20200611_nir150_new_copy_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_A_nir150_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_B_nir150_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_C_nir150_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_D_nir150_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_E_nir150_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_F_nir150_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_mdong_112x112_20200803_A_nir150_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_mdong_112x112_20200803_B_nir150_060921realign',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_mdong_112x112_20200803_C_nir150_060921realign',
        ]
'''

# computer01 + webface henan
IMAGE_ROOT = [
        '/mnt/sdc/johnnysun/data/FR/vloggerface/align_data_realign_bad_pose_060921realign', #vlog
        '/mnt/sdc/johnnysun/data/FR/WebFace260M-Full/WebFace260M_realign',
        '/mnt/sdd/johnnysun/data/FR/Guizhou_NIR_dataset_kfr_0304_trainval/train_060921realign',
        #'/mnt/sdd/johnnysun/data/FR/white850_realign/train',
        #'/mnt/sdd/johnnysun/data/FR/black940_realign/train',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_liu_112x112_20200608_nir150_new_copy_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_mdong_112x112_20200608_nir150_new_copy_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/20200611_nir150_new_copy_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_liu_112x112_20200709_A_nir150_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_liu_112x112_20200709_B_nir150_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_liu_112x112_20200709_C_nir150_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_liu_112x112_20200709_D_nir150_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_liu_112x112_20200709_E_nir150_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_liu_112x112_20200709_F_nir150_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_mdong_112x112_20200803_A_nir150_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_mdong_112x112_20200803_B_nir150_060921realign',
        '/mnt/sdc/johnnysun/data/FR/mi8/mi8_mdong_112x112_20200803_C_nir150_060921realign',
        ]

#mix nir # used in v12-16_3 (mi-8) 2020/11/13 + vloggerface

'''
IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign_bad_angle',
        '/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data_realign_bad_pose',
        '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_C_nir150',
        ]
'''
#mix nir # used in v12-16_3 (mi-8) 2020/11/13
'''
IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle/', #insight strictfix (bad_angle)  (--glass), Yuanlin
        '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        #'/mnt/sdd/craig/face_recognition/AsianTestDatasets_badpose_train_realign_bad_angle',
        #'/mnt/sdd/craig/face_recognition/AsianTestDatasets_badpose_val_realign_bad_angle',
        #'/mnt/sdd/craig/face_recognition/FR_TPE_badpose_cleaned_realign_bad_angle',
        #'/mnt/sdd/craig/face_recognition/Guizhou_NIR_dataset_kfr_0304_trainval_realign_bad_angle/train',
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        #'/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_train',
        #'/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_val',
        #'/mnt/sdc/craig/Asia_face_dataset/FR_TPE_Street_Video_Labeled_Dataset_badpose_cleaned',
        # RGB/NIR separated full
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_nir150',
        '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_C_nir150',
        ]
'''

"""

# RGB/NIR merged full
'/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy',
'/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy',
'/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy',

# RGB/NIR merged
'/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle',
'/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle',
'/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_bad_angle',
# RGB/NIR separated
'/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle_rgb',
'/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle_nir',
'/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle_rgb',
'/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle_nir',
'/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_bad_angle_rgb',
'/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_bad_angle_nir',
"""
#'''
'''
# no-insight
IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle/', #insight strictfix (bad_angle)  (--glass), Yuanlin
        '/mnt/sdd/craig/face_recognition/Guizhou_NIR_dataset_kfr_0304_trainval_realign_bad_angle/train',
        '/mnt/sdd/craig/face_recognition/AsianTestDatasets_badpose_train_realign_bad_angle',
        '/mnt/sdd/craig/face_recognition/AsianTestDatasets_badpose_val_realign_bad_angle',
        '/mnt/sdd/craig/face_recognition/FR_TPE_badpose_cleaned_realign_bad_angle',
        '/mnt/sdd/craig/face_recognition/mi8/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle',
        '/mnt/sdd/craig/face_recognition/mi8/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle',
        #'/mnt/sdd/craig/face_recognition/mi8/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle_rgb',
        #'/mnt/sdd/craig/face_recognition/mi8/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle_rgb',
        #'/mnt/sdd/craig/face_recognition/mi8/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle_nir',
        #'/mnt/sdd/craig/face_recognition/mi8/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle_nir',
        #'/mnt/sdd/craig/face_recognition/mi8/mi8_liu_112x112_20200608_nir150_new_copy',
        #'/mnt/sdd/craig/face_recognition/mi8/mi8_mdong_112x112_20200608_nir150_new_copy',
        ]

'''
'''
IMAGE_ROOT = [
        '/mnt/sdc/craig/Asia_face_dataset/FR_TPE_Street_Video_Labeled_Dataset_badpose_cleaned',
        ]
'''

ATTRIBUTE_THRES = {
    'badpose_hor_ver': [ (0.33333, 3), (0.25, 6)],
    'badangle_hor_ver': [1.5, 1.5],
    'pose_quality_yrp': [25, 10, 25],
    'pose_quality_facequality': 0.4,
    'occ_cls_eye_nose_chin': [0.0, 0.0, 0.0, 0.0],
    'ocr_filter_eye_nose_chin': [0.5, 0.5, 0.0],
    'occ_ratio': 0.2,
    'glass': 0,
    'sunglass': 0,
    'has_mask': 0
    }

'''
old_ATTRIBUTE_THRES = {
    'badpose_hor_ver': [ (0.33333, 3), (0.25, 6)],
    'badangle_hor_ver': [1.5, 1.5],
    'pose_quality_yrp': [35, 25, 35],
    'pose_quality_facequality': 0.5,
    'occ_cls_eye_nose_chin': [0.0, 0.0, 0.0, 0.0],
    'ocr_filter_eye_nose_chin': [0.5, 0.5, 0.0],
    'occ_ratio': 0.2,
    'glass': 0,
    'sunglass': 0,
    'has_mask': 0
    }
'''

MODEL_STRUCTURE = '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/'
IMAGE_LIST = None
#IMAGE_PER_LABEL = 3
IMAGE_PER_LABEL = 10

# Validation dataset
PAIR_FOLDER_NAME = "/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/pair_gen"
VALID_LIST = [
        #'/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_val',
        #'/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_val',
        #'/mnt/sdc/craig/Asia_face_dataset/FR_TPE_Street_Video_Labeled_Dataset_badpose_cleaned',
        ['/mnt/sdd/johnnysun/data/FR/henan_gz_val/henan_gz_val_data_20.pkl',
          '/mnt/sdd/johnnysun/data/FR/henan_gz_val/henan_gz_val_issame_20.pkl'],

        ]

# Logging/checkpoint parameters
LOG_DIR = './checkpoint'
TRAIN_LOG_FILE = 'train.log'
EVAL_LOG_FILE = 'train.log'
