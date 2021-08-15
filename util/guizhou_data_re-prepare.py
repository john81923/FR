from glob import glob
import cv2
from os.path import basename, splitext
import os
import sys

tar_path = '/mnt/atdata/FacialRecognition/Guizhou_NIR_dataset/Guizhou_NIR_dataset_kfr_0722_realign'
train_path = '/mnt/sdd/johnnysun/data/FR/Guizhou_NIR_dataset_kfr_0304_trainval/train'
val_path = '/mnt/sdd/johnnysun/data/FR/Guizhou_NIR_dataset_kfr_0304_trainval/val'

h1 = '/mnt/sdd/johnnysun/data/FR/white850_realign/val/'
h2 = '/mnt/sdd/johnnysun/data/FR/black940_realign/val'
tar_path = '/mnt/sdd/johnnysun/data/FR/henan_realign/val'

for idname_path in glob(f'{h1}/*'):
    idname = basename( idname_path )
    for img_path in glob(f'{idname_path}/*'):
        bname = basename(img_path)
        file_path = f'{tar_path}/{idname}/white_{bname}'
        if not os.path.exists( f'{tar_path}/{idname}' ):
            os.makedirs( f'{tar_path}/{idname}' )
        os.system( f'cp {img_path} {file_path}')

for idname_path in glob(f'{h2}/*'):
    idname = basename( idname_path )
    for img_path in glob(f'{idname_path}/*'):
        bname = basename(img_path)
        file_path = f'{tar_path}/{idname}/black_{bname}'
        if not os.path.exists( f'{tar_path}/{idname}' ):
            os.makedirs( f'{tar_path}/{idname}' )
        os.system( f'cp {img_path} {file_path}')




sys.exit(1)
ID_path_list = glob( f'{tar_path}/*')
train_ID_path_list = glob( f'{train_path}/*')
train_ID_path_list = [ basename(ID_path) for ID_path in train_ID_path_list ]
#print(train_ID_path_list)


for in_id_path in ID_path_list :
    id_name = basename( in_id_path)
    if not id_name in train_ID_path_list:
        print(id_name)
        os.system( f'cp -r {tar_path}/{id_name} {val_path}/')
