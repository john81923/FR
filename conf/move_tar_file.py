import os
import sys
from os.path import basename, splitext, dirname

IMAGE_ROOT = [
        #'/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign_bad_angle_bad_pose', #celeb
        '/data2/johnnyalg1/data/FR/vloggerface/align_data_realign_bad_pose', #vlog
        '/data2/johnnyalg1/data/FR/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        '/data2/johnnyalg1/data/FR/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/data2/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200608_nir150_new_copy',
        '/data2/johnnyalg1/data/FR/mi8/mi8_mdong_112x112_20200608_nir150_new_copy',
        '/data2/johnnyalg1/data/FR/mi8/20200611_nir150_new_copy',
        '/data2/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_A_nir150',
        '/data2/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_B_nir150',
        '/data2/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_C_nir150',
        '/data2/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_D_nir150',
        '/data2/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_E_nir150',
        '/data2/johnnyalg1/data/FR/mi8/mi8_liu_112x112_20200709_F_nir150',
        '/data2/johnnyalg1/data/FR/mi8/mi8_mdong_112x112_20200803_A_nir150',
        '/data2/johnnyalg1/data/FR/mi8/mi8_mdong_112x112_20200803_B_nir150',
        '/data2/johnnyalg1/data/FR/mi8/mi8_mdong_112x112_20200803_C_nir150',
        ]

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



for path in IMAGE_ROOT:
    #targetdir = '/mnt/sdd/johnnysun/FRdata'
    targetdir = '/datadisk/johnnyalg2/data/FR/FRdata_v30_3'
    bname = basename(path)
    dname = dirname(path)
    prefolder = basename(dname)
    os.system( f'cd {targetdir}/{prefolder}')
    print(f'cd {targetdir}/{prefolder}')
    print(bname)
    if not os.path.exists( f'{targetdir}/{prefolder}'):
        print( f' Not Found {targetdir}/{prefolder}' )
    #os.system( f'tar -zcf {targetdir}/{prefolder}/{bname}.tar.gz {bname}' )
    #sys.exit(1)
    os.system( f'tar -zxf {bname}.tar.gz' )

#os.system(f'cd /mnt/sdd/johnnysun')
#os.system(f' tar -cf FRdata.tar.gz FRdata' )
