from tqdm import tqdm
import sys
import os
from os.path import basename
import shutil

if __name__ =='__main__':

    ratio = 0.26
    ratio_test_dir = '/data2/johnnyalg1/data/FR/insightface/ratio_test'
    ratio_test_dir = f'{ratio_test_dir}_occ{ratio}'
    if os.path.exists(ratio_test_dir):
        shutil.rmtree(ratio_test_dir)
        os.makedirs(ratio_test_dir)
    else:
        os.makedirs(ratio_test_dir)

    insight_face_occ_dir = '/data2/johnnyalg1/data/FR/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign_occ'
    #img_path = '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign/77279/77279_4266239_0.png'
    img_paths = os.walk( f'{insight_face_occ_dir}/')
    img_name_list = [ ]
    count = 0
    total_exceed = 500
    progress = tqdm(total=total_exceed)
    for root, dirs, files in img_paths:
        for name in files:
            img_fname = os.path.join(root, name)
            bname = basename(img_fname)
            fratio = int( basename(img_fname).split('.')[-2] )*0.01
            if fratio>ratio:
                shutil.copyfile(img_fname, f'{ratio_test_dir}/{bname}')
                #print(f'{ratio_test_dir}/{bname}')
                count += 1
                progress.update(1)
                if count > total_exceed:
                    sys.exit(1)
