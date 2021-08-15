import sys
import os
current_path = '/mnt/sdd/johnnysun/kneron_data_argumentation/FRcleaningflow/kneron_hw_models'#os.getcwd()
sys.path.append(current_path)
sys.path.append(current_path+'/prepostprocess')
sys.path.append(current_path+'/prepostprocess/kneron_preprocessing')
sys.path.append(current_path+'/kneron_globalconstant')
sys.path.append(current_path+'/kneron_globalconstant/base')
sys.path.append(current_path+'/kneron_globalconstant/kneron_utils')
sys.path.append('/mnt/sdd/johnnysun/kneron_data_argumentation/FRcleaningflow')
import argparse
from prepostprocess import kneron_preprocessing as kp
from fd_ssd.fd_ssd_runner import FdSsdRunner
from onet.onet_runner import OnetRunner
from alignment.alignment_runner import AlignmentRunner
from fr.fr_runner import FrRunner
from function.function_runner import FunctionRunner
from os.path import basename, dirname
from glob import glob
from tqdm import tqdm
import shutil
import numpy as np
import cv2
import threading
from os.path import splitext

fd_ssd_runner = FdSsdRunner(model_path='/mnt/models/FD_models/fd_ssd/121620/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825_opt.onnx', anchor_path='/mnt/models/FD_models/fd_ssd/121620/anchor_face_ssd7s_cfg2.npy', input_shape=[200, 200], score_thres=0.5, only_max=0, iou_thres=0.35)
onet_runner = OnetRunner(gray_scale=0, model_path='/mnt/models/Landmark_Models/face_5pt/041421/onet-fp.onnx', input_shape=[56, 56], keep_ap=1, pad_center=1)
alignment_runner = AlignmentRunner(shape=[112, 112], src=[[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
function_runner = FunctionRunner(type=15, score1=0.78, score2=0.3, _comment='score1 (default 0.75) is for normal face, score2 (default 0.3) is for mask face. Higher than score will be kept')
#

from badpose_detector import  checkBadPose_strict_fix_runner
from get_LM_angle_ratio import checkBadAngle_runner
from occlusion_checker import OcclusionRunner
#
from occlusion_filter import OccFilterRunner
#
from sunglasses_filter import MaskSunglassRunner
#
from posequality_filter import PoseQualityYrpRunner

badangle_runner = checkBadAngle_runner( threshold_ratio=1.5) # True for passing the checker
# Face Occlusion
occlusion_ratio_runner = OcclusionRunner( occ_ratio=0.2, y_magnitude_threshold=10 ) # True for passing the checker
#
#occlusion_filter_runner = OccFilterRunner()
#
mask_sunglass_runner = MaskSunglassRunner()
#
posequality_filter = PoseQualityYrpRunner()
'''
IMAGE_ROOT = [

        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/vloggerface/align_data_realign_bad_pose', #vlog
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign', #insight strictfix  (--glass), Yuanlin
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200608_nir150_new_copy',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_mdong_112x112_20200608_nir150_new_copy',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/20200611_nir150_new_copy',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_A_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_B_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_C_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_D_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_E_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_liu_112x112_20200709_F_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_mdong_112x112_20200803_A_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_mdong_112x112_20200803_B_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/mi8/mi8_mdong_112x112_20200803_C_nir150',

        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_liu_112x112_20200608_nir150_new_copy',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_mdong_112x112_20200608_nir150_new_copy',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/20200611_nir150_new_copy',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_liu_112x112_20200709_A_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_liu_112x112_20200709_B_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_liu_112x112_20200709_C_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_liu_112x112_20200709_D_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_liu_112x112_20200709_E_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_liu_112x112_20200709_F_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_mdong_112x112_20200803_A_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_mdong_112x112_20200803_B_nir150',
        '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Asian_Raw_Data_clean_light_augmented/mi8_mdong_112x112_20200803_C_nir150',
        ]
'''
IMAGE_ROOT = [
    '/mnt/atdata/FacialRecognition/Guizhou_NIR_dataset_kfr_0722',
]
# result as unvalid bounding box, check base preprocess_hw size error
#checkthisimg = '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Henan_WhiteBlack/white850/1.27/zhenren/male-songzhouke-13/NIR_2021-01-27_16-26-02-2.png'
#checkthisimg2 = '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/FR_Henan_WhiteBlack/black940/1.26/zhenren/male-zhouyanzhi-14/NIR_2021-01-26_12-12-49-1.png'

#out_0_0 = fd_ssd_runner.run(img_path[0])
#out_1_0, out_1_1 = onet_runner.run(img_path[0], out_0_0)
#out_4_0, out_4_1, out_4_2 = function_runner.run(img_path[0], [out_0_0,out_1_0,out_1_1])
#out_2_0 = alignment_runner.run(img_path[0], out_4_1)
def iterte_align( data_path ):
    empty_count = 0
    total = 0
    dataid_list = glob(f'{data_path}/*')
    total = len(dataid_list)
    interval = total//splitnumb
    #sys.exit(1)

    print('splitnumb:', splitnumb, ', runid:', runid)
    current_list = dataid_list[runid*interval:(runid+1)*interval]
    progress = tqdm(total=len(current_list))
    counter = 0
    for dataidpath in current_list:
        counter += 1
        progress.update(1)

        for data_name in glob(f'{dataidpath}/*'):
            #print(data_name)

            if splitext(data_name)[1] not in ['.png', '.jpg']:
                continue
            empty_count += do_inf(data_name)
        #sys.exit(1)
    print(f" Empty_count [{data_path}]: {empty_count}/{total}")


def do_inf(img_path ):
    empty_count = 0
    bname = basename(img_path)
    id_name = basename(dirname(img_path))
    dst_name = f'{dir_name}/{id_name}'

    if fd_on:
        out_0_0 = fd_ssd_runner.run(img_path)
    else:
        out_0_0 = [[0,0,112,112, 1, 1.0]]
    #print(out_0_0)

    if len(out_0_0) > 1 or len(out_0_0)==0 :
        return 1
    if out_0_0[0][-2] < 0.78 or  out_0_0[0][-1] != 1.0:
        return 1
    #
    out_1_0, out_1_1 = onet_runner.run(img_path, out_0_0)
    #print('out_1_0, out_1_1', out_1_0, out_1_1)
    out_4_0, out_4_1, out_4_2 = function_runner.run(img_path, [out_0_0,out_1_0,out_1_1])
    #print('out_4_1', out_4_1)
    out_2_0 = alignment_runner.run(img_path, out_4_1)
    #print('out_2_0',  np.asarray(out_2_0, dtype=np.float32).shape)
    #
    bad_angle_hor, bad_angle_ver  = badangle_runner.run(out_4_1)
    if bad_angle_hor>1.7 or bad_angle_ver>1.7:
        return 1
    #print(bad_angle_hor, bad_angle_ver)
    posequality_facescore_yrp = posequality_filter.run(img_path)
    if posequality_facescore_yrp[0]<0.35 or any(np.absolute(posequality_facescore_yrp[1])>35):
        return 1
    #print('posequality_facescore_yrp', posequality_facescore_yrp)
    maks_glass_sugnlass = mask_sunglass_runner.run(img_path)
    if maks_glass_sugnlass[0]==1 or maks_glass_sugnlass[2]==1:
        return 1

    if out_2_0:
        if not os.path.exists( f'{dst_name}' ):
            os.makedirs(f'{dst_name}')
        out_2_0 = np.squeeze( np.asarray(out_2_0, dtype=np.float32) )
        fmean =  out_2_0.mean()
        if fmean < 42 or fmean > 215 :
            return 1
        #print( out_2_0.mean() )
        out_2_0 = cv2.cvtColor( out_2_0, cv2.COLOR_RGB2BGR)
        cv2.imwrite( f'{dst_name}/{bname}', out_2_0)
    else:
        #bad_img = cv2.imread(img_path)
        #cv2.imwrite( f'{bad_folder}/{bname}', bad_img)
        empty_count += 1
    return empty_count



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script so useful.')
    parser.add_argument("--splitnumb", type=int, default=1)
    parser.add_argument("--runid", type=int, default=0)


    args = parser.parse_args()
    splitnumb = args.splitnumb
    runid = args.runid

    fd_on = True
    for path in IMAGE_ROOT:
        print('load from: ',path)
        if not os.path.exists(path):
            raise ValueError(f"path not exist: {path}")

        dir_name = f'{path}_realign'
        bad_folder = f'{path}_bad'
        print(f'dst : {dir_name}')
        if not os.path.exists(dir_name):
            print(f'create : {dir_name}')
            os.makedirs(dir_name)

        #if not os.path.exists(bad_folder):
        #    os.makedirs(bad_folder )
        #else:
        #    shutil.rmtree(bad_folder)
        #    os.makedirs(bad_folder )
        iterte_align(path)
