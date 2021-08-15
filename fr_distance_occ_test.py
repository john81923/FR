import sys
import os
current_path='/data/johnny.sun/LPDR/kneron_hw_models'
sys.path.append(current_path)
sys.path.append(current_path+'/prepostprocess')
sys.path.append(current_path+'/prepostprocess/kneron_preprocessing')
sys.path.append(current_path+'/kneron_globalconstant')
sys.path.append(current_path+'/kneron_globalconstant/base')
sys.path.append(current_path+'/kneron_globalconstant/kneron_utils')
from prepostprocess import kneron_preprocessing as kp
from fd_ssd.fd_ssd_runner import FdSsdRunner
from onet.onet_runner import OnetRunner
from alignment.alignment_runner import AlignmentRunner
from fr.fr_runner import FrRunner
from function.function_runner import FunctionRunner

from glob import glob
import numpy as np
import shutil
from os.path import basename, splitext
import cv2
def compute_dist(embedding_register, embedding_unlock):
    diff = np.subtract(embedding_register, embedding_unlock)
    dist = np.sum(np.square(diff))*0.85
    return dist

fd_ssd_runner = FdSsdRunner(model_path='/mnt/models/FD_models/fd_ssd/121620/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825_opt.onnx', anchor_path='/mnt/models/FD_models/fd_ssd/121620/anchor_face_ssd7s_cfg2.npy', input_shape=[200, 200], score_thres=0.5, only_max=0, iou_thres=0.35)
onet_runner = OnetRunner(gray_scale=0, model_path='/mnt/models/Landmark_Models/face_5pt/041421/onet-fp-softmaxadjust.onnx', input_shape=[56, 56], keep_ap=1, pad_center=1)
alignment_runner = AlignmentRunner(shape=[112, 112], src=[[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
newfr = '/mnt/models/FR_models/FR_onnx_0609reproduce/0706_resnet_mi_v32_2_ckpt303/checkpoint-r50-I112-E256-e0069-av0.9997_0.9998.onnx.opt.onnx'
oldfr = '/mnt/models/FR_models/FR_onnx_0609reproduce/0617_resnet_mi_v31_2_ckpt303/checkpoint-r50-I112-E256-e0069-av0.9995_1.0000.onnx.opt.onnx'
fr_runner_new = FrRunner(model_path=newfr, input_shape=[112, 112])
fr_runner_old = FrRunner(model_path=oldfr, input_shape=[112, 112])

function_runner = FunctionRunner(type=15, score1=0.75, score2=0.3, _comment='score1 (default 0.75) is for normal face, score2 (default 0.3) is for mask face. Higher than score will be kept')

src_dir = '/data1/johnnyalg1/data/FR/occ_distance'
dst_folder_list = ['nir_occ_test' , 'fr_kn_occ', 'fr_kn_light' ]
inputid = 1

dst_dir = f'/data1/johnnyalg1/data/FR/occ_distance/{dst_folder_list[inputid]}_fr_dist'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
else:
    shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

# gen reg+test pair
#img_list = glob(f'{src_dir}/fr_kn_occ_realign/*') + [f'{src_dir}/fr_kn_light'] + glob(f'{src_dir}/nir_occ_test_jpg/*')
img_list =  [ glob(f'{src_dir}/nir_occ_test_jpg/*'),  glob(f'{src_dir}/fr_kn_occ_realign/*'), [f'{src_dir}/fr_kn_light'] ]
img_list = img_list[ inputid ]

img_pair_list = []
for imgdir in  img_list:
    test_list = []
    img_test_name = glob(f'{imgdir}/*')
    for img_item in img_test_name:
        if 'reg' not in img_item:
            test_list.append(img_item)
        else:
            reg_img = img_item
    img_pair_list.append( [reg_img, test_list] )

for reg_img, test_list in img_pair_list:
    regbname = basename(splitext(reg_img)[0])
    dst_folder = f'{dst_dir}/{regbname}'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    if 'align' not in regbname:
        out_0_0 = fd_ssd_runner.run(reg_img)
        out_1_0, out_1_1 = onet_runner.run(reg_img, out_0_0)
        out_4_0, out_4_1, out_4_2 = function_runner.run(reg_img, [out_0_0,out_1_0,out_1_1])
        out_2_0 = alignment_runner.run(reg_img, out_4_1)
    else:
        out_2_0 = []

    reg_encoding_old = fr_runner_old.run(reg_img)
    reg_encoding_new = fr_runner_new.run(reg_img)
    regvis = cv2.imread(reg_img)
    cv2.imwrite(f'{dst_folder}/{regbname}.jpg', regvis)

    for img_path in test_list:
        testbname = basename(splitext(img_path)[0])
        if 'align' not in testbname:
            out_0_0 = fd_ssd_runner.run(img_path)
            out_1_0, out_1_1 = onet_runner.run(img_path, out_0_0)
            out_4_0, out_4_1, out_4_2 = function_runner.run(img_path, [out_0_0,out_1_0,out_1_1])
            out_2_0 = alignment_runner.run(img_path, out_4_1)
        else:
            out_2_0 = []
        encoding_old = fr_runner_old.run(img_path)
        encoding_new = fr_runner_new.run(img_path)

        dist_old = compute_dist(reg_encoding_old, encoding_old)
        dist_new = compute_dist(reg_encoding_new, encoding_new)

        print( f'oldfr-{dist_old:.2f}_newfr-{dist_new:.2f}_{testbname}' )
        testimg = cv2.imread(img_path)
        cv2.imwrite( f'{dst_folder}/{testbname[:-4]}_oldfr-{dist_old:.2f}_newfr-{dist_new:.2f}.jpg', testimg)
