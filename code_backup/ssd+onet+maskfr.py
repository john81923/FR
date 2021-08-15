import sys
import os
import pickle
import cv2
current_path=os.getcwd()
sys.path.append(current_path+'/prepostprocess')
sys.path.append(current_path+'/prepostprocess/kneron_preprocessing')
sys.path.append(current_path+'/kneron_globalconstant')
sys.path.append(current_path+'/kneron_globalconstant/base')
sys.path.append(current_path+'/kneron_globalconstant/kneron_utils')
from fd_ssd.fd_ssd_runner import FdSsdRunner
from onet.onet_runner import OnetRunner
from alignment.alignment_runner import AlignmentRunner
from mask_fr.mask_fr_runner import MaskFrRunner
from function.function_runner import FunctionRunner

from tqdm import tqdm, trange
import numpy as np
fd_ssd_runner = FdSsdRunner(model_path='/mnt/models/FD_models/fd_ssd/011521/ssd7_mask_epoch-57_loss-0.1850_val_loss-0.1916.onnx', anchor_path='/mnt/models/FD_models/fd_ssd/011521/anchor_face_ssd7s_cfg2.npy', input_shape=[200, 200], score_thres=0.5, only_max=0, iou_thres=0.35)
onet_runner = OnetRunner(gray_scale=0, model_path='/mnt/models/Landmark_Models/face_5pt/121820/rgb2gray_Laplacian_onet_base_onet-combined-246_loss-0.7126_val_loss-0.4551.onnx', input_shape=[56, 56], keep_ap=1, pad_center=1)
alignment_runner = AlignmentRunner(shape=[112, 112], src=[[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
mask_fr_runner = MaskFrRunner(model_path='/mnt/models/FR_models/MASK_FR/011921_maskfr/kface_dual_model_tiny_v6_8_9_converted.onnx', input_shape=[112, 112])
function_runner = FunctionRunner(type=15, score1=0.75, score2=0.3, _comment='score1 is for normal face, score2 is for mask face. Higher than score will be kept')
input_video_path = '/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/applications/maskfr_0222/inference.mp4'
#img_path=['/home/craig/home_craig/kneron_tw_models/framework/face_recognition/mask_fr_fifo/test_input/test_no_mask.jpg',
#          '/home/craig/home_craig/kneron_tw_models/framework/face_recognition/mask_fr_fifo/test_input/test_no_mask.jpg']


def mask_fr(img_path):
    out_0_0 = fd_ssd_runner.run(img_path)
    #print(out_0_0)
    out_1_0 = onet_runner.run(img_path,out_0_0)
    #print(out_1_0)
    #exit(0)
    out_4_0, out_4_1 = function_runner.run(img_path,[out_0_0,out_1_0])
    #print(out_4_0)
    #print(out_4_1)
    #exit(0)
    out_2_0 = alignment_runner.run(img_path,out_4_1)
    out_3_0 = mask_fr_runner.run(img_path,[out_2_0,out_4_0])
    #print(out_3_0)
    return out_3_0, out_4_0

def pseudo_mask_fr(img_path):
    out_0_0 = fd_ssd_runner.run(img_path)
    out_0_0[0][-1] = 2.0
    #print(out_0_0)
    out_1_0 = onet_runner.run(img_path,out_0_0)
    #print(out_1_0)
    #exit(0)
    out_4_0, out_4_1 = function_runner.run(img_path,[out_0_0,out_1_0])
    out_2_0 = alignment_runner.run(img_path,out_4_1)
    out_3_0 = mask_fr_runner.run(img_path,[out_2_0,out_4_0])
    return out_3_0

def video_preprocess_(video_path, output_video_path='./output_video.avi'):
    videoCapture = cv2.VideoCapture(video_path)

    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    video_out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (640,360))

    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    #print('video_name: ', video_name)
    print('fps: ', int(fps))
    print('size: ', size)
    print('frame_number: ', fNUMS)
    return videoCapture , video_out, fNUMS, size

def search_top3(enroll_dict,embedding_unlock):
    dict_distance ={}
    for key, embedding_register in enroll_dict.items():
        diff = np.subtract(embedding_register, embedding_unlock)
        dist = np.sum(np.square(diff))
        dict_distance[key] = dist
    top1 = min(dict_distance.items(), key=lambda x: x[1])
    del dict_distance[top1[0]]
    top2 = min(dict_distance.items(), key=lambda x: x[1])
    del dict_distance[top2[0]]
    top3 = min(dict_distance.items(), key=lambda x: x[1])
    return top1, top2, top3
if __name__ == '__main__':
    fh = open('/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/applications/maskfr_0222/dict_no_mask.pkl', 'rb')
    dict_no_mask = pickle.load(fh)
    fh2 = open('/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/applications/maskfr_0222/dict_mask.pkl', 'rb')
    dict_mask = pickle.load(fh2)
    videoCapture, video_out, fNUMS, size = video_preprocess_(input_video_path, '/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/applications/maskfr_0222/video_out/output_video.avi')
    success, frame = videoCapture.read()
    progress = tqdm(total=fNUMS)
    counter = 0
    while success:
        if counter < 60:
            success, frame = videoCapture.read()  # next_frame
            counter +=1
            progress.update(1)
            continue
        elif counter > 10000:
            break
        else:
            temp_path = '/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/applications/maskfr_0222/temp/temp.png'
            cv2.imwrite(temp_path, frame)
            embedding_unlock, bbox = mask_fr(temp_path)
            for i in range(len(bbox)):
                box = bbox[i] #final_bbox[i]
                #print(box[0])
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (255, 255, 0), 4)
                #cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[3]), int(box[2])), (255, 255, 0), 4)
                #label = f"{probs[i]:.2f}"
                #label = f"{class_names[box[5]]}: {box[4]:.2f}"
                #label = f"{box[4]:.2f}"
                if box[5] == 2.0:
                    label = f"mask"
                    top1, top2, top3 = search_top3(dict_mask,embedding_unlock)
                else:
                    label = f"no mask"
                    top1, top2, top3 = search_top3(dict_no_mask,embedding_unlock)

                top1_o = str(top1[0]) +" : "+ f"{top1[1]:.4f}"
                top2_o = str(top2[0]) +" : "+ f"{top2[1]:.4f}"
                top3_o = str(top3[0]) +" : "+ f"{top3[1]:.4f}"
                cv2.putText(frame, label,
                            (int(box[0]) +3, int(box[1]) -5),
                            cv2.FONT_HERSHEY_PLAIN,
                            3,  # font scale
                            (0, 225, 225),
                            3, cv2.LINE_AA)

                if box[5] == 2.0:
                    cv2.putText(frame, label,
                            (50, 40),
                            cv2.FONT_HERSHEY_PLAIN,
                            3,  # font scale
                            (0, 0, 225),
                            3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, label,
                            (50, 40),
                            cv2.FONT_HERSHEY_PLAIN,
                            3,  # font scale
                            (0, 225, 225),
                            3, cv2.LINE_AA)
                cv2.putText(frame, top1_o,
                        (50, 80),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,  # font scale
                        (0, 225, 225),
                        3, cv2.LINE_AA)
                cv2.putText(frame, top2_o,
                        (50, 120),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,  # font scale
                        (0, 225, 225),
                        3, cv2.LINE_AA)
                cv2.putText(frame, top3_o,
                        (50, 160),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,  # font scale
                        (0, 225, 225),
                        3, cv2.LINE_AA)
            frame = cv2.resize(frame, (640, 360))

            video_out.write(frame)  # write out video
            success, frame = videoCapture.read()  # next_frame
            progress.update(1)
            counter +=1
#if __name__ == '__main__':
    '''
    dict_no_mask = {}
    dict_mask = {}

    enroll_path = '/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/applications/maskfr_0222/Enroll_Img'
    for enroll_folder in os.listdir(enroll_path):
        enroll_folder_path = os.path.join(enroll_path,enroll_folder)
        for enroll_file in os.listdir(enroll_folder_path):
            enroll_file_path = os.path.join(enroll_folder_path,enroll_file)
            embedding_no_mask = mask_fr(enroll_file_path)
            embedding_with_mask = pseudo_mask_fr(enroll_file_path)
            dict_no_mask[enroll_folder] = embedding_no_mask[0]
            dict_mask[enroll_folder] = embedding_with_mask[0]
            if len(embedding_no_mask[0]) != 256:
                print(enroll_folder)
                print("embedding issue for no mask embedding")
                exit(0)
            if len(embedding_with_mask[0]) != 256:
                print(enroll_folder)
                print("embedding issue for with mask embedding")
                exit(0)
            #print(len(embedding_no_mask[0]))
            #print(dict_no_mask)
    print(len(dict_mask))

    print(len(dict_no_mask))
    fh = open('/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/applications/maskfr_0222/dict_no_mask.pkl', 'wb')
    pickle.dump(dict_no_mask, fh)

    fh2 = open('/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/applications/maskfr_0222/dict_mask.pkl', 'wb')
    pickle.dump(dict_mask, fh2)
    '''

            #pseudo_mask_fr(enroll_file_path)
        #config.output_image_path = os.path.join(output_folder,filename)[:-4]+'_out.png'
        #print(image_path)
        #main(image_path)

#cv2.rectangle(orig_image, (int(box[0]*resize_ratio), int(box[1]*resize_ratio)), (int(box[2]*resize_ratio), int(box[3]*resize_ratio)), (0, 0, 255), 4)
