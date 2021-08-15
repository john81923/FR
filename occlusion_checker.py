import cv2
from glob import glob
import os
from os.path import basename, dirname, splitext
import sys
import shutil
from tqdm import tqdm
import threading


class OcclusionRunner:
    def __init__( self, occ_ratio, y_magnitude_threshold=10 ):
        self.occ_ratio = occ_ratio
        self.y_magnitude_threshold = y_magnitude_threshold

    def run( self, image ):
        if isinstance(image, str):
            image = cv2.imread(image)
        y_m = self.y_magnitude(image)
        normal = 1
        dark_cnt = 0
        XY = [25,48,85,108]
        for i in range(y_m.shape[0]):
            for j in range(y_m.shape[1]):
                if i>=XY[1] and j>=XY[0] and i<=XY[3] and j<=XY[2] and y_m[i][j] < self.y_magnitude_threshold:
                    dark_cnt +=1
        if dark_cnt > (XY[3] - XY[1] + 1) * (XY[2] - XY[0] + 1)*self.occ_ratio:
            return False
        else:
            return True

    def y_magnitude(self, bgr_img):
        yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv_img)
        return y


def y_magnitude(bgr_img):
    yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)
    return y

def insight_fr_occlusion(y_m, y_magnitude_threshold=10):
    normal = 1
    dark_cnt = 0
    XY = [25,48,85,108]
    for i in range(y_m.shape[0]):
        for j in range(y_m.shape[1]):
            if i>=XY[1] and j>=XY[0] and i<=XY[3] and j<=XY[2] and y_m[i][j] < y_magnitude_threshold:
                dark_cnt +=1
    #if dark_cnt > (XY[3] - XY[1] + 1) * (XY[2] - XY[0] + 1)*ratio:
    #    normal = 0
    #    print( f"Occ result normal/occluded:{normal}, Ratio = {dark_cnt / ((XY[3] - XY[1] + 1)*(XY[2] - XY[0] + 1))}, Threshold = {ratio} ")
    return  dark_cnt / ((XY[3] - XY[1] + 1) * (XY[2] - XY[0] + 1))

def action(img_path_list, progress):
    for img_path in img_path_list:
        img_path = img_path.split('\n')[0]
        progress.update(1)
        img = cv2.imread(img_path)
        y_m = y_magnitude(img)
        occ = insight_fr_occlusion(y_m)

        dname = dirname(img_path).split('/')[-1]
        bname, ext = splitext( basename(img_path) )
        #if occ<ratio_threshold:
        occ = f"{occ:.2f}"[1:]

        if not os.path.exists( os.path.join(filtered_dst, dname) ):
           os.makedirs( os.path.join(filtered_dst, dname) )
        if not os.path.exists( f'{filtered_dst}/{dname}/{bname}_occ{occ}{ext}' ):
            shutil.copyfile(img_path, f'{filtered_dst}/{dname}/{bname}_occ{occ}{ext}' )


if __name__ =='__main__':

    insight_face_dir = '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign_060921realign'
    filtered_dst = '/datadisk/johnnyalg2/data/FR/FRdata_v30_3/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign_060921realign_occ_all'
    #ratio_threshold = .2
    #filtered_dst = filtered_dst+f'{ratio_threshold:.1f}'

    img_paths = os.walk( f'{insight_face_dir}/')
    if False : # load data
        with open( 'ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign_060921realign_datadisk.txt', 'w') as fp:
            idfolerlist = glob(f'{insight_face_dir}/*')
            for idfolder in tqdm(idfolerlist):
                filenamelist = glob(f'{idfolder}/*')
                for img_path in filenamelist:
                    fp.write(img_path + '\n')
                    #img_name_list.append( img_path )
    else:
        with open( 'ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign_060921realign_datadisk.txt', 'r') as fp:
            img_name_list = fp.readlines()

            print(filtered_dst)
            thread_numb = 8
            data_len = len(img_name_list)
            ofst= data_len//thread_numb
            thread_list = []
            progress = tqdm(total=data_len)
            for i in range(thread_numb):
                thread_list.append( threading.Thread( target=action, args=(img_name_list[i*ofst:(i+1)*ofst], progress )))
                thread_list[i].start()

            for i in range(thread_numb):
                thread_list[i].join()
