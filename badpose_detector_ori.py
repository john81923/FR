# This script is based on current kneron SDK bad pose filter, which take LM as input and output if bad pose detected (2020.03.11) 
import os
from math import sqrt
import onet_infer 
from PIL import Image
import shutil
def weird_division(n, d):
    return n / d if d else 0

def verifyObtuseAngle(a_x, a_y, b_x, b_y, c_x, c_y):
    return (b_x - a_x) * (b_x - c_x) + (b_y - a_y) * (b_y - c_y)


def verticalDist(nose_x, nose_y, a_x, a_y, b_x, b_y):   
    nose2bDistSq = (nose_y - b_y) * (nose_y - b_y) + (nose_x - b_x) * (nose_x - b_x)
    abcbPointCrossSq = ((b_x - a_x) * (b_x - nose_x) + (b_y - a_y) * (b_y - nose_y) ) * ((b_x - a_x) * (b_x - nose_x) + (b_y - a_y) * (b_y - nose_y) )
    abDistSq = (a_y - b_y)*(a_y - b_y) + (a_x - b_x) * (a_x - b_x)
    verticalDistResult = sqrt(nose2bDistSq - weird_division(float(abcbPointCrossSq) , float(abDistSq)) )
    return verticalDistResult


def checkBadPose(landmarks):
    a_x = landmarks[0]
    a_y = landmarks[1]
    b_x = landmarks[2] # right_eye(actual left)
    b_y = landmarks[3]
    c_x = landmarks[4]  # nose
    c_y = landmarks[5]
    d_x = landmarks[6] # left_mouth(actual right)
    d_y = landmarks[7]
    e_x = landmarks[8] # right_mouth(actual left)
    e_y = landmarks[9]   

    # verify nose is out of bound or not. (obtuse angle)
    nose2LeftEyeAngleBad = verifyObtuseAngle(c_x, c_y, a_x, a_y, b_x, b_y) < 0
    nose2RightEyeAngleBad = verifyObtuseAngle(c_x, c_y, b_x, b_y, a_x, a_y) < 0

    if nose2LeftEyeAngleBad or nose2RightEyeAngleBad:
        return True
    
    # get h_1_div_h_2 and w_1_div_w_2
    h_1 = verticalDist(c_x, c_y, a_x, a_y, b_x, b_y)
    h_2 = verticalDist(c_x, c_y, e_x, e_y, d_x, d_y)
    w_1 = verticalDist(c_x, c_y, d_x, d_y, a_x, a_y)
    w_2 = verticalDist(c_x, c_y, b_x, b_y, e_x, e_y)
    h_1_div_h_2 = weird_division(h_1, h_2)
    w_1_div_w_2 = weird_division(w_1, w_2)
    #h_1_div_h_2 = h_1 / h_2
    #w_1_div_w_2 = w_1 / w_2

    # to modified if necessary.
    verticalBad = h_1_div_h_2 < 0.25 or h_1_div_h_2 > 6     # 0.25 <= h1/h2 <= 6
    horizontalBad = w_1_div_w_2 < 0.14 or w_1_div_w_2 > 7  # 0.14 <= w1/w2 <= 7

    return horizontalBad or verticalBad




if __name__ == "__main__":
    #ONET = onet_infer.onet_detector
    detector = onet_infer.onet_detector("kneron_face_recognition/ONET/models/onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5")
    img_dir='D:/face_recognition_datasets/insightface/ms1m-retinaface-t1-112x112'
    save_dir='D:/face_recognition_datasets/insightface/bad'
    for img_dir_i in os.listdir(img_dir):
        for image_file_i in os.listdir(os.path.join(img_dir,img_dir_i)):
            image_file = os.path.join(img_dir,img_dir_i, image_file_i)
            image = Image.open(image_file).convert("RGB")
            w, h =image.size
            landmarks = detector.run(image, [0, 0, w, h])
            print(image_file_i,landmarks)
            Bad = checkBadPose(landmarks)
            dst_dir = os.path.join(save_dir,img_dir_i)
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            if Bad:
                dst_path = os.path.join(dst_dir, image_file_i)
                shutil.copyfile(image_file, dst_path)
            print(image_file_i,Bad)

