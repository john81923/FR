import glob
from PIL import Image
import numpy as np
import sys, os
import matplotlib.pyplot as plt

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


def lmk_to_angle_ratio(landmark):
    landmark = landmark[:10]
    landmark = np.reshape(landmark, (5, 2))
    v1 = landmark[0,:] - landmark[2,:]
    v2 = landmark[3,:] - landmark[2,:]
    left_side_angle = py_ang(v1, v2)

    v3 = landmark[1,:] - landmark[2,:]
    v4 = landmark[4,:] - landmark[2,:]
    right_side_angle = py_ang(v3, v4)

    if left_side_angle > right_side_angle:
        big_ang, small_ang = left_side_angle, right_side_angle
    else:
        big_ang, small_ang = right_side_angle, left_side_angle

    return  big_ang/small_ang #the cutoff angle_ratio is 1.5

def checkBadAngle(landmarks, threshold_ratio=1.5):
    if not landmarks:
        return False
    landmarks = landmarks[0]
    ratio = lmk_to_angle_ratio(landmarks)
    return False if ratio>threshold_ratio else True


class checkBadAngle_runner:
    def __init__(self, threshold_ratio=1.5 ):
        self.threshold_ratio = threshold_ratio

    def py_ang(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'    """
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)


    def lmk_to_angle_ratio(self, landmark):
        landmark = landmark[:10]
        landmark = np.reshape(landmark, (5, 2))
        v1 = landmark[0,:] - landmark[2,:]
        v2 = landmark[3,:] - landmark[2,:]
        left_side_angle = self.py_ang(v1, v2)

        v3 = landmark[1,:] - landmark[2,:]
        v4 = landmark[4,:] - landmark[2,:]
        right_side_angle = self.py_ang(v3, v4)

        if left_side_angle > right_side_angle:
            big_ang, small_ang = left_side_angle, right_side_angle
        else:
            big_ang, small_ang = right_side_angle, left_side_angle

        return  big_ang/small_ang #the cutoff angle_ratio is 1.5

    def run(self, landmarks):
        if not landmarks:
            return False
        landmarks = landmarks[0]
        ratio = self.lmk_to_angle_ratio(landmarks)
        return False if ratio>self.threshold_ratio else True




if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath("../"))
    from fd_ssd.fd_ssd_runner import FdSsdRunner
    from onet.onet_runner import OnetRunner

    ## model file
    ssd_model_file = '/Users/jane/Downloads/kneron_sd_models/applications/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5'
    anchor_path = '/Users/jane/Downloads/kneron_sd_models/applications/models/anchor_face_ssd7s_cfg2.npy'
    onet_model_file = '/Users/jane/Downloads/kneron_sd_models/applications/models/rgb2gray_Laplacian_onet_base_onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5'


    SSD = FdSsdRunner(model_path = ssd_model_file, anchor_path = anchor_path, input_shape=(200, 200))
    ONET = OnetRunner(model_path = onet_model_file, gray_scale=False)

    def get_lmks(img):
        dets = SSD.run(img)
        lmk = ONET.run(img, dets[0])
        return dets, lmk

    imgs = glob.glob('/Users/jane/Downloads/kneron_sd_models/applications/test_images/*') ##<-----input images here

    #fig = plt.figure(figsize=(16, 5*len(imgs)))
    for i, img in enumerate(imgs):
        image = Image.open(img)
        image = np.array(image)
        image = np.squeeze(image)

        dets, lmk = get_lmks(image) #<--- get LM
        angle_ratio = lmk_to_angle_ratio(lmk)
        if (angle_ratio<1.5):

            print("landmark value in original image: ", lmk)
