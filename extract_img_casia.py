import os
import cv2
import tensorflow as tf
print(tf.__version__)
from PIL import Image
import numpy as np
from skimage import transform as trans
import cv2
from FD_SSD.SSD_infer import ssd_detector
from ONET.onet_infer import onet_detector

face_detector = ssd_detector("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)

landmark_detector = onet_detector("./ONET/models/onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5")


def create_folder (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def pre_process(image_path):
    dets = face_detector.run(image_path)
    ret = None
    if len(dets)>0:
        #for i in range(len(dets)): # only 1 face
        bbox =(dets[0][0], dets[0][1], dets[0][2], dets[0][3])

        image = Image.open(image_path).convert("RGB")
        img = np.array(image)
        #landmark = landmark_detector.run(img, bbox)
        landmark = landmark_detector.run(image_path, dets[0])

        #print("landmark: ", landmark)

        image_size=(112,112)
        if np.size(landmark) != 10:
            landmark = landmark[:10]
        assert np.size(landmark) == 10
        landmark = np.reshape(landmark, (5, 2))
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0

        dst = np.squeeze(np.asarray(landmark, np.float32))
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        #print("M: ", M)
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

        assert len(image_size) == 2
        ret = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        #ret_size = ret.shape
        #print("ret_size: ", ret_size)
        #ret = np.expand_dims(ret, axis=-1)
        return len(dets), ret
    else:
        return len(dets), ret



if __name__ == "__main__":
    root = '/mnt/sdc/craig/CASIA_NIR/NIR'
    target = '/mnt/sdc/craig/CASIA_NIR/NIR_kfr_0304/'
    create_folder(target)
    root_folder = os.listdir(root)
    for folder in root_folder:
        src_folder = os.path.join(root,folder)
        tgt_folder = os.path.join(target,folder)
        create_folder(tgt_folder)
        image_set = os.listdir(src_folder)
        for image in image_set:
            tgt_image_path = os.path.join(tgt_folder,image)[:-4] +'.png'
            if os.path.isfile(tgt_image_path):
                pass
            else:
                src_image_path  = os.path.join(src_folder,image)
                items, output_image = pre_process(src_image_path)
                #output_image = Image.fromarray(np.uint8(output_image))
                if items>0:
                    cv2.imwrite(tgt_image_path, output_image)
