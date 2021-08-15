import os
#os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
print(tf.__version__)
import cv2
from PIL import Image
import numpy as np
from skimage import transform as trans
import cv2
from FD_SSD.SSD_infer import ssd_detector
from ONET.onet_infer import onet_detector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#face_detector = ssd_detector("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
#                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)

#landmark_detector = onet_detector("./ONET/models/onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5")

face_detector = FdSsdRunner("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)
landmark_detector = OnetRunner("./ONET/models/rgb2gray_Laplacian_onet_base_onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5",False)


def create_folder (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def display(img, dets=None, landmarks_list=None, save_path='0304_display.png'):
    """
    show detection result.
    """
    if isinstance(img, str):
        img = Image.open(img)

    img = np.array(img)
    img = np.squeeze(img)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

    if dets:
        for i, box in enumerate(dets):
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', fill=False)
            ax.add_patch(rect)
            ax.text(box[0], box[1], "%.2f"%box[4], bbox=dict(facecolor='red', alpha=0.5))

    if landmarks_list:
        for i in range(len(landmarks_list)):
            for j in range(0, 10, 2):
                circle = patches.Circle((int(landmarks_list[i][j + 0]), int(landmarks_list[i][j + 1])), max(1,img.shape[0]/200), color='g')
                ax.add_patch(circle)
            ax.text(landmarks_list[i][8], landmarks_list[i][9], "%.2f"%landmarks_list[i][-1], bbox=dict(facecolor='green', alpha=0.5))




    if save_path:
        fig.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()
    return


def pre_process(image_path):
    dets = face_detector.run(image_path)
    ret = None
    test_path = "/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/00001_0001.png"
    if len(dets)>0:
        #for i in range(len(dets)): # only 1 face
        bbox =(dets[0][0], dets[0][1], dets[0][2], dets[0][3])

        image = Image.open(image_path).convert("RGB")
        img = np.array(image)
        #landmark = landmark_detector.run(img, bbox)
        landmark = landmark_detector.run(image_path, dets[0])
        #print("landmark: ", landmark)

        ''' ========== for testing start ===============================
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')

        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', fill=False)
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], "1.0", bbox=dict(facecolor='red', alpha=0.5))


        for j in range(0, 10, 2):
            circle = patches.Circle((int(landmark[j + 0]), int(landmark[j + 1])), max(1,img.shape[0]/200), color='g')
            ax.add_patch(circle)
        ax.text(landmark[8], landmark[9], "%.2f"%landmark[-2], bbox=dict(facecolor='green', alpha=0.5))

        display(test_path, dets, [landmark])
        #plt.plot(img)
        #plt.show()
        fig.savefig('0304_25.png')
        plt.close()
        exit(0)

        ========== for testing end  =============================== '''
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
        ret_size = ret.shape
        #print("ret_size: ", ret_size)
        #ret = np.expand_dims(ret, axis=-1)
        return len(dets), ret
    else:
        return len(dets), ret



if __name__ == "__main__":
    test_path = "/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/00001_0001.png"
    image = Image.open(test_path).convert("RGB")
    w, h =image.size
    landmarks = landmark_detector.run(image, [0, 0, w, h])
    #landmark = landmark_detector.run(image_path, dets[0])
    #landmarks = detector.run(image, [0, 0, w, h])
    #print(landmarks)
    display(test_path, [landmarks])
