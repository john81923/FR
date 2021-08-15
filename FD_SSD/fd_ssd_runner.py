import matplotlib.patches as patches
import matplotlib.pyplot as plt
from .ssd_postprocess import postprocess_, nms
from .ssd_preprocess import preprocess_
import glob
import numpy as np
from PIL import Image
import keras
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def display(img, dets=None, landmarks_list=None, save_path=None):
    """
    show detection result.
    """
    if isinstance(img, str):
        img = Image.open(img)

    img = np.array(img)
    img = np.squeeze(img)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

    if dets is not None:
        for i, box in enumerate(dets):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', fill=False)
            ax.add_patch(rect)
            ax.text(box[0], box[1], "%.2f" %
                    box[4], bbox=dict(facecolor='red', alpha=0.5))

    if landmarks_list:
        for i in range(len(landmarks_list)):
            for j in range(0, 10, 2):
                circle = patches.Circle((int(landmarks_list[i][j + 0]), int(
                    landmarks_list[i][j + 1])), max(1, img.shape[0]/200), color='g')
                ax.add_patch(circle)
            ax.text(landmarks_list[i][8], landmarks_list[i][9], "%.2f" %
                    landmarks_list[i][-1], bbox=dict(facecolor='green', alpha=0.5))

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()
    return


class FdSsdRunner:
    def __init__(self, model_path, anchor_path, input_shape=(200, 200), score_thres=0.5, only_max=True, iou_thres=0.35):
        """
        detection function for ssd models.
        model_path: str, path to keras.model
        anchor_path: str, path to anchor file .npy

        image: str or np.ndarray or PIL.Image.Image 
        input_shape: tuple(h, w)
        score_thres: float ~(0,1)
        only_max: bool
        iou_thres: float ~(0,1). Will be ignored when only_max is True
        """
        self.model = keras.models.load_model(model_path)
        self.anchor = np.load(anchor_path, encoding="latin1", allow_pickle=True)
        self.anchor = self.anchor.tolist()

        self.input_shape = input_shape
        self.score_thres = score_thres
        self.only_max = only_max
        self.iou_thres = iou_thres

    def run(self, image):
        """
        do inference on single image
        """

        img_data, scale, w_ori, h_ori = preprocess_(image, self.input_shape)
        outputs = self.model.predict(img_data)
        dets = postprocess_(outputs, self.anchor, scale, self.input_shape,
                            w_ori, h_ori, self.score_thres, self.only_max, self.iou_thres)
        if isinstance(dets, np.ndarray):
            dets = dets.tolist()
        return dets

    def run_batch(self, imagelist):
        """
        do inference on a list of images, which will be packed as batch
        """
        if len(imagelist) == 0:
            return []

        img_data_batch, scale_batch, w_batch, h_batch = [], [], [], []
        for image_path in imagelist:
            img_data, scale, w_ori, h_ori = preprocess_(
                image_path, self.input_shape)
            img_data_batch.append(img_data)
            scale_batch.append(scale)
            w_batch.append(w_ori)
            h_batch.append(h_ori)
        img_data_batch = np.concatenate(img_data_batch, axis=0)

        outputs = self.model.predict(img_data_batch)

        dets_batch = []
        for i in range(len(imagelist)):
            dets_batch.append(postprocess_(
                outputs, self.anchor, scale_batch[i], self.input_shape, w_batch[i], h_batch[i], self.score_thres, self.only_max, self.iou_thres, batch_idx=i))
        return dets_batch

    def run_rect(self, image, rectangle_list):
        """
        do inference on multiple areas of single image
        """
        if len(rectangle_list) == 0:
            return []

        """adjust tall rectangle to make face bigger on resized input"""
        image = Image.fromarray(image)
        w, h = image.size
        rectangle_list = np.array(rectangle_list)
        aspect_ratio_list = rectangle_list[:, 3] / rectangle_list[:, 2]
        high_rect_idx = np.nonzero(aspect_ratio_list > 2.5)
        rectangle_list[high_rect_idx,
                       3] = rectangle_list[high_rect_idx, 3] // 2
        rectangle_list[high_rect_idx, 1] = np.maximum(
            0, rectangle_list[high_rect_idx, 1] - rectangle_list[high_rect_idx, 3] // 5)

        """crop all rectangle area and make them batch"""
        body_batch = []
        for i in range(rectangle_list.shape[0]):
            left, top, width, height, _ = rectangle_list[i, ...]
            body_batch.append(image.crop(
                (left, top, left+width-1, top+height-1)))

        """run batch"""
        dets_batch = self.run_batch(body_batch)

        """map back to orginal image"""
        total_dets = []
        for i, dets in enumerate(dets_batch):
            if len(dets) == 0:
                continue
            dets = np.array(dets)
            dets[:, 0] += rectangle_list[i, 0]
            dets[:, 1] += rectangle_list[i, 1]
            total_dets.extend(list(dets))
        # total_dets = nms(total_dets)
        return total_dets


if __name__ == "__main__":
    #image = "test.jpg"
    image = "/home/jenna/Downloads/image_distance/1080_10_ft.jpg"
    # detector = ssd_detector("./models/ssd_face_cfg2.hdf5", "./models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=True)
    detector = ssd_detector("./models/ssd_person_hw.hdf5",
                            "./models/anchor_person_ssd10.npy",
                            score_thres=0.01, only_max=False)
    #detector = ssd_detector("./models/ssd_face_cfg2_smallbox_kneronPrep.hdf5",
    #                        "./models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)
    dets = detector.run(image)
    display(image, dets)
