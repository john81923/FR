import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import numpy as np
from PIL import Image
from .onet_preprocess import onet_preprocess_
from .onet_postprocess import onet_postprocess_
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class onet_detector:
    def __init__(self, model_path, gray_scale=True):
        self.model = keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape[1:3]
        self.gray_scale = gray_scale
        # self.score_thres = score_thres
        return

    def run(self, image, rectangle):
        """
        image: str or np.ndarray or ImageObject
        rectangle: list, [x, y, w, h, score]

        -return: landmarks, list of 11 flaots or []
        """
        if len(rectangle) == 0: return []

        img_data, scale_w, scale_h = onet_preprocess_(image, rectangle, self.input_shape, self.gray_scale)
        output = self.model.predict(img_data)
        landmarks = onet_postprocess_(output, scale_w, scale_h, self.input_shape, rectangle)
        return landmarks

    def run_batch(self, image, rectangle_list):
        if len(rectangle_list) == 0: return []

        img_data_batch, scale_batch = [], []
        for rectangle in rectangle_list:
            img_data, scale_w, scale_h = onet_preprocess_(image, rectangle, self.input_shape, self.gray_scale)
            img_data_batch.append(img_data)
            scale_batch.append((scale_w, scale_h))

        img_data_batch = np.concatenate(img_data_batch, axis=0)
        
        outputs = self.model.predict(img_data_batch)
        if not isinstance(outputs,list):
            outputs = [outputs]
        landmarks_list = []
        for i in range(len(rectangle_list)):
            landmarks = onet_postprocess_([item[i] for item in outputs], scale_batch[i][0], scale_batch[i][1], self.input_shape, rectangle_list[i])
            landmarks_list.append(landmarks)
        return landmarks_list


if __name__ == "__main__":
    import glob
    from PIL import Image
    def display(img, dets=None, landmarks_list=None, save_path=None):
        """
        show detection result.
        """
        if isinstance(img, str):
            img = Image.open(img)
        
        img = np.array(img)
        img = np.squeeze(img)
        fig = plt.figure(figsize=(5,5))
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
            if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
            fig.savefig(save_path)
            plt.close()
        else:
            plt.show()
            plt.close()
        return

    # detector = onet_detector("./models/weights-0.417197_wclass.hdf5")
    detector = onet_detector("./models/weights-0.417197_wclass_bn3.hdf5")

    for image_file in glob.glob("./Fail/*"):
        image = Image.open(image_file).convert("RGB")
        w, h =image.size
        landmarks = detector.run(image, [0, 0, w, h])
        print(landmarks[-1])
        # display("./lmPreOutput.jpg", landmarks=landmarks)
        display(image, landmarks_list=[landmarks], save_path="./res_bn3/"+os.path.basename(image_file))


# """
# Function: function to call Onet (landmark detection)
# - img: np array (grayscale)
# - Onetbd: keras model (input channel = 1)
# - rectangle: (left, top, width, height, score)
# """
# def run_onet_bd(img, rectangle, Onet, size = (48,48)):
#     """
#     Function: using Onet model to do inference on given image

#     Input:
#     - img: Image.Image (gray)
#     - rectangle: rough bounding boxs of faces where _bbox is tuple (left, top, right, bottom)
#     - Onet: Onet

#     Ouput:
#     a list of landmarks (5 points)
#     (leye x, leye y,  reye x, reye y, nose x, nose y, lmouse x, lmouse y, rmouse x, rmouse y)
#     """

#     if isinstance(img, np.ndarray):
#         img = Image.fromarray(img)

#     if img.mode == 'RGB':
#         img = img.convert('L')

#     # crop and scale the rectangle area of input image
#     w, h = rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]
#     rectangle[2] = rectangle[0] + max(w, h)
#     rectangle[3] = rectangle[1] + max(w, h)
#     scale_img = img.crop(rectangle).resize(size, Image.NEAREST)
#     x = np.array(scale_img)
#     x = x / 255. - 0.5
#     x = np.expand_dims(x, axis=0)
#     x = np.expand_dims(x, axis=3)

#     # conducting inference
#     # (x, y)*5
#     y = Onet.predict(x)
#     landmarks = y[0]
#     if np.min(landmarks) < 0:
#         landmarks += size[0]//2

#     scale_w, scale_h = 1.0*(rectangle[2] - rectangle[0]) / size[0], 1.0*(rectangle[3] - rectangle[1]) / size[1]
#     landmarks_mapped = []
#     for i in range(0, 10, 2):
#         landmarks_mapped.append(int(np.round(landmarks[i + 0] * scale_w + rectangle[0])))
#         landmarks_mapped.append(int(np.round(landmarks[i + 1] * scale_h + rectangle[1])))

#     # return np.asarray(landmarks_mapped)

#     landmarks_return = tuple(landmarks_mapped)

#     return landmarks_return
