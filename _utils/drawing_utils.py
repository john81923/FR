import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import cv2

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
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', fill=False)
            ax.add_patch(rect)
            if len(box) ==5:
                ax.text(box[0], box[1], "%.2f" % box[4], bbox=dict(facecolor='red', alpha=0.5))
            if len(box) == 6:
                ax.text(box[0], box[1], "%d:%.2f" % (box[5],box[4]), bbox=dict(facecolor='red', alpha=0.5))

    if landmarks_list is not None:
        for i in range(len(landmarks_list)):
            for j in range(0, len(landmarks_list[i])-2, 2):
                circle = patches.Circle((int(landmarks_list[i][j + 0]), int(landmarks_list[i][j + 1])),
                                        max(1, img.shape[0] / 200), color='g')
                ax.add_patch(circle)
            ax.text(landmarks_list[i][8], landmarks_list[i][9], "%.2f" % landmarks_list[i][-1][0],
                    bbox=dict(facecolor='green', alpha=0.5))

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()
    return


def draw_frame(frame, dets, landmark=None):
    if dets is not None:
        for i, box in enumerate(dets):
            box = list(box)
            left, top, right, bottom = map(int, box[:4])
            color = (0, 255, 0)
            if len(box) <= 5:
                color = (0, 255, 0)
            elif len(box) == 6:
                if box[-1] == 1:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)
            right += left
            bottom += top
            cv2.line(frame, (left, top), (right, top), color, 1)
            cv2.line(frame, (left, top), (left, bottom), color, 1)
            cv2.line(frame, (right, top), (right, bottom), color, 1)
            cv2.line(frame, (left, bottom), (right, bottom), color, 1)
            cv2.putText(frame, str(box[-1]) +": " + str(box[-2]), (left, top), 0,
                                0.5, (0, 255, 0), 1)
    if landmark is not None and len(landmark)>0:
        landmark = list(landmark.reshape((-1,)))
        for i in range(len(landmark)/2):
            cv2.circle(frame, tuple(landmark[2 * i:2 * i + 2]), 1, (255, 0, 0), 3)
    return frame



def get_label_json(landmarks, image_path, w, h):
    labelme_json = {u'fillColor': [255, 0, 0, 128],
                    u'flags': {},
                    u'imageData': None,
                    u'imageHeight': 86,
                    u'imagePath': u's0001_00001_0_0_0_0_0_01.png',
                    u'imageWidth': 86,
                    u'lineColor': [0, 255, 0, 128],
                    u'shapes': [],
                    u'version': u'3.10.0'}

    results = []
    for i in range(0, len(landmarks), 2):
        result = {u'fill_color': None,
                  u'label': u'3',
                  u'line_color': None,
                  u'points': [],
                  u'shape_type': u'point'}
        result[u'label'] = str(i // 2)
        result[u'points'] = [landmarks[i:i + 2]]
        results.append(result)
    labelme_json[u'shapes'] = results
    labelme_json[u'imagePath'] = os.path.split(image_path)[1]
    labelme_json[u'imageHeight'] = h
    labelme_json[u'imageWidth'] = w
    return  labelme_json