import numpy as np


def onet_postprocess_(output, scale_w, scale_h, input_shape, rectangle):
    """
    output: list(np.array, np.array)
    scale_w: float
    scale_h: float
    input_shape: (h,w)
    rectangle: [left, top, w, h, score]
    """
    score = 0
    blur_score = 0
    glass_cls = 0
    if len(rectangle) == 0:
        rectangle = [0,0,input_shape[1],input_shape[0]]
    if len(output) == 1:
        landmarks = np.squeeze(output[0])
    elif len(output) == 2:
        landmarks = np.squeeze(output[0])
        score = np.squeeze(output[1])[1]
        # score = np.squeeze(output[1])
    elif len(output) == 3:
        # landmarks = np.squeeze(output[0])
        # score = np.squeeze(output[1])[1]
        # blur_feat = np.squeeze(output[2])
        # blur_score = np.var(blur_feat)*256*256
        landmarks = np.squeeze(output[0])
        score = np.squeeze(output[1])[1]
        glass_cls = np.argmax(output[2])

    else:
        assert 0
    if landmarks.min() < 0:
        landmarks[1::2] = landmarks[1::2]+input_shape[0]//2
        landmarks[::2] = landmarks[::2]+input_shape[1]//2
    if np.max(landmarks) < 1 and np.min(landmarks)>0:
        landmarks[::2] = landmarks[::2] * input_shape[1]
        landmarks[1::2] = landmarks[1::2] * input_shape[0]

    landmarks_mapped = []
    left = rectangle[0] + rectangle[2]//2 - max(rectangle[2:4])//2
    top = rectangle[1] + rectangle[3]//2 - max(rectangle[2:4])//2
    for i in range(0, len(landmarks), 2):
        # landmarks_mapped.append(int(np.round(landmarks[i + 0] * scale_w + rectangle[0])))
        # landmarks_mapped.append(int(np.round(landmarks[i + 1] * scale_h + rectangle[1])))
        landmarks_mapped.append(int(np.round(landmarks[i + 0] * scale_w + left)))
        landmarks_mapped.append(int(np.round(landmarks[i + 1] * scale_h + top)))
    landmarks_mapped.append(float(score))
    # landmarks_mapped.append(float(blur_score))
    landmarks_mapped.append(float(glass_cls))
    return landmarks_mapped
