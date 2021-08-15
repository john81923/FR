from PIL import Image
import numpy as np
from keras.preprocessing.image import array_to_img

def onet_preprocess_(image, rectangle, input_shape, gray_scale=True):
    if isinstance(image, str):
        filename = image
        image = Image.open(filename)
    if isinstance(image, np.ndarray):
        image = array_to_img(image)
    assert (isinstance(image, Image.Image))

    if gray_scale:
        image = image.convert("L")
    else:
        image = image.convert('RGB')

    # cropping with same width and height
    if len(rectangle)==0:
        left, top, width, height = 0,0, image.size[0], image.size[1]
    else:
        left, top, width, height = rectangle[:4]
    size = max(width, height)
    scale_img = image.crop([left, top, left+size, top+size]).resize(input_shape, Image.NEAREST)
    if gray_scale:
        x = np.array(scale_img).reshape((1, input_shape[0], input_shape[1], 1))
    else:
        x = np.array(scale_img).reshape((1, input_shape[0], input_shape[1], 3))
    x = x / 256. - 0.5

    return x, size / input_shape[0], size / input_shape[1]
