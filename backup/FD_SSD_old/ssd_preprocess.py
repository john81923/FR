import numpy as np
from PIL import Image


def preprocess_(image, input_shape):
    """
    preprocess includes color-converting, rescale, crop, normalize
    image: PIL.Image.Image
    input_shape: (h, w)
    """
    if isinstance(image, str):
        filename = image
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    assert (isinstance(image, Image.Image)) 

    w_ori, h_ori = image.size
    image = image.convert("RGB")

    if image.size[0] > input_shape[0] or image.size[1] > input_shape[1]:
        scale = max(1.0*w_ori / input_shape[0], 1.0*h_ori / input_shape[1])
        image.thumbnail(input_shape)
    else:
        scale = 1

    image = image.crop((0, 0, input_shape[0], input_shape[1]))
    img_data = np.array(image).reshape((1, input_shape[1], input_shape[0], 3))    
    img_data = img_data / 256. - 0.5

    return img_data, scale, w_ori, h_ori