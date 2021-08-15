import numpy as np
from PIL import Image
import kneron_preprocessing

def preprocess_(image, input_shape, keep_ap = True, rectangle=None, pad_center=False, gray_scale=False, **kwargs):
    """
    preprocess includes color-converting, rescale, crop, normalize
    image: str
    input_shape: (h, w)
    keep_ap: bool if resize keep ar
    pad_center: bool if padding center or cornor
    rectangle: (xmin, ymin, w, h)
    :return
    image: numpy.ndarray
    dict: scale, w, h
    """

    if isinstance(image, str):
        try:
            image = kneron_preprocessing.API.load_image(image)
        except:
            try:
                image = kneron_preprocessing.API.load_bin(image, **kwargs)
            except:
                print('input format error')
                assert 0
    else:
        assert isinstance(image, np.ndarray)

    # only do part of the image
    '''
    if rectangle is not None:
        left, top, width, height = rectangle[:4]
        x1, y1, x2, y2 = left, top, left+width, top+height
        image = kneron_preprocessing.API.crop(image, box=(x1, y1, x2, y2))
    '''
    # get image original shape
    h_ori, w_ori = image.shape[:2]
    h, w = input_shape
    scale = [1, 1]
    '''
    if keep_ap:
        if w_ori > h or h_ori > w:
            scale = max(1.0*w_ori / w, 1.0*h_ori / h)
            scale = [scale, scale]
            image = kneron_preprocessing.API.resize(image, size=(w, h), keep_ratio=keep_ap, type='bilinear')
        else:
            scale = [1, 1]
    else:
        if w_ori > h or h_ori > w:
            scale = [1.0*w_ori / w, 1.0*h_ori / h]
            image = kneron_preprocessing.API.resize(image, size=(w, h), keep_ratio=keep_ap, type='bilinear')
        else:
            scale = [1, 1]

    if pad_center:
        image = kneron_preprocessing.API.pad_center(image, size=(w, h), pad_val=0)
    else:
        image = kneron_preprocessing.API.pad_corner(image, size=(w, h), pad_val=0)
    '''
    img_data = np.array(image).reshape((h, w, 3))
    if gray_scale:
        img_data = kneron_preprocessing.API.convert(image=img_data, out_fmt='NIR')
        img_data = img_data.reshape((h, w, 1))
    img_data = img_data[None]

    img_data = kneron_preprocessing.API.norm(img_data)
    
    return img_data, {'scale': scale, 'w_ori': w_ori, 'h_ori': h_ori}
