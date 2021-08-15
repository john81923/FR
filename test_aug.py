import os
from PIL import Image
import cv2
import torchvision
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


ia.seed(1)
def create_folder (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
#indoor_transform = [torchvision.transforms.ColorJitter(brightness=0.2), torchvision.transforms.ColorJitter(brightness=(0.4, 0.6))]
#outdoor_transform = [torchvision.transforms.ColorJitter(brightness=0.2), torchvision.transforms.ColorJitter(brightness=(1.4, 1.6))]
indoor_transform = [ torchvision.transforms.ColorJitter(brightness=(0.3, 0.4))]
outdoor_transform = [ torchvision.transforms.ColorJitter(brightness=(1.3, 1.5))]

affine_transform = [ torchvision.transforms.RandomAffine(5, translate=None, scale=None, shear=10, resample=False, fillcolor=0)]
image_transforms = {

    'kneron-gray-indoor': torchvision.transforms.Compose([
        #torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(contrast=0.2),
        torchvision.transforms.RandomChoice(indoor_transform),
        torchvision.transforms.RandomChoice(affine_transform),

        #torchvision.transforms.RandomErasing(),
    ]),

    'kneron-gray-outdoor': torchvision.transforms.Compose([
        #torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(contrast=0.2),
        torchvision.transforms.RandomChoice(outdoor_transform),
        torchvision.transforms.RandomChoice(affine_transform),
        #torchvision.transforms.RandomErasing(),
    ]),
}
src_folder = "/mnt/storage1/craig/test/input"
#tgt_folder = "/mnt/storage1/craig/test/out_small_both"
tgt_folder = "/mnt/storage1/craig/test/out_test4"

seq = iaa.Sequential([

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    #iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.GaussianBlur(sigma=(0.5, 1)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel
    #. This can change the color (not only brightness) of the
    # pixels.
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=False),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.05*255, 0.1*255), per_channel=False),
], random_order=True) # apply augmenters in random order

create_folder(tgt_folder)
image_set = os.listdir(src_folder)
#print(image_set)
for image in image_set:
    src_image_path  = os.path.join(src_folder,image)
    tgt_image_path  = os.path.join(tgt_folder,image)

    img = Image.open(src_image_path)
    #print(img.size)
    #exit(0)
    if 'indoor' in src_image_path:
        output_image = image_transforms['kneron-gray-indoor'](img)
    elif 'outdoor' in src_image_path:
        output_image = image_transforms['kneron-gray-outdoor'](img)
    output_image = np.asarray(output_image, dtype='f')



    cv2.imwrite(tgt_image_path, output_image)
