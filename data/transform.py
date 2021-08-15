import torchvision

indoor_transform = [torchvision.transforms.ColorJitter(brightness=0.2), torchvision.transforms.ColorJitter(brightness=(0.4, 0.6))]
outdoor_transform = [torchvision.transforms.ColorJitter(brightness=0.2), torchvision.transforms.ColorJitter(brightness=(1.3, 1.5))]
general_transform = [torchvision.transforms.ColorJitter(brightness=0.2), torchvision.transforms.ColorJitter(brightness=(0.3, 0.6)), torchvision.transforms.ColorJitter(brightness=(1.3, 1.6))]
affine_transform = [ torchvision.transforms.RandomAffine(5, translate=None, scale=None, shear=10, resample=False, fillcolor=0)]
#indoor_transform = [ torchvision.transforms.ColorJitter(brightness=(0.3, 0.6))]
#outdoor_transform = [ torchvision.transforms.ColorJitter(brightness=(1.3, 1.6))]


'''
'kneron': torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    #torchvision.transforms.ColorJitter(brightness=(0.8,1.2))
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
    #torchvision.transforms.RandomGrayscale(0.4),
]),
'''

image_transforms = {
    'kneron-gray': torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01), #origin
        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.7, hue=0.015), #origin
        #torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03),
        #torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.01),
        #torchvision.transforms.RandomChoice(general_transform),
        #torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3),
        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4),
        #torchvision.transforms.RandomGrayscale(0.3),
        #torchvision.transforms.RandomGrayscale(0.1),
        #torchvision.transforms.Grayscale(3)
    ]),
    'kneron-gray-indoor': torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        #torchvision.transforms.ColorJitter(contrast=0.2),
        #torchvision.transforms.RandomChoice(indoor_transform),
        #torchvision.transforms.RandomChoice(affine_transform),
        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4),
        #torchvision.transforms.RandomGrayscale(0.2),
        #torchvision.transforms.Grayscale(3)
    ]),

    'kneron-gray-outdoor': torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),

        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        #torchvision.transforms.ColorJitter(contrast=0.2),
        #torchvision.transforms.RandomChoice(outdoor_transform),
        #torchvision.transforms.RandomChoice(affine_transform),
        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4),
        #torchvision.transforms.RandomGrayscale(0.2),
        #torchvision.transforms.Grayscale(3)
    ]),
}


data_transforms = {
    'kneron': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[1.0, 1.0, 1.0], mean=[0.5, 0.5, 0.5])
    ]),
    'tf': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
    ]),
    'lfw': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[128., 128., 128.], mean=[128., 128., 128.])
    ]),
    'darker': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[1., 1., 1.], mean=[0., 0., 0.])
    ]),
    'CV-kneron': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(std=[256., 256., 256.], mean=[128, 128, 128]),
        #torchvision.transforms.RandomErasing(),
    ]),
}
