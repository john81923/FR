## this is only for temp exp for model concat, please not use it for other models
import math
import sys
import numpy as np
import cv2
from skimage import transform as trans
from PIL import Image, ImageOps, ImageFilter
import torch, torchvision
from torch import nn
import sys,os
import magic


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def import_pytorch_model_def_from_file(model_def_path, import_cmd_string=None):
    if '.py' in model_def_path:
        import_dir = os.path.dirname(model_def_path)
        script_name = os.path.basename(model_def_path).replace('.py','')
        sys.path.append(import_dir)
        import_cmd_string = "from {} import *".format(script_name)
        print(import_cmd_string)
        exec(import_cmd_string, globals())
    else:
        sys.path.append(model_def_path)
        print(import_cmd_string)
        exec(import_cmd_string, globals())

    return


class FR_torch_test:
    def __init__(self, model_path, model_def_path=None, import_cmd_string=None,  **kwargs):
        if model_def_path:
            import_pytorch_model_def_from_file(model_def_path, import_cmd_string)
        self.net = torch.load(model_path)
        self.net.eval()
        print(self.net)
        #exit(0)

class FR_torch:
    def __init__(self, model_path, model_def_path=None, import_cmd_string=None, model_def_string=None, **kwargs):
        self.gray_scale = kwargs.get('gray_scale', False)
        self.im_size = kwargs.get('im_size', (112, 112, 3))
        self.img_warp = np.zeros((1,) + self.im_size)
        self.device = get_device()
        if model_def_path:
            import_pytorch_model_def_from_file(model_def_path, import_cmd_string)

        self.net = torch.load(model_path)
        self.net.eval()
        self.net = self.net.to(self.device)
        '''
        #file_type = magic.from_file(model_path)

        if 'Zip' in file_type:
            loaded_obj = torch.jit.load(model_path) # JIT IR model
        else:
            loaded_obj = torch.load(model_path)

        if isinstance(loaded_obj, dict):
            if loaded_obj.get('head'):
                ## the following codes are just a workaround of craig's NIR head case:
                rgb_head = torch.load('../model/craig_kface_40m/rgbhead.tar')
                nir_head = loaded_obj['head']
                backbone = loaded_obj.get('backbone')
                if backbone is None:
                    backbone = torch.load("../model/craig_kface_40m/0306_v1.tar").get('backbone')
                self.backbone = backbone
                self.rgb_head = rgb_head
                self.nir_head = nir_head

                self.backbone.eval()
                self.backbone = self.backbone.to(self.device)
                self.rgb_head.eval()
                self.rgb_head = self.rgb_head.to(self.device)
                self.nir_head.eval()
                self.nir_head = self.nir_head.to(self.device)
        '''


    def run(self, img, landmark = None, aligned=False, input_range='[-1.0, 1.0]'):
        """
         :param img: Image
         :param aligned: if image is aligned
         :return: feature
         """

        # (1)transformation range
        if input_range == '[-1.0, 1.0]':
            rgb_mean = [0.5, 0.5, 0.5]
            rgb_std = [0.5, 0.5, 0.5]
        elif input_range == '[-0.5, 0.5]':
            rgb_mean = [0.5, 0.5, 0.5]
            rgb_std = [1, 1, 1]

        self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(std=rgb_std, mean=rgb_mean)
            ])

        #(2) load image
        if isinstance(img, str):
            img = Image.open(img)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        #(3) run alignment
        if aligned:
            img = img.resize(self.im_size[:2])
            if self.gray_scale:
                img = img.convert('L')
            elif img.mode == 'L':
                img = img.convert('RGB')
            if isinstance(img, Image.Image):
                img = np.array(img)[..., :3]
            img_test = np.expand_dims(self.normalize(img), axis=0)

        else:
            if landmark is None or len(landmark) == 0:
                return []

            # [Jeff] : check landmark is nparray or not
            if isinstance(landmark, list):
                landmark = np.asarray(landmark[:10])
                landmark = landmark.reshape([5, 2])

            _, landmark = self.align(img, landmark)
            if landmark is None:
                # img = ImageOps.expand(img, (10, 10, 10, 10,))
                return []
            img_test = self.img_warp
            # print(img_test.shape)
            # print('img_test :' ,img_test[0])

        img_test = img_test[0]
        img_test_ = img_test.astype(np.uint8)
        face = self.transform(img_test_)
        input_imgs_tensor = torch.zeros([1, 3, 112, 112], dtype=torch.float)
        input_imgs_tensor[0] = face
        input_imgs_tensor = input_imgs_tensor.to(get_device()) ###

        #(4) run model
        '''
        with torch.no_grad():
            output = self.backbone(input_imgs_tensor)
            rgb_output = self.rgb_head(output)
            emb_rgb = rgb_output[0].cpu().numpy()
            normed_emb_rgb = emb_rgb / np.linalg.norm(emb_rgb)

            nir_output = self.nir_head(output)
            emb_nir = nir_output[0].cpu().numpy()
            normed_emb_nir = emb_nir / np.linalg.norm(emb_nir)
        if len(normed_emb_rgb) != 0 and len(normed_emb_nir) != 0:
            return normed_emb_rgb.tolist(), normed_emb_nir.tolist()
        else:
            return [], []
        '''
        #(4) run model (single model)
        with torch.no_grad():
            rgb_output , nir_output= self.net(input_imgs_tensor)
            emb_rgb = rgb_output[0].cpu().numpy()
            normed_emb_rgb = emb_rgb / np.linalg.norm(emb_rgb)
            emb_nir = nir_output[0].cpu().numpy()
            normed_emb_nir = emb_nir / np.linalg.norm(emb_nir)
        if len(normed_emb_rgb) != 0 and len(normed_emb_nir) != 0:
            return normed_emb_rgb.tolist(), normed_emb_nir.tolist()
        else:
            return [], []


    def align(self, img, landmark):
        if isinstance(img, str):
            img = Image.open(img)
        if self.gray_scale:
            img = img.convert('L')
        elif img.mode == 'L':
            img = img.convert('RGB')
        if isinstance(img, Image.Image):
            img = np.array(img)[..., :3]
        ## load landmark
        #### add here ###

        if landmark is None:
            return None, None, None

        self.img_warp[0, :] = np.array(self._preprocess(img, landmark, image_size='112,112'))

        # return self.img_warp[0] / 255., landmark ##<----- [Lu] removed cuz there is ToTensor transformation
        return self.img_warp[0], landmark

    @staticmethod
    def normalize(x):
        # return x / 255. - 0.5      ##<-------- [Lu] removed cuz there is ToTensor transformation
        return x


    @staticmethod
    def _preprocess(img, landmark=None, **kwargs):
        if isinstance(img, str):
            img = Image.open(img)

        img = np.array(img)

        M = None
        image_size = []
        str_image_size = kwargs.get('image_size', '')
        if len(str_image_size) > 0:
            image_size = [int(x) for x in str_image_size.split(',')]
            if len(image_size) == 1:
                image_size = [image_size[0], image_size[0]]
            assert len(image_size) == 2
            assert image_size[0] == 112
            assert image_size[0] == 112 or image_size[1] == 96
        if landmark is not None:
            assert len(image_size) == 2
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
            if image_size[1] == 112:
                src[:, 0] += 8.0

            dst = np.squeeze(landmark.astype(np.float32))
            tform = trans.SimilarityTransform()

            tform.estimate(dst, src)
            M = tform.params[0:2, :]
            # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        if M is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]

            margin = kwargs.get('margin', 44)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
            bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
            ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
            if len(image_size) > 0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret
        else:  # do align using landmark
            assert len(image_size) == 2
            warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
            if len(warped.shape) == 2:
                warped = np.expand_dims(warped, axis=-1)
            return warped

if __name__ == "__main__":
    #from  model_import.Kneron_SD_Models.FR_pth.FR_pth_infer import FR_torch
    #from PIL import Image, ImageOps, ImageFilter
    def distance(e1, e2):
        diff = np.subtract(np.array(e1), np.array(e2))
        d = np.sum(np.square(diff),0)
        return d

    def fusion_with_alpha(pair_dists_long_to_thr, pair_dists_craig_to_thr, alpha):
        res = [alpha*x+(1-alpha)*y for x,y in zip(pair_dists_long_to_thr, pair_dists_craig_to_thr)]
        return res

    model_def_path=  "../model/craig_kface_40m/model"
    test_model_def_path=  "../"
    import_cmd_string = "import kfr"
    craig_model_path = "../model/craig_kface_40m/0314_v23_e99.tar"
    full_model_path = "/mnt/storage1/craig/kneron_fr/full_net/merged_v23.tar"
    # hyper-parameters (based on experiment 2020.03.23
    treshold_long = 1.0807082000000001
    treshold_craig = 1.1704632
    alpha = 0.13

    # model init
    FR_torch(model_path=full_model_path, model_def_path=model_def_path, import_cmd_string=import_cmd_string)
    #FR_torch_test(model_path=full_model_path, model_def_path=test_model_def_path, import_cmd_string=import_cmd_string)

    # run inference and decisions
    img1 = "../datasets/lfw_original/Adam_Scott/Adam_Scott_0001.jpg"
    emb_rgb1, emb_nir1  = fusion_fr_model.run(img1, landmark = None, aligned=True, input_range='[-0.5, 0.5]')
    img2 = "../datasets/lfw_original/Adam_Scott/Adam_Scott_0002.jpg"
    emb_rgb2, emb_nir2  = fusion_fr_model.run(img2, landmark = None, aligned=True, input_range='[-0.5, 0.5]')
    decision = fusion_with_alpha(dists_long_to_thr, dists_craig_to_thr, alpha)
    decision = [True if d <0 else False for d in decision]
