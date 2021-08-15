from base.preprocess import preprocess_
from .fr_postprocess import postprocess_
import base.inference

class FrRunner:
    def __init__(self, model_path, input_shape, gray_scale=False,  **kwargs):

        # --------------- STEP 1 load model----------------
        from kneron_utils.model import load_model
        self.model, self._inference_type = load_model(model_path, **kwargs)

        # --------------- STEP 2 get all parameter as dict-----------------
        self.init_config = locals()
        self.init_config.update(kwargs)
        #print(self.init_config)
        #exit(0)

    def run(self, image, aligned_images=None, **kwargs):
        if aligned_images is None:
            images = [image]
        else:
            images = aligned_images
        emb_list = []


        for image in images:

            #  -----------------preprocess------------------
            pre_config = {
                'image': image,
                "keep_ap": True,
                "pad_center": True
            }
            pre_config.update(self.init_config)
            pre_config.update(kwargs)
            img_data, pre_info = preprocess_(**pre_config)

            # -----------------do inference------------------
            infer_config = {
                'pre_results': img_data,
                'model': self.model,
                'type': self._inference_type
            }
            infer_config.update(self.init_config)
            outputs = base.inference.inference_(**infer_config)

            # -----------------post process------------------
            post_config = {
                'outputs': outputs
            }
            post_config.update(pre_info)
            emb = postprocess_(**post_config)

            emb_list.append(emb)
        return emb_list
