import os
import base.inference
def load_model(model_path, **kwargs):
    if "USER_CONFIG" not in kwargs:
        type = os.path.splitext(model_path.lower())[1][1:]
        if type == 'onnx':
            import onnxruntime
            sess = onnxruntime.InferenceSession(model_path)
            input_name = sess.get_inputs()[0].name
            model = [sess, input_name]
            _inference_type = base.inference.ONNX
        elif type in ['h5', "hdf5"]:
            import keras
            model = keras.models.load_model(model_path)
            _inference_type = base.inference.KERAS
        elif type in ['pts', 'pt', 'tar', 'pth']:
            from kneron_utils.torch_utils import load_torch_model, _get_device
            assert 'model_def_path' in kwargs and 'module_name' in kwargs
            model = load_torch_model(model_path, **kwargs)
            device = _get_device()
            model.eval()
            model = model.to(device)
            _inference_type = base.inference.TORCH
        else:
            raise TypeError("unknown model type")
    else:
        model = None
        _inference_type = base.inference.EMULATOR

    return model, _inference_type