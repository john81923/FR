ONNX = 0
TORCH = 1
KERAS = 2
MXNET = 3
EMULATOR = 4

import numpy as np


def inference_(pre_results, model, type, **kwargs):
    # TODO: align multiple output situation
    if type == 0:
        # onnx
        sess, input_name = model
        img_data = np.transpose(pre_results, [0, 3, 1, 2])
        inf_results = sess.run(None, {input_name: img_data.astype(np.float32)})
        if isinstance(inf_results, list):
            inf_results = [item.transpose([0, 2, 3, 1]) if np.ndim(item)==4 else item for item in inf_results]
        elif np.ndim(inf_results) == 4:
            inf_results = inf_results.transpose([0, 2, 3, 1])

    elif type == 1:
        import torch
        from kneron_utils.torch_utils import _get_device

        # convert numpy to tensor

        device = _get_device()
        img_data = torch.from_numpy(np.transpose(pre_results, [0, 3, 1, 2]))
        tensor = img_data.float()
        #print("tensor:", tensor)
        img_tensor = tensor.to(device)
        #model = model.eval()
        # do model inference
        with torch.no_grad():
            inf_results = model(img_tensor)
            if inf_results is tuple or inf_results is list:
                inf_results = [item.cpu().numpy() for item in inf_results]
            else:
                inf_results = inf_results.cpu().numpy()

            #print("inf_results: ", inf_results)
    elif type == 2:
        # keras
        inf_results = model.predict(pre_results)

    elif type == 4:
        import python_flow.emulator.emulator as emu
        assert "USER_CONFIG" in kwargs
        config = kwargs.get("USER_CONFIG")
        assert isinstance(config, dict)
        emu_mode = "bypass"
        if "emu" in config and "emu_mode" in config["emu"]:
            emu_mode = config["emu"]["emu_mode"]
        emu_dict = {
            "csim": emu.emulator_csim,
            "float": emu.emulator_float,
            "fixed": emu.emulator_fixed,
            "bypass": lambda x, y: print("Bypassing inference...")
        }
        assert emu_mode in emu_dict
        inf_results = emu_dict.get(emu_mode)(config, pre_results)
        if inf_results is None:
            inf_results = []
    else:
        raise Exception("missing inference implement")

    return inf_results
