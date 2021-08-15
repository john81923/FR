import sys
import torch
from torch import nn
import os

def _target_model_instantiate(model_def_path, module_name, **kwargs):
    import_dir = os.path.dirname(model_def_path)
    script_name = os.path.basename(model_def_path).replace('.py','')
    sys.path.append(import_dir)
    try:
        import_cmd_string = "from {} import {} as target_model".format(script_name, module_name)
        print(import_cmd_string)
    except Exception:
        print(Exception)
        print('missing model define file path')
        assert 0
    exec(import_cmd_string, globals())
    return target_model(**kwargs)

def load_torch_model(model_path, lib_path=None, model_def_path=None, module_name=None, **kwargs):
    """
    :param model_path:
        string, path of the pytorch model file.
    :param lib_path:
        a folder that include all dependencies scripts of building block classes. (optional)
    :param model_def_path:
        a script that instantiates `target_model` based on the imported building blocks.
        if `lib_path` is not defined, one should difine all building block classes in this script
        before instantiating `target_model`
    :param module_name:
        a function return model, it takes parameter from kwargs which should in init_para.json
    """

    # TODO: double check here
    # file_type = magic.from_file(model_path)
    if model_def_path is None and 'zip' in model_path:
        loaded_obj = torch.jit.load(model_path) # JIT IR model, for future usage
        model = loaded_obj
    else:
        # (1) include lib path
        if lib_path is not None:
            sys.path.append(lib_path)

        # (3) load model weight file
        loaded_obj = torch.load(model_path)
        if not isinstance(loaded_obj, nn.Module):
            # note: there should be a import module session before targe_model instantiate
            assert model_def_path is not None and module_name is not None
            target_model = _target_model_instantiate(model_def_path, module_name, **kwargs)
            target_model.load_state_dict(loaded_obj)
            model = target_model
        else:
            assert 0
            model = loaded_obj
        return model

def _get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device