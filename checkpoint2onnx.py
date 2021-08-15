import os
import sys
from datetime import datetime
import numpy as np
import json
import torch
sys.path.append('../')

src_dir = '/mnt/sdd/johnnysun/model/FR'
ckpt_folder = '0722_resnet_mi_v35_2'  #
ckpt_name = 'checkpoint-r50-I112-E256-e0065-av0.9998_1.0000'
#'''

dst_dir = '/mnt/models/FR_models/FR_onnx_0609reproduce'
dst_folder = f'{dst_dir}/{ckpt_folder}'
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

checkpoint_name = f'{src_dir}/{ckpt_folder}/'+ ckpt_name +'.tar'
onnx_name = f'{src_dir}/{ckpt_folder}/'+ ckpt_name +'.onnx'
target_onnx = f'{dst_dir}/{ckpt_folder}/'+ ckpt_name +'.onnx'

checkpoint = torch.load( checkpoint_name )
model = checkpoint['model']
model.eval()

img = torch.zeros((1, 3, 112, 112), device='cuda')
torch.onnx.export( model, img, onnx_name, verbose=False, opset_version=11,  keep_initializers_as_inputs=True,
                        input_names=['images'], output_names=['embedding'])

#print('torch export onnx: ', onnx_name )
# py2onnx
os.system( f'python ~/johnnysun/ONNX_Convertor/optimizer_scripts/pytorch2onnx.py {onnx_name} {onnx_name}.py2ox.onnx')
#sys.exit(1)
# onnx2onnx
os.system( f'python ~/johnnysun/ONNX_Convertor/optimizer_scripts/onnx2onnx.py -o {target_onnx}.opt.onnx {onnx_name}.py2ox.onnx ')


os.system( f'chmod 777 {target_onnx}.opt.onnx' )
os.system( f'rm {onnx_name}.py2ox.onnx {onnx_name}')
print( f'Generate ONNX: {target_onnx}.opt.onnx')

# gen json_file
init_params = {
  "model_path": f"{target_onnx}.opt.onnx",
  "input_shape": [112, 112]
}
# dump json file
init_parmas_path = f'{dst_dir}/{ckpt_folder}/{ckpt_folder}_init_params.json'
json_object = json.dumps(init_params, indent=4)
print('Generate init params json file')
print(json_object)
with open( init_parmas_path , "w") as outfile:
    outfile.write(json_object)
    os.system( f'chmod 777 {init_parmas_path}' )

# load DAG template
DAG_templte_path = '/mnt/sdd/johnnysun/kneron_hw_models/applications/ssd+onet+applications/ssd+onet+fr_021921_DAG.json' # new landmark
with open( DAG_templte_path, 'r') as DAG_file:
    DAG_object = json.load(DAG_file)
    DAG_object['model_list'][3]['model_init_params_file'] = init_parmas_path
# dump new DAG
print('Generate DAG file:', f'{src_dir}/{ckpt_folder}/{ckpt_folder}_DAG.json')
DAG_save_path = f'{src_dir}/{ckpt_folder}/{ckpt_folder}_DAG.json'
with open( DAG_save_path , 'w') as DAG_save:
    json.dump(DAG_object, DAG_save, indent=4)
    os.system( f'chmod 777 {DAG_save_path}' )
