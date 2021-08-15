import os
import sys
from datetime import datetime
import numpy as np

import torch
from torch import nn

import torchvision
from ..conf.config import DEVICE

from ..model.resnet import  Head, ResNet50Backbone,  Merged_Resnet, resnet50, resnet50_large
from ..model.arc_margin import ArcMarginModel, update_arc_margin

def finetune_network(args):

    checkpoint = torch.load(args.checkpoint)
    trained_model = checkpoint['model']

    #trained_model = torch.load(args.checkpoint)
    #print(trained_model)
    #exit(0)
    if args.regnet:
        model = Merged_RGB(emb_size=args.emb_size, load_pretrain = False)
        #print(model.state_dict())
        #print(model)
        #print(trained_model)
        model.load_state_dict(trained_model.state_dict())
        #torch.save(model.state_dict(), '/mnt/storage1/craig/kneron_fr/for_test/regnet_v2_state_dict.tar')
        torch.save(model.state_dict(), '/mnt/storage1/craig/kneron_fr/rgb/'+ args.ckpt_name+ '_state_dict.tar')
        model.load_state_dict(torch.load('/mnt/storage1/craig/kneron_fr/rgb/'+ args.ckpt_name+ '_state_dict.tar'))
        print("yeah")
    elif args.large_resnet:
        model = resnet50_large(args)

        from collections import OrderedDict
        trimmed_trained_model = OrderedDict()
        map_table = {   'module.':   '',
                        }

        for k in trained_model.state_dict().keys(): # GPU
            for prefix in map_table.keys():
                if k.startswith(prefix):
                    k2 = k.replace(prefix, map_table[prefix])
                    trimmed_trained_model[k2] = trained_model.state_dict()[k]

        model.load_state_dict(trimmed_trained_model)
        torch.save(model.state_dict(), '/mnt/storage1/craig/kneron_fr/rgb/'+ args.ckpt_name+ '_state_dict.tar')
        print("yeah_large")
    else:
        model = resnet50(args)
        '''
        from collections import OrderedDict
        trimmed_trained_model = OrderedDict()
        map_table = {   'module.':   '',
                        }

        for k in trained_model.state_dict().keys(): # GPU
            for prefix in map_table.keys():
                if k.startswith(prefix):
                    k2 = k.replace(prefix, map_table[prefix])
                    trimmed_trained_model[k2] = trained_model.state_dict()[k]
        '''
        model.load_state_dict(trained_model.state_dict())

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        img = torch.zeros((1, 3, 112, 112), device='cuda')

        #torch.onnx.export( model, img, '/mnt/storage1/craig/kneron_fr/rgb/'+ args.ckpt_name+ '.onnx', verbose=False, opset_version=9,  keep_initializers_as_inputs=True,
        #                input_names=['images'], output_names=['embedding'])

        torch.save(model.state_dict(), '/data2/johnnyalg1/model/FR/kgen/'+ args.ckpt_name+ '_state_dict.tar')
        #torch.save(model.state_dict(), '/mnt/storage1/craig/kneron_fr/rgb/'+ args.ckpt_name+ '_state_dict.tar')
        #torch.save(model.state_dict(), '/mnt/storage1/craig/kneron_fr/rgb/original_model_state_dict_0528.tar')
        print("yeah_ori")
    exit(0)


# load nir head
def finetune_network_nir(args):

    #checkpoint = torch.load(args.checkpoint)
    #trained_model = checkpoint['model']

    checkpoint = torch.load(args.checkpoint)
    trained_model = checkpoint['model']

    dict_trained = torch.load(args.checkpoint)
    checkpoint_state = torch.load(args.nir_head)
    nir_head = checkpoint_state['head']

    model = resnet50(args)

    model.load_state_dict(trained_model.state_dict())



    print(model)
    '''
    from collections import OrderedDict

    temp_rgb_head = OrderedDict()
    rgb_map_table = {   'head.':   '',
                    }

    for k in dict_trained.state_dict().keys(): # GPU
        for prefix in rgb_map_table.keys():
            if k.startswith(prefix):
                k2 = k.replace(prefix, rgb_map_table[prefix])
                temp_rgb_head[k2] = dict_trained.state_dict()[k]

    #model.load_state_dict(trained_model.state_dict())
    '''
    model.head.load_state_dict(nir_head.state_dict())
    exit(0)
    torch.save(model.state_dict(), '/mnt/storage1/craig/kneron_fr/rgb/'+ args.ckpt_name+ '_state_dict_nir_version.tar')
    #torch.save(model.state_dict(), '/mnt/storage1/craig/kneron_fr/rgb/original_model_state_dict_0528.tar')
    print("yeah_ori")
    exit(0)

def load_checkpoint_nir(args, num_labels):

    merged_nir_model = Merged_NIR(nir_emb_size = 256)
    dict_trained = torch.load(args.checkpoint)
    checkpoint_state = torch.load(args.nir_head)
    nir_head = checkpoint_state['head']


    from collections import OrderedDict
    temp_backbone = OrderedDict()
    map_table = {   'input.':   '0.',
                    'layer1.' :  '1.',
                    'layer2.' :  '2.',
                    'layer3.' :  '3.',
                    'layer4.' :  '4.',
                    }

    for k in dict_trained.state_dict().keys(): # GPU
        for prefix in map_table.keys():
            if k.startswith(prefix):
                k2 = k.replace(prefix, map_table[prefix])
                temp_backbone[k2] = dict_trained.state_dict()[k]

    temp_rgb_head = OrderedDict()
    rgb_map_table = {   'head.':   '',
                    }

    for k in dict_trained.state_dict().keys(): # GPU
        for prefix in rgb_map_table.keys():
            if k.startswith(prefix):
                k2 = k.replace(prefix, rgb_map_table[prefix])
                temp_rgb_head[k2] = dict_trained.state_dict()[k]

    merged_nir_model.backbone.load_state_dict(temp_backbone)
    merged_nir_model.rgb_head.load_state_dict(temp_rgb_head)
    merged_nir_model.nir_head.load_state_dict(nir_head.state_dict())

    output_name = os.path.split(args.nir_head)[1][:-4]
    #merged_nir_model.load_state_dict(torch.load('/mnt/storage1/craig/kneron_fr/for_test/' + output_name + '_state_dict.tar'))
    target_model = nn.Sequential(merged_nir_model.backbone, merged_nir_model.nir_head) #concatenate

    #torch.save(target_model.state_dict(), '/mnt/storage1/craig/kneron_fr/for_test/' + output_name + '_state_dict_m.tar')
    '''
    lucas_target = torch.load('/mnt/storage1/craig/kneron_fr/for_test/' + output_name + '_state_dict.tar')
    from collections import OrderedDict
    temp_lucas = OrderedDict()
    lucas_map_table = {   'backbone.':   '0.',
                        'nir_head.':   '1.',
                    }

    for k in lucas_target: # GPU
        for prefix in lucas_map_table.keys():
            if k.startswith(prefix):
                k2 = k.replace(prefix, lucas_map_table[prefix])
                temp_lucas[k2] = lucas_target[k]
    target_model.load_state_dict(temp_lucas)
    '''

    torch.save(merged_nir_model.state_dict(), '/mnt/storage1/craig/kneron_fr/for_test/' + output_name + '_state_dict.tar')
    exit(0)
    img = Image.open('/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/val/1_NIR/indoor_1_10_M_Y_C1_E0_G_0p30m_0000.png')


    img = np.asarray(img, dtype='f')
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(std=[256., 256., 256.], mean=[128, 128, 128]),
        #torchvision.transforms.RandomErasing(),
    ])
    img = data_transform(img)
    img = img.unsqueeze_(0)
    img = img.to(DEVICE)
    #print(img.shape)
    #exit(0)
    backbone = ResNet50Backbone(dict_trained)
    backbone.eval()
    nir_head.eval()
    merged_nir_model.eval()
    backbone = backbone.to(DEVICE)
    nir_head = nir_head.to(DEVICE)
    merged_nir_model = merged_nir_model.to(DEVICE)
    # original
    feature = backbone(img)
    output = nir_head(feature)

    # merged
    rgb_out, nir_out = merged_nir_model(img)


    print("output - nir_out:" , output- nir_out)
    exit(0)
    #return backbone, head, metric_fc_nir, optimizer, rgb_head




def train_net(args):
    # Initialize / load checkpoint
    model, metric_fc_rgb, optimizer = finetune_network(args)
    return 0


"""
def create_folder (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
"""

def run(args):
    cmdline = 'Command: '
    for s in sys.argv:
        if ' ' in s:
            s = '\''+s+'\''
        cmdline += s+' '
    cmdline += '\n'


    #if not args.log_dir:
    #    args.log_dir = LOG_DIR

    #create_folder(args.log_dir)
    #save_log(args.log_dir, TRAIN_LOG_FILE, cmdline, heading='Start training......\n')
    #save_log(args.log_dir, TRAIN_LOG_FILE, 'Arguments: '+str(args))

    train_net(args)



#if __name__ == '__main__':
