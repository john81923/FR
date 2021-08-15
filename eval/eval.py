import os
from os.path import basename, dirname
import torch
from ..util.lfw import load_bin, evaluate_pytorch, load_folder, evaluate_onnx
import sys
sys.path.append('kneron_hw_models/')
from fr.fr_runner import FrRunner

def load_val_dataset( VALID_LIST, input_shape=(112,112,3), flip=False, nrof_folds=10, to_grayscale=False):
    # load testset
    names = [ basename(item[0]) for item in VALID_LIST]
    val_dataset = []
    val_labels = []

    for item in VALID_LIST:
        #val_data, actual_issame = load_bin(item, image_size=input_shape[:-1], to_grayscale=to_grayscale)
        val_data, actual_issame = load_folder( item[0], item[1], flip=flip,image_size=input_shape[:-1])
        val_dataset.append(val_data)
        val_labels.append(actual_issame)

    return val_dataset, val_labels, names


def eval_model(model, val_dataset, val_labels, names, nrof_folds=10, visfpfn=False,
        log_dir=None, log_file=None, samplenumb=20, step=0, input_shape=(112,112,3), flip=False, data_transform_id='CV-kneron'):
    # run test
    accs = []
    vals = []
    for i in range(len(val_dataset)):
        acc, val, _ = evaluate_pytorch(model, val_dataset[i], val_labels[i], nrof_folds,
                log_dir=log_dir, log_file=log_file, step=step, img_size=input_shape, name=names[i],
                flip=flip, visfpfn=visfpfn, samplenumb=samplenumb, data_transform_id=data_transform_id)
        accs.append(acc)
        vals.append(val)

    return accs, vals

def load_val_dataset_debug( VALID_LIST, input_shape=(112,112,3), flip=False, nrof_folds=10, to_grayscale=False):
    # load testset
    names = [ basename(item[0]) for item in VALID_LIST]
    val_dataset = []
    val_labels = []

    for item in VALID_LIST:
        #val_data, actual_issame = load_bin(item, image_size=input_shape[:-1], to_grayscale=to_grayscale)
        (val_data, actual_issame), image_list = load_folder( item[0], item[1], flip=flip,image_size=input_shape[:-1])
        val_dataset.append(val_data)
        val_labels.append(actual_issame)

    return val_dataset, val_labels, names, image_list


def eval_model_debug(model, val_dataset, val_labels, names, nrof_folds=10, visfpfn=False,
        log_dir=None, log_file=None, samplenumb=20, step=0, input_shape=(112,112,3), flip=False, data_transform_id='CV-kneron'):
    # run test
    accs = []
    vals = []
    for i in range(len(val_dataset)):
        acc, val, fnfpids = evaluate_pytorch(model, val_dataset[i], val_labels[i], nrof_folds,
                log_dir=log_dir, log_file=log_file, step=step, img_size=input_shape, name=names[i],
                flip=flip, visfpfn=visfpfn, samplenumb=samplenumb, data_transform_id=data_transform_id)
        accs.append(acc)
        vals.append(val)

    return accs, vals, fnfpids

def eval_model_onnx(model, val_dataset, val_labels, names, nrof_folds=10,
        log_dir=None, log_file=None, step=0, input_shape=(112,112,3), flip=False, data_transform_id='CV-kneron'):
    # run test
    accs = []
    vals = []
    for i in range(len(val_dataset)):
        acc, val = evaluate_onnx(model, val_dataset[i], val_labels[i], nrof_folds,
                log_dir=log_dir, log_file=log_file, step=step, batch_size=1, img_size=input_shape, name=names[i],
                flip=flip, data_transform_id=data_transform_id)
        accs.append(acc)
        vals.append(val)
    return accs, vals



if __name__ == '__main__':
    VALID_LIST = [
            ['/mnt/sdd/johnnysun/data/FR/henan_gz_val/henan_gz_val_data_50.pkl',
              '/mnt/sdd/johnnysun/data/FR/henan_gz_val/henan_gz_val_issame_50.pkl'],
            ]
    PAIR_FOLDER_NAME = ''
    print('load val datasets')
    val_datasets, val_labels, names, image_list = load_val_dataset(VALID_LIST, flip=False)
    #sys.exit(1)
    # model 1
    print( "======= model1")
    v30_3 = '/mnt/storage1/craig/kneron_fr/0612_resnet_mi_v30_3/checkpoint-r50-I112-E256-e0059-av0.9998_1.0000.tar'
    print( 'checkpoint',v30_3)
    checkpoint = torch.load( v30_3)
    model = checkpoint['model']
    accs, vals, fnfpids = eval_model( model, val_datasets, val_labels, names, flip=False, samplenumb=100,visfpfn=f'/mnt/sdd/johnnysun/data/FR/VAL_visfpfn/{basename(dirname(v30_3))}', data_transform_id='CV-kneron' )
    fpids = fnfpids[1]
    print(fpids )
    for id in fpids:
        print( basename(dirname(image_list[id][0])), basename(dirname(image_list[id][1]))  )
    sys.exit(1)
    # model 2
    print( "======= model2")
    checkpoint = '/mnt/sdd/johnnysun/model/FR/0722_resnet_mi_v35_2/checkpoint-r50-I112-E256-e0068-av0.9996_1.0000.tar'
    print( 'checkpoint',checkpoint)
    checkpoint = torch.load( checkpoint)
    model = checkpoint['model']
    #eval_model( model, val_datasets, val_labels, names, flip=False, data_transform_id='CV-kneron' )

    print( "======= model3")
    checkpoint = '/mnt/sdd/johnnysun/model/FR/0722_resnet_mi_v35_2/checkpoint-r50-I112-E256-e0067-av0.9997_1.0000.tar'
    print( 'checkpoint',checkpoint)
    checkpoint = torch.load( checkpoint)
    model = checkpoint['model']
    #eval_model( model, val_datasets, val_labels, names, flip=False, data_transform_id='CV-kneron' )

    print( "======= model4")
    checkpoint = '/mnt/sdd/johnnysun/model/FR/0722_resnet_mi_v35_2/checkpoint-r50-I112-E256-e0066-av0.9997_1.0000.tar'
    print( 'checkpoint',checkpoint)
    checkpoint = torch.load( checkpoint)
    model = checkpoint['model']
    #eval_model( model, val_datasets, val_labels, names, flip=False, data_transform_id='CV-kneron' )

    print( "======= model5")
    checkpoint_name = '/mnt/sdd/johnnysun/model/FR/0722_resnet_mi_v35_2/checkpoint-r50-I112-E256-e0065-av0.9998_1.0000.tar'
    print( 'checkpoint',checkpoint_name)
    checkpoint = torch.load( checkpoint_name)
    model = checkpoint['model']
    eval_model( model, val_datasets, val_labels, names, flip=False, samplenumb=100,visfpfn=f'/mnt/sdd/johnnysun/data/FR/VAL_visfpfn/{basename(dirname(checkpoint_name))}', data_transform_id='CV-kneron' )

    print( "======= model6")
    checkpoint = '/mnt/sdd/johnnysun/model/FR/0722_resnet_mi_v35_2/checkpoint-r50-I112-E256-e0060-av0.9991_0.9998.tar'
    print( 'checkpoint',checkpoint)
    checkpoint = torch.load( checkpoint)
    model = checkpoint['model']
    #eval_model( model, val_datasets, val_labels, names, flip=False, data_transform_id='CV-kneron' )
