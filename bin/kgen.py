import os
import sys
import argparse

from ..conf.config import MODEL_STRUCTURE
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import kfr.bin  # noqa: F401
    __package__ = "kfr.bin"


#—net r50 —block-size ‘32 32 64 128 256’ —se 0 —focal True —emb 256 —do 0.0 —margin 0.5 —gr
def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--network', default='r50', help='specify network')
    parser.add_argument('--inp-size', type=int, default=112, help='input image size')
    parser.add_argument('--block-layout', type=str, default='8 28 6', help='feature block layout')
    #parser.add_argument('--block-size', type=str, default='32 384 1152 2144', help='feature block size')
    parser.add_argument('--block-size', type=str, default='32 32 64 128 256', help='feature block size') # original size
    #parser.add_argument('--blocksize-large', type=str, default='48 48 72 160 288', help='feature block size')
    parser.add_argument('--se-ratio', type=int, default=0, help='SE reduction ratio')
    parser.add_argument('--head', type=str, default='fc', help='head fc or varg')
    #parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--emb-size', type=int, default=256, help='embedding length')
    #parser.add_argument('--do-rate', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--do-rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
    parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
    parser.add_argument('--dropblock', help='dropblock.', action='store_true')
    #parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
    parser.add_argument('--focal-loss', type=bool, default=True, help='focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    #parser.add_argument('--checkpoint', type=str, default='None', help='checkpoint')
    parser.add_argument('--checkpoint', type=str, default='/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/checkpoint/kface_fr_gray-r50_nb-E256-av0.9980_0.9730_0.9786_0.9967_0.9260_0.9154.pth', help='checkpoint')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')
    parser.add_argument('--warmup', type=int, default=0, help='warmup training epochs without validation')
    parser.add_argument('--cooldown', type=int, default=0, help='keep training with repeating the last few epochs')
    parser.add_argument('--max-cool', type=int, default=5, help='Maxium cooling down without improvment')
    parser.add_argument('--end-epoch', type=int, default=100, help='training epoch size.')
    parser.add_argument('--gpus', type=str, default='0', help='running on GPUs ID')
    #parser.add_argument('--log-dir', type=str, default=None, help='Checkpoint/log root directory')
    parser.add_argument('--log-dir', type=str, default='/mnt/storage1/craig/kneron_fr', help='Checkpoint/log root directory')
    parser.add_argument('--ckpt_name', type=str, default='test123', help='Ckpt_name')
    #parser.add_argument('-gr', '--grayscale', help='Use grayscale input.', action='store_true')
    parser.add_argument('-nf', '--no-flip', help='No face flip in evaluation.', action='store_false')
    parser.add_argument('-pre', '--pre-norm', help='Preprocessing normalization id.', default='CV-kneron')
    parser.add_argument('--finetune', help='whether finetune from some checkpoint .', action='store_true')
    parser.add_argument('--large_resnet', help='large_resnet.', action='store_true')
    parser.add_argument('--regnet', help='regnet600MF.', action='store_true')
    parser.add_argument('--regnet800', help='regnet800MF.', action='store_true')
    parser.add_argument('--new_metric_fc', help='whether adjust the total label counts', action='store_true')
    parser.add_argument('--freeze_backbone', help='set model to eval()', action='store_true')

    #parser.add_argument('--nir-head', type=str, default='/mnt/storage1/craig/kneron_fr/0314_10_label_v23/checkpoint-r50-I112-E512-e0099-av0.9987_0.9990.tar', help='nir_head')
    #parser.add_argument('--nir-head', type=str, default='/mnt/storage1/craig/kneron_fr/0314_10_label_v30/checkpoint-r50-I112-E512-e0099-av0.9981_0.9982.tar', help='nir_head')
    #v23
    parser.add_argument('--nir-head', type=str, default='/mnt/storage1/craig/kneron_fr/0314_10_label_v23/checkpoint-r50-I112-E512-e0099-av0.9987_0.9990.tar', help='nir_head')
    #33
    #parser.add_argument('--nir-head', type=str, default='/mnt/storage1/craig/kneron_fr/0314_10_label_v33/checkpoint-r50-I112-E512-e0099-av0.9984_0.9988.tar', help='nir_head')



    args = parser.parse_args()
    args.block_layout = [int(n) for n in args.block_layout.split()]
    args.block_size = [int(n) for n in args.block_size.split()]
    args.gpus = [int(n) for n in args.gpus.split()]

    return args


args = parse_args()

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in args.gpus)

#from ..train.train import run
#from ..train.gen_rgb_head import run
#from ..train.gen_end2end import run
from ..train.gen_target_model_rgb import run

if __name__ == '__main__':
    sys.path.insert(0, MODEL_STRUCTURE)
    run(args)
