import os
import sys
from datetime import datetime
import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from ..conf.config import DEVICE
from ..conf.config import IMAGE_ROOT, IMAGE_LIST, IMAGE_PER_LABEL, VALID_LIST, PAIR_FOLDER_NAME
from ..conf.config import LOG_DIR, TRAIN_LOG_FILE, EVAL_LOG_FILE
from ..conf.config import GRAD_CLIP, PRINT_FREQ, NUM_WORKERS

from ..model.resnet import  Head, ResNet50Backbone
#from ..model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, Head, ResNet50Backbone

from ..model.arc_margin import ArcMarginModel, update_arc_margin
from ..model.focal_loss import FocalLoss

from ..util.utils import AverageMeter, clip_gradient, accuracy, BestKeeper
from ..util.utils import save_model_state, save_log

from ..data.generator import ImageDataGenerator
from ..data.transform import image_transforms, data_transforms

from .validate import load_validate_dataset, validate_model

def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False

#https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
def freeze_partial_net_layers(net):
    for num, child in enumerate(net.children()):
        if num <8:
            print("abc: ", child.eval())
            for param in child.parameters():
                param.requires_grad = False
            #print(num, ":  ", child)

def build_net(args, num_labels):
    if args.network == 'r34':
        model = resnet34(args)
    elif args.network == 'r50':
        model = resnet50(args)
    elif args.network == 'r101':
        model = resnet101(args)
    elif args.network == 'r152':
        model = resnet152(args)
    else:
        model = resnet18(args)
    model = nn.DataParallel(model)
    metric_fc = ArcMarginModel(num_labels, args)
    metric_fc = nn.DataParallel(metric_fc)
    #metric_fc_nir = ArcMarginModel(num_labels, args)
    #metric_fc_nir = nn.DataParallel(metric_fc_nir)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                lr=args.lr, weight_decay=args.weight_decay)

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    return model, metric_fc, optimizer


def load_checkpoint(args):
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    model = checkpoint['model']
    metric_fc = checkpoint['metric_fc']
    update_arc_margin(metric_fc, args.margin_m)
    optimizer = checkpoint['optimizer']

    return model, metric_fc, optimizer, start_epoch, epochs_since_improvement

def load_checkpoint_nir(args, num_labels):
    #model = resnet50(args)
    #model.load_state_dict(torch.load(args.checkpoint) )
    #exit(0)
    model = torch.load(args.checkpoint)

    backbone = ResNet50Backbone(model)
    backbone.eval()

    #print(backbone)
    #exit(0)
    head = Head(emb_size=args.emb_size, head=args.head)
    #print(head)
    #https://blog.csdn.net/loseinvain/article/details/86476010
    #freeze_partial_net_layers(backbone)

    metric_fc_nir = ArcMarginModel(num_labels, args)

    if args.optimizer == 'sgd':
        #optimizer = torch.optim.SGD([{'params': head.parameters()}, {'params': metric_fc_nir.parameters()}],
        #        lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)

        optimizer = torch.optim.SGD([ {'params': metric_fc_nir.parameters()}],
                lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([ {'params': metric_fc_nir.parameters()}],
                lr=args.lr, weight_decay=args.weight_decay)

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    return backbone, head, metric_fc_nir, optimizer


def finetune_nir(args, num_labels):
    print("Finetune from: ", args.nir_head)

    model = torch.load(args.checkpoint)

    backbone = ResNet50Backbone(model)
    backbone.eval()

    #head = Head(emb_size=args.emb_size, head=args.head)
    checkpoint_state = torch.load(args.nir_head)
    head = checkpoint_state['head']

    #print(head)
    #metric_fc_nir = ArcMarginModel(num_labels, args)
    metric_fc_nir = checkpoint_state['metric_fc_nir']
    update_arc_margin(metric_fc_nir, args.margin_m)
    #print(metric_fc_nir)
    #print(metric_fc_nir.m )
    #exit(0)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': head.parameters()}, {'params': metric_fc_nir.parameters()}],
                lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': head.parameters()}, {'params': metric_fc_nir.parameters()}],
                lr=args.lr, weight_decay=args.weight_decay)

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    return backbone, head, metric_fc_nir, optimizer

def save_checkpoint(args, epoch, epochs_since_improvement, backbone, head, metric_fc_nir, optimizer, acc_val, val_enable =1 ):

    if epoch > (args.end_epoch-5):
        state = {#'epoch': epoch,
                 #'epochs_since_improvement': epochs_since_improvement,
                 #'acc': acc_val,
                 #'backbone': backbone,
                 'head': head,
                 'metric_fc_nir': metric_fc_nir,
                 #'optimizer': optimizer
                 }
    else:
        state = {
                 'head': head,
                 #'metric_fc_nir': metric_fc_nir,
                 }

    heading = 'checkpoint-{}-I{}-E{}'.format(args.network, args.inp_size, args.emb_size)
    if (val_enable==1):
        acc_str = '_'.join(['{:.4f}'.format(i) for i in acc_val])
    else:
        acc_str = "_test_"
    filename = os.path.join(args.log_dir, '{}-e{:04d}-av{}.tar'.format(heading, epoch, acc_str))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    #if epochs_since_improvement==0 or (args.end_epoch-epoch <= 5 and args.cooldown == 0):
    save_model_state(state, filename)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)

    #TO_GRAYSCALE = args.grayscale
    DATA_TRANSFORM_ID = 'CV-kneron' #'kneron'
    #IMG_TRANSFORM_ID = 'kneron-gray' if TO_GRAYSCALE else 'kneron'
    IMG_TRANSFORM_ID_INDOOR = 'kneron-gray-indoor'
    IMG_TRANSFORM_ID_OUTDOOR = 'kneron-gray-outdoor'
    IMG_TRANSFORM_ID = 'kneron-gray'

    # Custom dataloaders
    for root in IMAGE_ROOT:
        save_log(args.log_dir, TRAIN_LOG_FILE, 'Loading data from directory: '+root)

    train_loader = ImageDataGenerator(image_root=IMAGE_ROOT, image_list=IMAGE_LIST,
        image_transform_indoor=image_transforms[IMG_TRANSFORM_ID_INDOOR], image_transform_outdoor=image_transforms[IMG_TRANSFORM_ID_OUTDOOR],
        image_transform=image_transforms[IMG_TRANSFORM_ID],
        data_transform=data_transforms[DATA_TRANSFORM_ID],
        batch_size=args.batch_size, images_per_label=IMAGE_PER_LABEL, shuffle=True, num_workers=NUM_WORKERS)
    # Initialize / load checkpoint
    if (args.finetune):
        print("finetune_nir")
        backbone, head, metric_fc_nir, optimizer = finetune_nir(args, train_loader.dataset.num_labels)
        #backbone, head, metric_fc_nir, optimizer = finetune_nir(args, 0)
        start_epoch = 0
        epochs_since_improvement = 0

    else:
        if args.checkpoint is None:
            print("not_expectm no checkpoint")
            exit(0)
            start_epoch = 0
            epochs_since_improvement = 0
            model, metric_fc, optimizer = build_net(args, train_loader.dataset.num_labels)
        else:
            print("load_checkpoint_nir")
            print("loading: ", args.checkpoint)
            backbone, head, metric_fc_nir, optimizer = load_checkpoint_nir(args, train_loader.dataset.num_labels)
            start_epoch = 0
            epochs_since_improvement = 0



    val_enable =1
    if (val_enable):
        val_datasets, val_labels, names = load_validate_dataset(VALID_LIST, PAIR_FOLDER_NAME, to_grayscale=True)



    # Move to GPU, if available
    backbone = backbone.to(DEVICE)
    head = head.to(DEVICE)
    metric_fc_nir = metric_fc_nir.to(DEVICE)

    #logging.info("Freeze net.")
    #freeze_net_layers(net.base_net)
    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(DEVICE)
    else:
        criterion = nn.CrossEntropyLoss().to(DEVICE)

    #scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.end_epoch, last_epoch=start_epoch)
    #scheduler = CosineAnnealingLR(optimizer, T_max=(args.end_epoch/2), last_epoch=start_epoch)

    if (val_enable):
        best_val = BestKeeper(len(VALID_LIST)*2)

    # Epochs
    #for epoch in range(start_epoch, args.end_epoch):
    epoch = start_epoch
    max_cool = 0
    while epoch < args.end_epoch:
        #scheduler.step()

        lr_info = 'Epoch: [{}] begins with learning rate:'.format(epoch)
        for param_group in optimizer.param_groups:
            lr_info += ' {:.6f} '.format(param_group['lr'])

        save_log(args.log_dir, TRAIN_LOG_FILE, lr_info, stdout=True)

        start = datetime.now()
        # One epoch's training
        backbone.eval()

        #double ensure
        for m in backbone.bn1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                #m.weight.requires_grad = False
                #m.bias.requires_grad = False
        for m in backbone.layer1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                #m.weight.requires_grad = False
                #m.bias.requires_grad = False
        for m in backbone.layer2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                #m.weight.requires_grad = False
                #m.bias.requires_grad = False
        for m in backbone.layer3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                #m.weight.requires_grad = False
                #m.bias.requires_grad = False
        for m in backbone.layer4.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        if (args.finetune):
            head.train()
        else:
            head.eval()
        metric_fc_nir.train()
        train_loss, train_top5_accs = train(train_loader=train_loader,
                backbone=backbone, head = head, metric_fc_nir=metric_fc_nir, criterion=criterion,
                optimizer=optimizer, epoch=epoch, log_dir=args.log_dir)
        scheduler.step()

        end = datetime.now()
        epoch_time = 'Epoch: [{}] duration: {} seconds'.format(epoch, (end-start).seconds)
        save_log(args.log_dir, TRAIN_LOG_FILE, epoch_time, stdout=True)

        # skip validation and checkpoint during warmup training
        if epoch < args.warmup and (epoch%5 != 0):
            warmup_message = 'Epoch: [{}] skiping validation with {} epochs warming up'.format(epoch, args.warmup)
            save_log(args.log_dir, TRAIN_LOG_FILE, warmup_message, stdout=True)
            continue

        # One epoch's validation

        if (val_enable):
            accs, vals = validate_model(backbone, head, val_datasets, val_labels, names,
                    log_dir=args.log_dir, log_file=EVAL_LOG_FILE, step=epoch, flip=args.no_flip, data_transform_id=args.pre_norm)


        if (val_enable):
            acc_val = accs + vals
            print("acc_val: ", acc_val)
            epochs_since_improvement = best_val.update(acc_val)
        else:
            acc_val = 'test'
            epochs_since_improvement = 0

        if epochs_since_improvement > 0:
            improve_message = 'Epoch: [{}] no improvement during last {} epochs'.format(epoch, epochs_since_improvement)
            save_log(args.log_dir, TRAIN_LOG_FILE, improve_message, stdout=True)
        else:
            max_cool = 0

        # Save checkpoint
        save_checkpoint(args, epoch, epochs_since_improvement, backbone, head, metric_fc_nir, optimizer, acc_val, val_enable)

        epoch += 1
        if epoch == args.end_epoch and args.cooldown > 0 and max_cool <= args.max_cool:
            max_cool += 1
            cooldown_message = 'Epoch: [{}] cooling down by repeating {} epochs'.format(epoch, args.cooldown)
            save_log(args.log_dir, TRAIN_LOG_FILE, cooldown_message, stdout=True)
            args.end_epoch += args.cooldown
            scheduler = CosineAnnealingLR(optimizer, T_max=args.end_epoch, last_epoch=epoch-1)


def train(train_loader, backbone, head, metric_fc_nir, criterion, optimizer, epoch, log_dir=None):
    #backbone.eval()
    #head.train() # train mode (dropout and batchnorm is used)
    #metric_fc_nir.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        # Forward prop.
        backbone_out = backbone(img)
        feature = head(backbone_out)
        output = metric_fc_nir(feature, label)

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, GRAD_CLIP)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(output, label, 5)
        top5_accs.update(top5_accuracy)

        log_info = ('Epoch: [{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top5 Accuracy {top5_accs.val:.4f} ({top5_accs.avg:.4f})'.format(
            epoch, i, len(train_loader), loss=losses, top5_accs=top5_accs))

        # Print status
        last_printed = False
        if i % PRINT_FREQ == 0:
            last_printed = True
            save_log(log_dir, TRAIN_LOG_FILE, log_info, stdout=True)

    # print the last if not already done
    if last_printed == False:
        save_log(log_dir, TRAIN_LOG_FILE, log_info, stdout=True)

    return losses.avg, top5_accs.avg

def create_folder (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run(args):
    cmdline = 'Command: '
    for s in sys.argv:
        if ' ' in s:
            s = '\''+s+'\''
        cmdline += s+' '
    cmdline += '\n'


    if not args.log_dir:
        args.log_dir = LOG_DIR

    create_folder(args.log_dir)
    save_log(args.log_dir, TRAIN_LOG_FILE, cmdline, heading='Start training......\n')
    save_log(args.log_dir, TRAIN_LOG_FILE, 'Arguments: '+str(args))

    train_net(args)


#if __name__ == '__main__':
