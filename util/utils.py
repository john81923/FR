# common utils shared by other components

import math
import os
from datetime import datetime

import torch


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_model_state(state, state_file):
    torch.save(state, state_file)

def save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, acc_val,
        near_end=False, heading='checkpoint', log_dir='.'):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc_val,
             'model': model,
             'metric_fc': metric_fc,
             'optimizer': optimizer}
    acc_str = '_'.join(['{:.4f}'.format(i) for i in acc_val])
    filename = os.path.join(log_dir, '{}-e{:04d}-av{}.tar'.format(heading, epoch, acc_str))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if epochs_since_improvement==0 or near_end:
        torch.save(state, filename)


def save_log(log_dir=None, log_file=None, log_info='', heading=None, stdout=False):
    if stdout:
        if heading:
            print('\n{}\n'.format(heading))
        print('{} \t{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), log_info))

    if log_dir and log_file:
        with open(os.path.join(log_dir, log_file),'at') as f:
            if heading:
                f.write('\n'+heading+'\n')
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' \t'+log_info+'\n')



class BestKeeper(object):
    """
    Keeps track of best of list each item, half-total, and total
    """

    def __init__(self, count):
        self.count = count
        self.reset()


    def reset(self):
        self.last_val = [0.0]*self.count
        self.last_sum = 0.
        self.best_val = [0.0]*self.count
        self.best_sum = 0.
        self.lapse_no_improve = 0
        if self.count > 1:
            self.last_half1 = 0.
            self.last_half2 = 0.
            self.best_half1 = 0.
            self.best_half2 = 0.


    def update(self, val):
        self.last_val = val
        self.last_sum = sum(self.last_val)
        if self.count > 1:
            self.last_half1 = sum(self.last_val[:(self.count+1)//2])
            self.last_half2 = sum(self.last_val[self.count//2:])

        is_best = self.is_best(self.last_val)
        if is_best:
            self.lapse_no_improve = 0
        else:
            self.lapse_no_improve += 1

        for i in range(len(self.best_val)):
            self.best_val[i] = max(self.last_val[i], self.best_val[i])

        self.best_sum = max(self.last_sum, self.best_sum)

        if self.count > 1:
            self.best_half1 = max(self.last_half1, self.best_half1)
            self.best_half2 = max(self.last_half2, self.best_half2)

        return self.lapse_no_improve


    def is_best(self, val):
        def is_better(val, new_val):
            EPSILON = 1e-10

            return val+EPSILON >= new_val

        is_best = False
        for i in range(len(self.best_val)):
            is_best = is_best or is_better(val[i], self.best_val[i])

        is_best = is_best or is_better(sum(val), self.best_sum)

        if self.count > 1:
            is_best = is_best or is_better(sum(val[:(self.count+1)//2]), self.best_half1)
            is_best = is_best or is_better(sum(val[self.count//2:]), self.best_half2)

        return is_best


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
