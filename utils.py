import sys
import time
import os
import shutil
import torch
from PIL import Image

from colorama import Fore


def create_save_folder(save_path, force=False, ignore_patterns=[]):
    if os.path.exists(save_path):
        print(Fore.RED + save_path + Fore.RESET
              + ' already exists!', file=sys.stderr)
        from getpass import getuser
        tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
                                                      os.path.basename(save_path),
                                                      time.time())
        print('move existing {} to {}'.format(save_path, Fore.RED
                                              + tmp_path + Fore.RESET))
        shutil.copytree(save_path, tmp_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)

    # copy code to save folder
    if save_path.find('debug') < 0:
        shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
                        ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
                                                      '*.path.tar', '*.pth',
                                                      '*.ipynb', '.*', 'data',
                                                      'save', 'save_backup',
                                                      save_path,
                                                      *ignore_patterns))


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate**2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                beta=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def compute_score(output, target):
    batch_size = target.size(0)
    prob = torch.sigmoid(output)
    pred = (prob > 0.5).long() #dim = 1
    target = target.long()
    correct = torch.sum(torch.eq(pred, target)).data[0]
    # total_pred = pred.numel()
    #print('pred:{}'.format(pred.data)) #batch, 1
    #print('target:{}'.format(target.data))
    #b = torch.mul(torch.eq(pred, target).long(), pred)
    tp = torch.sum(torch.mul(torch.eq(pred, target).long(), pred)).data[0]
    fp = torch.sum(torch.mul(torch.ne(pred, target).long(), pred)).data[0]
    tn = torch.sum(torch.mul(torch.eq(pred, target).long(), 1+(-1)*pred)).data[0]
    fn = torch.sum(torch.mul(torch.ne(pred, target).long(), 1+(-1)*pred)).data[0]
    # print(correct, tp, fp, tn, fn)
    return correct, tp, fp, tn, fn


