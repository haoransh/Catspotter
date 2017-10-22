from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from utils import AverageMeter, adjust_learning_rate, compute_score
import time


class Trainer(object):
    def __init__(self, model, criterion=None, optimizer=None, args=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args

    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # switch to train mode
        self.model.train()
        lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                  self.args.decay_rate, epoch,
                                  self.args.epochs)  # TODO: add custom
        print('Epoch {:3d} lr = {:.6e}'.format(epoch, lr))
        end = time.time()
        sample_cnt, all_loss, all_correct, all_tp, all_fp, all_tn, all_fn =0,0,0,0,0,0,0
        for i, (input, target,name) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            #target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target).float().unsqueeze(1)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)
            sample_cnt += target_var.size(0)
            correct, tp, fp, tn, fn = compute_score(output, target_var)
            all_correct += correct
            all_tp += tp
            all_fp += fp
            all_tn += tn
            all_fn += fn
            all_loss += loss.data[0]
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        accuracy = all_correct/sample_cnt
        if all_tp+all_fp !=0:
            precision = float(all_tp)/(all_tp+all_fp)
        else:
            precision = 0
        if all_tp+all_fn!= 0:
            recall = float(all_tp)/(all_tp+all_fn)
        else:
            recall = 0
        f1 = 2*precision*recall/(precision+recall)
        print('epoch {} training accuracy:{}, total_cnt:{}, precision:{}, recall:{}, f1:{}'.format(\
                epoch, accuracy, sample_cnt, precision, recall, f1))
        return all_loss, accuracy, lr

    def test(self, val_loader, epoch, silence=False):
        batch_time = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        end = time.time()
        sample_cnt = 0
        all_loss, all_correct, all_tp, all_fp, all_tn, all_fn =0, 0,0,0,0,0
        for i, (input, target, name) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True).float().unsqueeze(1)
            sample_cnt += target_var.size(0)
            output = self.model(input_var)
            prob = torch.sigmoid(output)
            loss = self.criterion(output, target_var)
            correct, tp, fp, tn, fn = compute_score(output, target_var)
            print('file :{} correct:{} probdata:{}'.format(name, correct, prob.data[0]))
            all_correct += correct
            all_tp += tp
            all_fp += fp
            all_tn += tn
            all_fn += fn
            all_loss += loss.data[0]
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        if not silence:
            accuracy = all_correct/sample_cnt
            if all_tp+all_fp !=0:
                precision = float(all_tp)/(all_tp+all_fp)
            else:
                precision = 0
            if all_tp+all_fn!= 0:
                recall = float(all_tp)/(all_tp+all_fn)
            else:
                recall = 0
            f1 = 2*precision*recall/(precision+recall)
            print('epoch {} evaluation accuracy:{}, total_cnt:{}, precision:{}, recall:{}, f1:{}'.format(epoch, \
                accuracy, sample_cnt , precision, recall, f1))

        return all_loss, accuracy
