import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import shutil
import math
import os

def save_checkpoint(state, is_best, checkpoint, itr):
    filename=f'checkpoint_{itr}.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,f'model_best_{itr}.pth.tar'))


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        try:
            res.append(correct_k.mul_(100.0 / batch_size))
        except:
            res = (torch.tensor(0.0), torch.tensor(0.0))
    return res


def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.PL.NO_PROGRESS:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.PL.NO_PROGRESS:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.PL.NO_PROGRESS:
            test_loader.close()

    return losses.avg, top1.avg



def train_regular(args, lbl_loader, nl_loader, model, optimizer, scheduler, epoch, itr, iteration, num_classes ):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.PL.NO_PROGRESS:
        p_bar = tqdm(range(iteration))

    train_loader = zip(lbl_loader, nl_loader)
    model.train()
    for batch_idx, (data_x, data_nl) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs_x, targets_x, _, nl_mask_x = data_x
        inputs_nl, targets_nl, _, nl_mask_nl = data_nl

        inputs = torch.cat((inputs_x, inputs_nl)).to(args.device)
        targets = torch.cat((targets_x, targets_nl)).to(args.device)
        nl_mask = torch.cat((nl_mask_x, nl_mask_nl)).to(args.device)

        #network outputs
        logits = model(inputs)

        positive_idx = nl_mask.sum(dim=1) == num_classes #the mask for negative learning is all ones
        nl_idx = (nl_mask.sum(dim=1) != num_classes) * (nl_mask.sum(dim=1) > 0)
        loss_ce = 0
        loss_nl = 0

        #positive learning
        if sum(positive_idx*1) > 0:
            loss_ce += F.cross_entropy(logits[positive_idx], targets[positive_idx], reduction='mean')

        #negative learning
        if sum(nl_idx*1) > 0:
            nl_logits = logits[nl_idx]
            pred_nl = F.softmax(nl_logits, dim=1)
            pred_nl = 1 - pred_nl
            pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
            nl_mask = nl_mask[nl_idx]
            y_nl = torch.ones((nl_logits.shape)).to(device=args.device, dtype=logits.dtype)
            loss_nl += torch.mean((-torch.sum((y_nl * torch.log(pred_nl))*nl_mask, dim = -1))/(torch.sum(nl_mask, dim = -1) + 1e-7))

        loss = loss_ce + loss_nl
        loss.backward()
        losses.update(loss.item())

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if not args.PL.NO_PROGRESS:
            p_bar.set_description("Train PL-Iter: {itr}/{itrs:4}. Epoch: {epoch}/{epochs:4}. BT-Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                itr=itr + 1,
                itrs=args.PL.ITERATIONS,
                epoch=epoch + 1,
                epochs=args.TRAIN.EPOCH,
                batch=batch_idx + 1,
                iter=iteration,
                lr=scheduler.get_lr()[0],  #scheduler.get_last_lr()[0]
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.PL.NO_PROGRESS:
        p_bar.close()
    return losses.avg


def train_initial(args, train_loader, model, optimizer, scheduler, epoch, itr, iteration):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    if not args.PL.NO_PROGRESS:
        p_bar = tqdm(range(iteration))

    model.train()
    for batch_idx, (inputs, targets, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction='mean')
        loss.backward()
        losses.update(loss.item())

        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.PL.NO_PROGRESS:
            p_bar.set_description("Train PL-Iter: {itr}/{itrs:4}. Epoch: {epoch}/{epochs:4}. BT-Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                itr=itr + 1,
                itrs=args.PL.ITERATIONS,
                epoch=epoch + 1,
                epochs=args.TRAIN.EPOCH,
                batch=batch_idx + 1,
                iter=iteration,
                lr=scheduler.get_lr()[0],  #scheduler.get_last_lr()[0]
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
    if not args.PL.NO_PROGRESS:
        p_bar.close()

    return losses.avg