import os
import sys
import random
import math
import csv
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision.models import resnet34, resnet50

from conf import cfg, load_cfg_fom_args

import utils
from train_utils_gbc import agree
from conf import cfg, load_cfg_fom_args
from loader_utils import *
from calibration_library.metrics import ECELoss, SCELoss
from calibration_library.calibrators import TemperatureScaling, DirichletScaling
from calibration_library.utils import AverageMeter


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def testz(model, dataloader, no_roi=True, verbose=True, logits=False, criterion=torch.nn.CrossEntropyLoss()):
    
    model.eval()
    y_true, y_pred = [], []
    softmaxes = []
    losses = AverageMeter()
    for i, (inp, target, fname) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.cuda())
            target_var = torch.autograd.Variable(target)
            outputs = model(input_var)

            # if soft labels, extract hard label from it
            if len(target_var.shape) > 1:
                target_var = target_var.argmax(axis=1)
            loss = criterion(outputs, target_var.cuda())
            losses.update(loss.item(), input_var.size(0))

            _, pred_label = torch.max(outputs, dim=1)
            y_pred.append(pred_label.tolist()) 
            softmaxes.extend(np.asarray(outputs.cpu()))
            
            y_true.append(target_var.tolist())

    y_pred = np.concatenate(y_pred, 0)
    y_true = np.concatenate(y_true, 0)
    softmaxes = np.asarray(softmaxes)
    print('y_pred ', y_pred.shape)
    print('y_true ', y_true.shape)
    print('softmaxes ', softmaxes.shape)

    # print('y_pred: ', y_pred)
    # print('y_true: ', y_true)

    acc = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
    sens = cfm[2][2]/np.sum(cfm[2])
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    eces = ECELoss().loss(softmaxes, y_true, n_bins=15, logits=logits)
    cces = SCELoss().loss(softmaxes, y_true, n_bins=15, logits=logits)

    if verbose == True:
        print('specificity = {}/{}'.format(cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1], np.sum(cfm[0]) + np.sum(cfm[1])))
        print('sensitivity = {}/{}'.format(cfm[2][2], np.sum(cfm[2])))
    
    return acc, f1, spec, sens, eces, cces, losses.avg


def compute_pseudolabel_acc(target_model, thief_model, thief_dataset, p_cutoff):
    
    dataloader = DataLoader(thief_dataset, batch_size=256,
                        pin_memory=False, num_workers=4, shuffle=False)
    target_model.eval()
    thief_model.eval()
    y_true, y_pred = [], []
    for (inp, target, fname) in tqdm(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.cuda())
            target_var = torch.autograd.Variable(target)

            outputs_thief = thief_model(input_var)
            _, pred_thief = torch.max(outputs_thief, dim=1)
            probs_thief = torch.softmax(outputs_thief, dim=-1)
            conf_thief, _ = torch.max(probs_thief.detach(), dim=-1)
            # y_pred.append(pred_thief.tolist()) 

            outputs_target = target_model(input_var)
            _, pred_target = torch.max(outputs_target, dim=1)
            # y_true.append(pred_target.tolist()) 

            for a, b, c in zip(conf_thief, pred_thief.tolist(), pred_target.tolist()):
                if a > p_cutoff:
                    y_pred.append(b)
                    y_true.append(c)

    # y_pred = np.concatenate(y_pred, 0)
    # y_true = np.concatenate(y_true, 0)

    acc = accuracy_score(y_true, y_pred)
    
    return acc


if __name__ == "__main__":

    load_cfg_fom_args(description='Model Stealing')
    thief_model_dir = 'results/gbusg_radformer/GBUSV_resnet50/SGD/5000_val500/random_v4/'
    # thief_model_dir = 'results/gbusg_radformer/GBUSV_resnet50/SGD/15800_val1580/random_v3/'

    trial = 1
    cycle = 1
    
    # Load victim dataset (test split only)
    testset, test_loader, n_classes = load_victim_dataset(cfg, cfg.VICTIM.DATASET)
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")

    # Load victim model    
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH)

    # Evaluate target model on target dataset: sanity check
    target_model.eval()
    acc, f1, spec, sens, ece, cce,_ = testz(target_model, test_loader, no_roi=False, logits=False)
    print('Target model Acc: {:.4f} Spec: {:.4f} Sens: {:.4f} ECE {:.4f} SCE {:.4f}'\
            .format(acc, spec, sens, ece, cce))

    
    # Load trained thief model
    # thief_model_path = os.path.join(thief_model_dir, f'trial_{trial}_cycle_{cycle}_best.pth')
    # thief_model_path = os.path.join(thief_model_dir, f'trial_{trial}_cycle_{cycle}_last.pth')
    thief_model_path = os.path.join(thief_model_dir, f'trial_{trial}_cycle_{cycle}_temperature.pth')


    thief_model = load_thief_model(cfg, cfg.THIEF.ARCH, n_classes, cfg.ACTIVE.PRETRAINED_PATH, load_pretrained=False)
    thief_state = thief_model.state_dict()
    print("Load thief model weights")
    pretrained_state = torch.load(thief_model_path) 
    if 'state_dict' in pretrained_state:
        pretrained_state = pretrained_state['state_dict']
    pretrained_state_common = {}
    for k, v in pretrained_state.items():
        if k in thief_state and v.size() == thief_state[k].size():
            pretrained_state_common[k] = v
        elif 'backbone.'+k in thief_state and v.size() == thief_state['backbone.'+k].size():
            pretrained_state_common['backbone.'+k] = v
        # remove 'module.' from pretrained state dict
        elif k[7:] in thief_state and v.size() == thief_state[k[7:]].size():
            pretrained_state_common[k[7:]] = v
        # remove 'base_model.' from pretrained state dict
        elif k[11:] in thief_state and v.size() == thief_state[k[11:]].size():
            pretrained_state_common[k[11:]] = v
        else:
            print('key not found', k)

    # print('pretrained state: ', pretrained_state_common.keys())
    assert(len(thief_state.keys()) == len(pretrained_state_common.keys()))
    thief_state.update(pretrained_state_common)
    thief_model.load_state_dict(thief_state, strict=True)
    thief_model = thief_model.cuda()

    # thief_model.load_state_dict(torch.load(thief_model_path)['state_dict'])
    # thief_model = thief_model.cuda()
    # print(f"Loaded thief model {thief_model_path}")

    # Compute accuracy and agreement on test dataset
    print('Thief model')
    thief_model.eval()
    acc, f1, spec, sens, ece, cce, _ = testz(thief_model, test_loader, logits=True)
    agr = agree(target_model, thief_model, test_loader)
    print(f'Thief model on target dataset: acc = {acc:.4f}, agreement = {agr:.4f}, \
          f1 = {f1:.4f}, spec = {spec:.4f}, sens = {sens:.4f}, ECE {ece:.4f}, SCE {cce:.4f}')

    # Load thief dataset
    thief_data, thief_data_aug = load_thief_dataset(cfg, cfg.THIEF.DATASET, cfg.THIEF.DATA_ROOT, target_model)

    # Setup validation, initial labeled, and unlabeled sets
    indices = list(range(min(cfg.THIEF.NUM_TRAIN, len(thief_data))))
    random.shuffle(indices)
    # do not use the entire unlabeled set, use only SUBSET number of samples
    indices = indices[:cfg.THIEF.SUBSET]
    val_set = indices[:cfg.ACTIVE.VAL]
    labeled_set = indices[cfg.ACTIVE.VAL:cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL]
    unlabeled_set = indices[cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL:]
    
    # Create train, val and unlabeled dataloaders
    # if cfg.THIEF.HARD_LABELS is True:
    #     train_loader, val_loader, unlabeled_loader, list1 = create_thief_loaders(thief_data, thief_data_aug, labeled_set, 
    #                                                                         val_set, unlabeled_set, cfg.TRAIN.BATCH, 
    #                                                                         target_model)
    # else:
    #     train_loader, val_loader, unlabeled_loader = create_thief_loaders_soft_labels(thief_data, thief_data_aug, 
    #                                                             labeled_set, val_set, unlabeled_set, cfg.TRAIN.BATCH, 
    #                                                                 target_model)
        
    
    # Compute pseudolabel accuracy
    placc_thief = compute_pseudolabel_acc(target_model, thief_model, thief_data, p_cutoff=0.98)
    print('Thief model pseudlabel acc = ', placc_thief)
            
    