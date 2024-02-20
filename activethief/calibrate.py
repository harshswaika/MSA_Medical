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
from torchvision.models import resnet34#, ResNet34_Weights

from conf import cfg, load_cfg_fom_args

import utils
from train_utils_gbc import agree
from conf import cfg, load_cfg_fom_args

sys.path.append('/home/ankita/scratch/MSA_Medical')
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

            # if len(input_var.shape) == 5:
            #     images = input_var.squeeze(0)
            #     outputs =  model(images)
            #     _, pred = torch.max(outputs, dim=1)
            #     pred_label = torch.max(pred)
            #     pred_label = pred_label.unsqueeze(0)
            #     y_true.append([target_var.tolist()][0][0])
            #     y_pred.append([pred_label.tolist()])
            #     softmaxes.append(np.asarray(outputs.cpu()))

            # else:
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


if __name__ == "__main__":

    load_cfg_fom_args(description='Model Stealing')
    thief_model_dir = '/home/ankita/mnt/data_msa_medical/results/gbusg_radformer/GBUSV_resnet50/SGD/5000_val500/random_v8/'
    # thief_model_dir = '/home/ankita/mnt/data_msa_medical/results/gbusg_radformer/GBUSV_resnet50/SGD/15800_val1580/random_v3/'

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
    thief_model_path = os.path.join(thief_model_dir, f'trial_{trial}_cycle_{cycle}_best.pth')
    thief_model = load_thief_model(cfg, cfg.THIEF.ARCH, n_classes, cfg.ACTIVE.PRETRAINED_PATH)
    thief_model.load_state_dict(torch.load(thief_model_path)['state_dict'])
    thief_model = thief_model.cuda()
    print(f"Loaded thief model {thief_model_path}")

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
    if cfg.THIEF.HARD_LABELS is True:
        train_loader, val_loader, unlabeled_loader, list1 = create_thief_loaders(thief_data, thief_data_aug, labeled_set, 
                                                                            val_set, unlabeled_set, cfg.TRAIN.BATCH, 
                                                                            target_model)
    else:
        train_loader, val_loader, unlabeled_loader = create_thief_loaders_soft_labels(thief_data, thief_data_aug, 
                                                                labeled_set, val_set, unlabeled_set, cfg.TRAIN.BATCH, 
                                                                    target_model)
        
    # Set up temperature scaling
    temperature_model = TemperatureScaling(base_model=thief_model)
    temperature_model.cuda()

    print("\nRunning temp scaling:")
    temperature_model.calibrate(val_loader)
    print('best temp = ', temperature_model.T)
    
    # test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, temperature_model, criterion)
    acc, f1, spec, sens, ece, cce, _ = testz(temperature_model, test_loader, logits=True)
    print(["{:.2f}".format(temperature_model.T), cce, ece])
    print(f'Temperature model: acc = {acc:.4f}, agreement = {agr:.4f}, \
          f1 = {f1:.4f}, spec = {spec:.4f}, sens = {sens:.4f}, ECE {ece:.4f}, SCE {cce:.4f}')
    torch.save({'state_dict': temperature_model.state_dict(), 
                'temp': temperature_model.T},
                f'{cfg.SAVE_DIR}/trial_{trial}_cycle_{cycle}_temperature.pth')


    # Set up dirichlet scaling
    print("\nRunning dirichlet scaling:")
    lambdas = [0, 0.01, 0.1, 1, 10, 0.005, 0.05, 0.5, 5, 0.0025, 0.025, 0.25, 2.5]
    mus = [0, 0.01, 0.1, 1, 10]

    min_stats = {}
    min_error = float('inf')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(thief_model.parameters(), lr=cfg.TRAIN.LR, 
                                        momentum=cfg.TRAIN.MOMENTUM,weight_decay=cfg.TRAIN.WDECAY)

    for l in lambdas:
        for m in mus:
            # Set up dirichlet model
            dir_model = DirichletScaling(base_model=thief_model, num_classes=n_classes, optim=optimizer, Lambda=l, Mu=m)
            dir_model.cuda()

            # calibrate
            dir_model.calibrate(val_loader, lr=cfg.TRAIN.LR, epochs=cfg.TRAIN.EPOCH, patience=5)
            _, _, _, _, _, _, val_nll = testz(dir_model, val_loader, logits=True, criterion=criterion)
            acc, f1, spec, sens, ece, cce, test_nll = testz(dir_model, test_loader, logits=True, criterion=criterion)
            
            if val_nll < min_error:
                min_error = val_nll
                min_stats = {
                    "test_loss" : test_nll,
                    "top1" : acc,
                    "spec" : spec,
                    "sens" : sens,
                    "ece_score" : ece,
                    "sce_score" : cce,
                    "pair" : (l, m)
                }    

            print(["Dir=({:.2f},{:.2f})".format(l, m), test_nll, acc, spec, sens, cce, ece])
    
    print(["Best_Dir={}".format(min_stats["pair"]), 
                                            min_stats["test_loss"], 
                                            min_stats["top1"], 
                                            min_stats["spec"], 
                                            min_stats["sens"], 
                                            min_stats["sce_score"], 
                                            min_stats["ece_score"]])
    
    # train the model again for the best pair
    l, m = min_stats["pair"]
    dir_model = DirichletScaling(base_model=thief_model, num_classes=n_classes, optim=optimizer, Lambda=l, Mu=m)
    dir_model.cuda()
    # calibrate
    dir_model.calibrate(val_loader, lr=cfg.TRAIN.LR, epochs=cfg.TRAIN.EPOCH, patience=5)
    acc, f1, spec, sens, ece, cce, test_nll = testz(dir_model, test_loader, logits=True, criterion=criterion)
    print(f'\nDirichlet model: acc = {acc:.4f}, agreement = {agr:.4f}, \
          f1 = {f1:.4f}, spec = {spec:.4f}, sens = {sens:.4f}, ECE {ece:.4f}, SCE {cce:.4f}')
    torch.save({'state_dict': dir_model.state_dict(),
                'l': l,
                'm': m},
                f'{cfg.SAVE_DIR}/trial_{trial}_cycle_{cycle}_dirichlet.pth')