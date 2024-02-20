import argparse
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from collections import OrderedDict
import pickle
import numpy as np
from re import search
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter

from UPS.utils_train_test import train_initial, train_regular, test, get_cosine_schedule_with_warmup, save_checkpoint
from UPS.pseudo_labeling_util import pseudo_labeling
from UPS.gbusv import get_gbusv_ssl

import utils
from train_utils_gbc import testz, train_with_validation, train_with_kd, agree, dist
from conf import cfg, load_cfg_fom_args
from loader_utils import *


def main():
    # run_started = datetime.today().strftime('%d-%m-%y_%H%M') #start time to create unique experiment name

    load_cfg_fom_args(description='Model Stealing')
    writer = SummaryWriter(cfg.SAVE_DIR)

    # Load victim dataset (test split only)
    testset, test_loader, n_classes = load_victim_dataset(cfg, cfg.VICTIM.DATASET)
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")
    
    # Load victim model    
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH)

    # Evaluate target model on target dataset: sanity check
    acc, f1, spec, sens = testz(target_model, test_loader, no_roi=False)
    print(f"\nTarget model acc = {acc}")
    print('Val-Acc: {:.4f} Val-Spec: {:.4f} Val-Sens: {:.4f}'\
            .format(acc, spec, sens))
    
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
    train_loader, val_loader, unlabeled_loader, list1 = create_thief_loaders(thief_data, thief_data_aug, labeled_set, 
                                                                              val_set, unlabeled_set, cfg.TRAIN.BATCH, 
                                                                              target_model)

    start_itr = 0
    for itr in range(start_itr, cfg.PL.ITERATIONS):
        if itr == 0 and cfg.ACTIVE.BUDGET < 4000: #use a smaller batchsize to increase the number of iterations
            cfg.TRAIN.BATCH = 64
            cfg.TRAIN.EPOCH = 1024

        # if os.path.exists(f'data/splits/{args.dataset}_basesplit_{args.n_lbl}_{args.split_txt}.pkl'):
        #     lbl_unlbl_split = f'data/splits/{args.dataset}_basesplit_{args.n_lbl}_{args.split_txt}.pkl'
        # else:
        lbl_unlbl_split = None
        
        #load the saved pseudo-labels
        if itr > 0:
            pseudo_lbl_dict = f'{cfg.SAVE_DIR}/pseudo_labeling_iteration_{str(itr)}.pkl'
        else:
            pseudo_lbl_dict = None
        
        # Load thief dataset
        lbl_dataset, nl_dataset, unlbl_dataset, val_dataset = get_gbusv_ssl(cfg, cfg.THIEF.DATA_ROOT, cfg.ACTIVE.BUDGET,
                                                                lbl_unlbl_split, pseudo_lbl_dict, itr, cfg.METHOD_NAME)

        # Load thief model
        thief_model = load_thief_model(cfg, cfg.THIEF.ARCH, n_classes, cfg.ACTIVE.PRETRAINED_PATH)

        nl_batchsize = int((float(cfg.TRAIN.BATCH) * len(nl_dataset))/(len(lbl_dataset) + len(nl_dataset)))

        if itr == 0:
            lbl_batchsize = cfg.TRAIN.BATCH
            iteration = len(lbl_dataset) // cfg.TRAIN.BATCH
        else:
            lbl_batchsize = cfg.TRAIN.BATCH - nl_batchsize
            iteration = (len(lbl_dataset) + len(nl_dataset)) // cfg.TRAIN.BATCH

        lbl_loader = DataLoader(
            lbl_dataset,
            sampler=RandomSampler(lbl_dataset),
            batch_size=lbl_batchsize,
            num_workers=4,
            drop_last=True)

        nl_loader = DataLoader(
            nl_dataset,
            sampler=RandomSampler(nl_dataset),
            batch_size=nl_batchsize,
            num_workers=4,
            drop_last=True)

        val_loader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=cfg.TRAIN.BATCH,
            num_workers=4)
        
        unlbl_loader = DataLoader(
            unlbl_dataset,
            sampler=SequentialSampler(unlbl_dataset),
            batch_size=cfg.TRAIN.BATCH,
            num_workers=4)

        optimizer = optim.SGD(thief_model.parameters(), lr=cfg.TRAIN.LR, 
                              momentum=cfg.TRAIN.MOMENTUM,weight_decay=cfg.TRAIN.WDECAY)
        args.total_steps = cfg.TRAIN.EPOCH * iteration
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup * iteration, args.total_steps)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)
        
        start_epoch = 0

        # if args.resume and itr == start_itr and os.path.isdir(args.resume):
        #     resume_itrs = [int(item.replace('.pth.tar','').split("_")[-1]) for item in resume_files if 'checkpoint_iteration_' in item]
        #     if len(resume_itrs) > 0:
        #         checkpoint_itr = max(resume_itrs)
        #         resume_model = os.path.join(args.resume, f'checkpoint_iteration_{checkpoint_itr}.pth.tar')
        #         if os.path.isfile(resume_model) and checkpoint_itr == itr:
        #             checkpoint = torch.load(resume_model)
        #             best_acc = checkpoint['best_acc']
        #             start_epoch = checkpoint['epoch']
        #             model.load_state_dict(checkpoint['state_dict'])
        #             optimizer.load_state_dict(checkpoint['optimizer'])
        #             scheduler.load_state_dict(checkpoint['scheduler'])

        thief_model.zero_grad()
        best_acc = 0
        for epoch in range(start_epoch, cfg.TRAIN.EPOCH):
            if itr == 0:
                train_loss = train_initial(cfg, lbl_loader, thief_model, optimizer, scheduler, epoch, itr, iteration)
            else:
                train_loss = train_regular(cfg, lbl_loader, nl_loader, thief_model, optimizer, scheduler, epoch, itr, iteration, n_classes)

            val_loss = 0.0
            val_acc = 0.0
            test_model = thief_model
            if epoch > (cfg.TRAIN.EPOCH+1)/2 and epoch % 10==0:
                val_loss, val_acc = test(cfg, val_loader, test_model)
                _, test_acc = test(cfg, test_loader, test_model)
            elif epoch == (cfg.TRAIN.EPOCH-1):
                val_loss, val_acc = test(cfg, val_loader, test_model)
                _, test_acc = test(cfg, test_loader, test_model)

            writer.add_scalar('train/1.train_loss', train_loss, (itr*cfg.TRAIN.EPOCH)+epoch)
            writer.add_scalar('val/1.val_acc', val_acc, (itr*cfg.TRAIN.EPOCH)+epoch)
            writer.add_scalar('val/2.val_loss', val_loss, (itr*cfg.TRAIN.EPOCH)+epoch)
            writer.add_scalar('test/1.test_acc', test_acc, (itr*cfg.TRAIN.EPOCH)+epoch)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            model_to_save = thief_model.module if hasattr(thief_model, "module") else thief_model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, cfg.SAVE_DIR, f'iteration_{str(itr)}')
    
        checkpoint = torch.load(f'{cfg.SAVE_DIR}/checkpoint_iteration_{str(itr)}.pth.tar')
        thief_model.load_state_dict(checkpoint['state_dict'])
        thief_model.zero_grad()

        #pseudo-label generation and selection
        pl_loss, pl_acc, pl_acc_pos, total_sel_pos, pl_acc_neg, total_sel_neg, unique_sel_neg, pseudo_label_dict = pseudo_labeling(args, unlbl_loader, model, itr)

        writer.add_scalar('pseudo_labeling/1.regular_loss', pl_loss, itr)
        writer.add_scalar('pseudo_labeling/2.regular_acc', pl_acc, itr)
        writer.add_scalar('pseudo_labeling/3.pseudo_acc_positive', pl_acc_pos, itr)
        writer.add_scalar('pseudo_labeling/4.total_sel_positive', total_sel_pos, itr)
        writer.add_scalar('pseudo_labeling/5.pseudo_acc_negative', pl_acc_neg, itr)
        writer.add_scalar('pseudo_labeling/6.total_sel_negative', total_sel_neg, itr)
        writer.add_scalar('pseudo_labeling/7.unique_samples_negative', unique_sel_neg, itr)

        with open(os.path.join(cfg.SAVE_DIR, f'pseudo_labeling_iteration_{str(itr+1)}.pkl'),"wb") as f:
            pickle.dump(pseudo_label_dict,f)
        
        with open(os.path.join(cfg.SAVE_DIR, 'log.txt'), 'a+') as ofile:
            ofile.write(f'############################# PL Iteration: {itr+1} #############################\n')
            ofile.write(f'Last Test Acc: {test_acc}, Best Test Acc: {best_acc}\n')
            ofile.write(f'PL Acc (Positive): {pl_acc_pos}, Total Selected (Positive): {total_sel_pos}\n')
            ofile.write(f'PL Acc (Negative): {pl_acc_neg}, Total Selected (Negative): {total_sel_neg}, Unique Negative Samples: {unique_sel_neg}\n\n')

    writer.close()


if __name__ == '__main__':
    cudnn.benchmark = True
    main()