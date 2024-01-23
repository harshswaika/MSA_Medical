from __future__ import print_function, division
import os
import sys
import random
import math
import csv
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from PIL import Image


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler
from dataloader import GbDataset, GbRawDataset, GbCropDataset
import torchvision.transforms as T
import json
# from models import GbcNet
from resnet_gc import Resnet50

import utils
import utilsdfal
from train_utils_gbc import train_cutmix, train_mixup, train_augmix, testz, train_with_validation, train_with_kd, agree, dist, FocalLoss
from conf import cfg, load_cfg_fom_args
from loader_utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
if __name__ == "__main__":

    load_cfg_fom_args(description='Model Stealing')
    
    # if (cfg.THIEF.DATASET != 'imagenet32') and (cfg.THIEF.DATASET != 'imagenet32_soft'):
    #     torch.multiprocessing.set_start_method("spawn")
    width=224
    height=224
    set_dir="/home/harsh_s/scratch/datasets/GBCU-Shared"
    train_set_name="train.txt"
    test_set_name="test.txt"
    meta_file="/home/harsh_s/scratch/datasets/GBCU-Shared/roi_pred.json"
    img_dir="/home/harsh_s/scratch/datasets/GBCU-Shared/imgs"

    with open(meta_file, "r") as f:
        df = json.load(f)
    transforms = [T.ToPILImage()]
    transforms.append(T.Resize((width, height)))
    #transforms.append(T.RandomHorizontalFlip(0.25))
    transforms.append(T.ToTensor())
    img_transforms = T.Compose(transforms)
    
    val_transforms = T.Compose([T.ToPILImage(), T.Resize((width, height)),\
                                T.ToTensor()])
    
    train_labels = []
    t_fname = os.path.join(set_dir, train_set_name)
    with open(t_fname, "r") as f:
        for line in f.readlines():
            train_labels.append(line.strip())
    val_labels = []
    v_fname = os.path.join(set_dir, test_set_name)
    with open(v_fname, "r") as f:
        for line in f.readlines():
            val_labels.append(line.strip())

    train_dataset = GbRawDataset(img_dir, df, train_labels, img_transforms=img_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=5)
    testset = GbRawDataset(img_dir, df, val_labels, img_transforms=val_transforms)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=5)

    n_classes=3
    # Load victim dataset (test split only)
    # testset, victim_normalization_transform, n_classes = load_victim_dataset(cfg, cfg.VICTIM.DATASET, cfg.VICTIM.ARCH)
    # test_loader = DataLoader(testset, batch_size=128, num_workers=4, shuffle=False, pin_memory=False)  
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")
    # Load victim model    

    target_model = Resnet50(num_cls=3, last_layer=False, pretrain=True) 

    target_model.load_state_dict(torch.load(cfg.VICTIM.PATH))
    target_model.net = target_model.net.float().cuda()
    # target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH, victim_normalization_transform, n_classes)

    # Evaluate target model on target dataset: sanity check
    acc, f1, spec, sens = testz(target_model, test_loader)
    print(f"Target model acc = {acc}")
    print('Val-Acc: {:.4f} Val-Spec: {:.4f} Val-Sens: {:.4f}'\
            .format(acc, spec, sens))
    
    # Begin trials
    li = []
    results_arr = []
    uncertainty_arr = []
    for trial in range(cfg.RNG_SEED):
    #for trial in [cfg.RNG_SEED-1]:

        # Load thief dataset
        # Set up thief data with and without augmentation (teacher and student versions)
        # imagenet_id_labels_file = '/home/ankita/scratch/model_stealing/MSA/datasets/imagenet_birds.csv'
        # imagenet_id_labels_file = '/home/ankita/model_stealing/MSA/datasets/imagenet_8_random_classes.csv'
        
        main_directory = '/home/harsh_s/scratch/datasets/GBUSV-Shared'

# Define a transformation for the images (you can customize this based on your requirements)
        transforms1 = T.Compose([T.Resize((width, height)),\
                                T.ToTensor()])

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, main_directory, transform=None):
                self.main_directory = main_directory
                self.transform = transform

                # Get the list of subdirectories (each subdirectory represents a class)
                self.classes = [d.name for d in os.scandir(main_directory) if d.is_dir()]

                # Create a list to store image paths and corresponding labels
                self.data = []
                self.samples = []

                # Iterate through subdirectories to collect image paths and labels
                for i, class_name in enumerate(self.classes):
                    class_path = os.path.join(main_directory, class_name)
                    image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.jpg')]
                    self.data.extend([(img_path, i) for img_path in image_paths])
                    self.samples.append((img_path, i) for img_path in image_paths)
                print(len(self.samples))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                # Load and preprocess the image
                img_path, label = self.samples[index]
                image = Image.open(img_path)#.convert('RGB')

                if self.transform:
                    image = self.transform(image)

                return image, label, index

        # Create an instance of the custom dataset
        thief_data = CustomDataset(main_directory, transforms1)
        thief_data_aug = CustomDataset(main_directory, transforms1)

        # Create a data loader for the custom dataset
        batch_size = 16  # You can adjust this based on your needs
        # thief_data = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
        # thief_data_aug = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
        
        # imagenet_id_labels_file = None
        # thief_data, thief_data_aug = load_thief_dataset(cfg, cfg.THIEF.DATASET, cfg.THIEF.DATA_ROOT, target_model, cfg.VICTIM.DATASET, id_labels_file=imagenet_id_labels_file)
        
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
            train_loader, val_loader, unlabeled_loader,list1 = create_thief_loaders(thief_data, thief_data_aug, labeled_set, 
                                                                              val_set, unlabeled_set, cfg.TRAIN.BATCH, 
                                                                              target_model)
        else:
            train_loader, val_loader, unlabeled_loader = create_thief_loaders_soft_labels(cfg, thief_data, thief_data_aug, 
                                                                    labeled_set, val_set, unlabeled_set, cfg.TRAIN.BATCH, 
                                                                    target_model)

        dataloaders  = {'train': train_loader, 'test': test_loader, 'val': val_loader, 'unlabeled': unlabeled_loader}
        
        # print(list1)
        
        for cycle in range(cfg.ACTIVE.CYCLES):
            
            print('Validation set distribution: ')
            val_dist =   dist(val_set, dataloaders['val'])
            print(val_dist)

            print('Labeled set distribution: ')
            label_dist = dist(labeled_set, dataloaders['train'])
            print(label_dist)

            # Load thief model            
            # thief_model = load_thief_model(cfg, cfg.THIEF.ARCH, n_classes, cfg.ACTIVE.PRETRAINED_PATH)
            thief_model = Resnet50().cuda()

            print("Thief model initialized successfully")

            # Compute metrics on target dataset
            acc, f1, spec, sens = testz(thief_model, test_loader)
            agr = agree(target_model, thief_model, test_loader)
            print(f'Initial model on target dataset: acc = {acc:.4f}, agreement = {agr:.4f}, f1 = {f1:.4f}')
            
            # Compute metrics on validation dataset
            acc, f1, spec, sens = testz(thief_model, dataloaders['val'])
            agr = agree(target_model, thief_model, dataloaders['val'])
            print(f'Initial model on validation dataset: acc = {acc:.4f}, agreement = {agr:.4f}, f1 = {f1:.4f}')

            # Set up thief optimizer, scheduler
            criterion = nn.CrossEntropyLoss(reduction='none')
            # criterion = FocalLoss(gamma=3)
            if cfg.TRAIN.OPTIMIZER == 'SGD':
                optimizer = optim.SGD(thief_model.parameters(), lr=cfg.TRAIN.LR, 
                                        momentum=cfg.TRAIN.MOMENTUM,weight_decay=cfg.TRAIN.WDECAY)
            elif cfg.TRAIN.OPTIMIZER == 'Adam':
                optimizer = optim.Adam(thief_model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WDECAY)
            else:
                raise AssertionError('Unknown optimizer')
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)

            # Logit adjustment
            if cfg.ACTIVE.LA is True:
                logit_adjustments = utils.compute_adjustment(dataloaders['train'], tro=cfg.ACTIVE.TRO, num_classes=n_classes)
            else:
                logit_adjustments = None
            print('la: ', logit_adjustments)

            # Train thief model for current cycle
            if cfg.THIEF.HARD_LABELS is True:
                train_with_validation(thief_model, criterion, optimizer, scheduler, dataloaders, 
                                  cfg.TRAIN.EPOCH, trial, cycle, cfg.SAVE_DIR,
                                  la=logit_adjustments)
            else:
                train_with_kd(thief_model, criterion, optimizer, scheduler, dataloaders, 
                          cfg.TRAIN.EPOCH, trial, cycle, cfg.SAVE_DIR, 
                          temp=cfg.ACTIVE.TEMP, alpha=cfg.ACTIVE.ALPHA)


            # Compute accuracy and agreement on target dataset
            acc,f1, spec, sens = testz(thief_model, test_loader)
            agr = agree(target_model, thief_model, test_loader)
            print('Acc, agreement for latest model: ', acc, agr)

            # Load checkpoint for best model
            print("Load best checkpoint for thief model")
            best_model_path = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_best.pth')
            best_state = torch.load(best_model_path)['state_dict']
            thief_model.load_state_dict(best_state)

            # Compute accuracy and agreement on target dataset
            acc, f1, spec, sens = testz(thief_model, test_loader)
            agr = agree(target_model, thief_model, test_loader)
            print('Acc, agreement for best model: ', acc, agr)
            
            print('Trial {}/{} || Cycle {}/{} || Label set size {} || Test acc {} || Test agreement {}'.format(trial, cfg.RNG_SEED, cycle+1, cfg.ACTIVE.CYCLES, len(labeled_set), acc, agr))
            print("*"*100, "\n")

            # Select labeled subset for next cycle            
            if cycle!=cfg.ACTIVE.CYCLES-1:

                if cfg.ACTIVE.METHOD == 'random':
                    continue
                    
                elif cfg.ACTIVE.METHOD == 'entropy':
                    # Compute entropy for all unlabeled samples
                    uncertainty, indexes = utils.get_uncertainty_entropy(thief_model, thief_data, unlabeled_set)

                    # select samples with highest entropy
                    arg = np.argsort(uncertainty)
                    selected_index_list = indexes[arg][-(cfg.ACTIVE.ADDENDUM):].numpy().astype('int')

                elif cfg.ACTIVE.METHOD == 'dfal':
                    # Compute df perturbations for all unlabeled samples
                    pert, indexes = utilsdfal.dfalv1(thief_model, thief_data, unlabeled_set)

                    # select samples with highest entropy
                    arg = np.argsort(pert)
                    selected_index_list = indexes[arg][0:(cfg.ACTIVE.ADDENDUM)].numpy().astype('int')
                
                elif cfg.ACTIVE.METHOD == 'dk':
                    # Compute dfal+kcenter for all unlabeled samples
                    from kcenter_greedy import kCenterGreedy

                    pert, indexes = utilsdfal.dfalv1(thief_model, thief_data, unlabeled_set)

                    # select samples with highest entropy
                    arg = np.argsort(pert)
                    sele = indexes[arg][0:(cfg.ACTIVE.BUDGET)].numpy().astype('int')
                    sel = list(sele)
                
                    #afterdfal=Subset(thief_data, sel)



                    sampler = kCenterGreedy(thief_model, thief_data)
                                        
                    # select unlabeled points farthest from all centers
                    selected_index_list = sampler.select_batch(labeled_set, sel, N=cfg.ACTIVE.ADDENDUM)

                elif cfg.ACTIVE.METHOD == 'kcdfal':
                    from kcenter_greedy import kCenterGreedy
                    
                    # init greedy k center
                    # sampler = kCenterGreedy(thief_model, thief_data_aug)
                    sampler = kCenterGreedy(thief_model, thief_data)
                                        
                    # select unlabeled points farthest from all centers
                    sele = sampler.select_batch(labeled_set, unlabeled_set, N=cfg.ACTIVE.BUDGET)
                    sel = list(sele)

                    afterkc=Subset(thief_data, sel)

                    pert, indexes = utilsdfal.dfalv1(thief_model, thief_data, sel)

                    arg = np.argsort(pert)
                    selected_index_list = indexes[arg][0:(cfg.ACTIVE.ADDENDUM)].numpy().astype('int')

                elif cfg.ACTIVE.METHOD == 'kcenter':
                    from kcenter_greedy import kCenterGreedy
                    
                    # init greedy k center
                    # sampler = kCenterGreedy(thief_model, thief_data_aug)
                    sampler = kCenterGreedy(thief_model, thief_data)
                                        
                    # select unlabeled points farthest from all centers
                    selected_index_list = sampler.select_batch(labeled_set, unlabeled_set, N=cfg.ACTIVE.ADDENDUM)
                    
                else:
                    raise(AssertionError)
                
                l = list(selected_index_list)
                labeled_set += l
                unlabeled_set = [unlabeled_set[x] for x in range(len(unlabeled_set)) if unlabeled_set[x] not in selected_index_list]
                addendum_loader = DataLoader(Subset(thief_data_aug, l), batch_size=cfg.TRAIN.BATCH, 
                                                pin_memory=False, num_workers=4, shuffle=False)
                labeled_set = list(set(labeled_set))
                unlabeled_set = list(set(unlabeled_set))
                print('labeled set: ', len(labeled_set))
                print('unlabeled set: ', len(unlabeled_set))
                
                # Query the newly selected samples
                print("replacing addendum labels with victim labels")
                target_model.eval()
                with torch.no_grad():
                    for (d,l0,ind0) in tqdm(addendum_loader):
                        d = d.cuda()
                        label = target_model(d).argmax(axis=1,keepdim=False)
                        label = label.detach().cpu().tolist()
                        for ii, jj in enumerate(ind0):
                            thief_data_aug.samples[jj] = (thief_data_aug.samples[jj][0], label[ii])
                
                print('Label distribution of newly added samples after modification')
                label_dist_new = dist(l, addendum_loader)
                print(label_dist_new)

                # Update the labeled and unlabeled dataloaders
                dataloaders['train'] = DataLoader(Subset(thief_data_aug, labeled_set), batch_size=cfg.TRAIN.BATCH,
                                pin_memory=False, num_workers=4, shuffle=True)
                dataloaders['unlabeled'] = DataLoader(Subset(thief_data_aug, unlabeled_set), batch_size=cfg.TRAIN.BATCH, 
                                            pin_memory=False, num_workers=4, shuffle=True) 
                
                with open(f'{cfg.SAVE_DIR}/X_trial_{trial+1}_cycle_{cycle+1}_labeled_set.npy', 'wb') as f:
                    np.save(f, labeled_set)
                with open(f'{cfg.SAVE_DIR}/X_trial_{trial+1}_cycle_{cycle+1}_val_set.npy', 'wb') as f:
                    np.save(f, val_set) 
                

        # Final stats at the end of a trial
        trial_results = {}
        f = agree(target_model, thief_model, test_loader)
        trial_results['acc'] = acc
        trial_results['agr'] = f
        trial_results['label dist'] = dist(labeled_set, dataloaders['train']) #dist(val_set, val_loader) 
        results_arr.append(trial_results)
        li.append(f)
        
        with open(f'{cfg.SAVE_DIR}/X_trial_{trial+1}_cycle_{cycle+1}_labeled_set.npy', 'wb') as f:
            np.save(f, labeled_set)
        with open(f'{cfg.SAVE_DIR}/X_trial_{trial+1}_cycle_{cycle+1}_val_set.npy', 'wb') as f:
            np.save(f, val_set) 
            
    # Average agreement at the end of all trials
    li=np.array(li)
    print(np.mean(li),np.std(li))
    
    # Compile results of all trials
    df = pd.DataFrame.from_dict(results_arr)
    out_file = f'{cfg.SAVE_DIR}/results.csv'
    df.to_csv(out_file)
    print(df)

    # Compile uncertainty values
    # dfu = pd.DataFrame.from_dict(uncertainty_arr)
    # out_file = f'{cfg.SAVE_DIR}/uncertainty.csv'
    # dfu.to_csv(out_file)

    print('Results saved to ', cfg.SAVE_DIR)