import os
import numpy as np
from PIL import Image
from gbusv_dataset import GbVideoDataset
from torchvision import transforms
import pickle
import random


def get_gbusv_ssl(cfg, root='data/datasets', n_lbl=4000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    os.makedirs(root, exist_ok=True) #create the root directory for saving data
    # augmentations
    transform_train = transforms.Compose([
        # RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        # CutoutRandom(n_holes=1, length=16, random=True)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])

    if ssl_idx is None:
        base_dataset = GbVideoDataset(root)
        val_dix, train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(cfg, base_dataset.samples)
        
        os.makedirs('data/splits', exist_ok=True)
        f = open(os.path.join('data/splits', f'cifar10_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx, 'val_idx': val_dix}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']
        val_dix = lbl_unlbl_dict['val_idx']

    lbl_idx = train_lbl_idx
    if pseudo_lbl is not None:
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #balance the labeled and unlabeled data 
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    train_lbl_dataset = GBUSV_SSL(
        root, lbl_idx, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None:
        train_nl_dataset = GBUSV_SSL(
            root, np.array(nl_idx), transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = GBUSV_SSL(root, train_unlbl_idx, transform=transform_val)
    val_dataset = GBUSV_SSL(root, val_dix, transform=transform_val)

    print("replacing labeled set labels with victim labels")
    thiefdataset = Subset(thief_data, labeled_set)
    train_loader = DataLoader(thiefdataset, batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=False)
    target_model.eval()
    list1=[]
    with torch.no_grad():
        for d, l0, ind0 in tqdm(train_loader):
            d = d.cuda()
            l = target_model(d).argmax(axis=1, keepdim=False)
            l = l.detach().cpu().tolist()
            for ii, jj in enumerate(ind0):
                thief_data_aug.samples[jj] = (thief_data_aug.samples[jj][0], l[ii])
                list1.append((jj.cpu().tolist(), l[ii]))
        
    train_loader = DataLoader(Subset(thief_data_aug, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=True)
    unlabeled_loader = DataLoader(Subset(thief_data_aug, unlabeled_set), batch_size=batch_size, 
                                        pin_memory=False, num_workers=4, shuffle=True)
    
    print("replacing val labels with victim labels")
    val_loader = DataLoader(Subset(thief_data, val_set), batch_size=batch_size, 
                            pin_memory=False, num_workers=4, shuffle=True)
    target_model.eval()
    with torch.no_grad():
        for d,l,ind0 in tqdm(val_loader):
            d = d.cuda()
            l = target_model(d).argmax(axis=1, keepdim=False)
            l = l.detach().cpu().tolist()
            # print(l)
            for ii, jj in enumerate(ind0):
                thief_data.samples[jj] = (thief_data.samples[jj][0], l[ii])
            

    if nl_idx is not None:
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, val_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, val_dataset


class GBUSV_SSL(GbVideoDataset):

    def __init__(self, root, indexs, 
                 transform=None, return_all_video_frames=True, data_split='all',
                 pseudo_idx=None, pseudo_target=None,
                nl_idx=None, nl_mask=None):
        
        super().__init__(root, transform=transform, 
                         return_all_video_frames=return_all_video_frames, 
                         data_split=data_split)
        
        self.data = np.array([e[0] for e in self.samples])
        self.targets = np.array([e[1] for e in self.samples])
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
        
        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        path, target = self.data[index], self.targets[index]

        ## Loading the image at the chosen index 
        if self.transform is not None:
            sample = self.loader(path)
            sample = self.transform(sample)

        # img = Image.fromarray(img)
        # if self.transform is not None:
            # img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.indexs[index], self.nl_mask[index]


def lbl_unlbl_split(cfg, lbls):
    indices = list(range(min(cfg.THIEF.NUM_TRAIN, len(lbls))))
    random.shuffle(indices)
    # do not use the entire unlabeled set, use only SUBSET number of samples
    indices = indices[:cfg.THIEF.SUBSET]
    val_set = indices[:cfg.ACTIVE.VAL]
    labeled_set = indices[cfg.ACTIVE.VAL:cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL]
    unlabeled_set = indices[cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL:]

    return val_set, labeled_set, unlabeled_set

# def lbl_unlbl_split(lbls, n_lbl, n_class):
#     lbl_per_class = n_lbl // n_class
#     lbls = np.array(lbls)
#     lbl_idx = []
#     unlbl_idx = []
#     for i in range(n_class):
#         idx = np.where(lbls == i)[0]
#         np.random.shuffle(idx)
#         lbl_idx.extend(idx[:lbl_per_class])
#         unlbl_idx.extend(idx[lbl_per_class:])
#     return lbl_idx, unlbl_idx
    