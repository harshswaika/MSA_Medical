import csv
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision.models import resnet34#, ResNet34_Weights
from datasa.sampler import SubsetSequentialSampler

from models.victim import Victim
from models.Simodel import *
from models.cifar10_models import resnet34 as cifar10_resnet34
import datasets


def load_victim_dataset(cfg, dataset_name, model_arch):
    test_datasets = datasets.__dict__.keys()
    if dataset_name not in test_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(test_datasets))
    dataset = datasets.__dict__[dataset_name]

    # load model family
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    print('modelfamily: ', modelfamily)
    
    # load victim normalization transform as per victim model family
    if dataset_name == 'CIFAR10' and model_arch == 'cnn32':
        print("No normalization in test data")
        victim_mean_std = {'mean': (0.0,), 'std': (1.0,),}
    else:
        victim_mean_std = datasets.modelfamily_to_mean_std[modelfamily]
    victim_normalization_transform = transforms.Compose([transforms.Normalize(mean=victim_mean_std['mean'],
                                 std=victim_mean_std['std'])])
    # load test dataset
    test_transform = datasets.modelfamily_to_transforms_sans_normalization[modelfamily]['test']    
    testset = dataset(cfg, train=False, transform=test_transform) 
    n_classes = len(testset.classes)  
    
    return testset, victim_normalization_transform, n_classes


def load_victim_model(arch, model_path, normalization_transform, n_classes):
    # Define architecture
    if arch == 'cnn32':
        target_model = Simodel(channels=1) 
    elif arch == 'resnet32':
        target_model = cifar10_resnet34(num_classes=n_classes)
    elif arch == 'resnet34':
        target_model = resnet34(num_classes=n_classes)
    
    # Load weights
    try:
        state_dict = torch.load(model_path)['state_dict']
        state_dict = {key.replace("last_linear", "fc"): value for key, value in state_dict.items()}
        target_model.load_state_dict(state_dict, strict=False)
    except: 
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        target_model.load_state_dict(state_dict, strict=False)

    target_model=target_model.cuda()
    print(f"Loaded target model {model_path}")
    
    # Set victim model to use normalization transform internally
    target_model = Victim(target_model, normalization_transform)
    
    return target_model
    
    
def load_thief_dataset(cfg, dataset_name, data_root, target_model, victim_dataset_name, id_labels_file=None):
    
    if dataset_name == 'imagenet32':
        dataset = datasets.__dict__["ImageNet32"]
        if victim_dataset_name == 'MNIST':
            teacher_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            student_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        # elif victim_dataset_name == 'SVHN':
        #     teacher_transform = None
        #     student_transform = transforms.Compose([
        #         transforms.RandomCrop(32, padding=4),
        #         # transforms.RandomHorizontalFlip(),
        #     ])
        else:            
            teacher_transform = None
            student_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        
        thief_data = dataset(data_root, transform=teacher_transform)         
        thief_data_aug = dataset(data_root, transform=student_transform)  

    elif dataset_name == 'imagenet32_soft':
        dataset = datasets.__dict__["ImageNet32_Soft"]
        if victim_dataset_name == 'MNIST':
            teacher_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            student_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:            
            teacher_transform = None
            student_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        
        thief_data = dataset(data_root, transform=teacher_transform)         
        thief_data_aug = dataset(data_root, transform=student_transform)  
   
    elif dataset_name == 'imagenet_full':
        dataset = datasets.__dict__["ImageNet1k"]
        teacher_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
        student_transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.RandomCrop(224, pad_if_needed=True),
                # transforms.RandomRotation(5),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        thief_data = dataset(cfg, target_model, transform=teacher_transform)
        thief_data_aug = dataset(cfg, target_model, transform=student_transform)
        
    elif cfg.THIEF.DATASET == 'imagenet_subset':
        dataset = datasets.__dict__["ImageNet_Subset"]
        if id_labels_file is None:
            raise(AssertionError)
        else:
            with open(id_labels_file) as f:
                reader = csv.reader(f, delimiter=',')
                imagenet_id_labels = next(reader)
        
        imagenet_id_labels = [a for a in imagenet_id_labels if len(a) != 0]      
        print('Number of ID classes in imagenet: ', len(imagenet_id_labels))
        print(imagenet_id_labels)

        teacher_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
        student_transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.RandomCrop(224, pad_if_needed=True),
                # transforms.RandomRotation(5),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        thief_data = dataset(cfg, target_model, transform=teacher_transform, subset_class_names=imagenet_id_labels)
        thief_data_aug = dataset(cfg, target_model, transform=student_transform, subset_class_names=imagenet_id_labels)
    
    else:
        raise AssertionError('invalid thief dataset')
    
    return thief_data, thief_data_aug
        
    
def  create_thief_loaders(thief_data, thief_data_aug, labeled_set, val_set, unlabeled_set, batch_size, target_model):
    
    print("replacing labeled set labels with victim labels")
    # print(labeled_set)
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
            
                
    return train_loader, val_loader, unlabeled_loader, list1
        

def create_thief_loaders_soft_labels(cfg, thief_data, thief_data_aug, labeled_set, val_set, unlabeled_set, batch_size, target_model):
    
    print("replacing labeled set labels with victim labels")
    train_loader = DataLoader(Subset(thief_data, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=True)
    target_model.eval()
    with torch.no_grad():
        for data in tqdm(train_loader):
            d = data[0].cuda()
            ind0 = data[-1]

            # query hard labels
            if cfg.THIEF.HARD_LABELS is True:
                l = target_model(d).argmax(axis=1, keepdim=False)
                l = l.detach().cpu().tolist() 

                for ii, jj in enumerate(ind0):
                    thief_data_aug.samples[jj] = (thief_data_aug.samples[jj][0], l[ii], thief_data_aug.samples[jj][2])

            # query soft labels
            else:
                l_soft = target_model(d)
                l_soft = l_soft.detach().cpu()
                l = l_soft.argmax(axis=1, keepdim=False)
                l = l.detach().cpu().tolist()

                for ii, jj in enumerate(ind0):
                    thief_data_aug.samples[jj] = (thief_data_aug.samples[jj][0], l[ii], l_soft[ii])

        
    train_loader = DataLoader(Subset(thief_data_aug, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=True)
    unlabeled_loader = DataLoader(Subset(thief_data_aug, unlabeled_set), batch_size=batch_size, 
                                        pin_memory=False, num_workers=4, shuffle=True)
    
    print("replacing val labels with victim labels")
    val_loader = DataLoader(Subset(thief_data, val_set), batch_size=batch_size, 
                            pin_memory=False, num_workers=4, shuffle=True)
    target_model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader):
            d = data[0].cuda()
            ind0 = data[-1]

            # query hard labels
            if cfg.THIEF.HARD_LABELS is True:
                l = target_model(d).argmax(axis=1, keepdim=False)
                l = l.detach().cpu().tolist() 

                for ii, jj in enumerate(ind0):
                    thief_data.samples[jj] = (thief_data.samples[jj][0], l[ii], thief_data.samples[jj][2])

            # query soft labels
            else:
                l_soft = target_model(d)
                l_soft = l_soft.detach().cpu()
                l = l_soft.argmax(axis=1, keepdim=False)
                l = l.detach().cpu().tolist()

                for ii, jj in enumerate(ind0):
                    thief_data.samples[jj] = (thief_data.samples[jj][0], l[ii], l_soft[ii])

    return train_loader, val_loader, unlabeled_loader
        
    
def load_thief_model(cfg, arch, n_classes, pretrained_path):
    if arch == 'cnn32':
        thief_model = Simodel()
    elif arch == 'resnet32':
        thief_model = cifar10_resnet34(num_classes=n_classes)
    elif arch == 'resnet34':
        thief_model = resnet34(num_classes=n_classes)

    if cfg.ACTIVE.USE_PRETRAINED == True:
        print("Load pretrained model for initializing the thief")
        thief_state = thief_model.state_dict()
        # weights = ResNet34_Weights.DEFAULT
        pretrained_state = torch.load(pretrained_path)  #['state_dict']
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in thief_state and v.size() == thief_state[k].size() }
        thief_state.update(pretrained_state)
        thief_model.load_state_dict(thief_state, strict=False)
        # thief_model = resnet34(weights=weights,num_classes=n_classes)
    thief_model = thief_model.cuda()
    
    return thief_model
    
    