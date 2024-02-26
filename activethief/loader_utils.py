import csv
from tqdm import tqdm
import os, sys
import json

from PIL import Image
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, resnet152, resnet18, resnet101
from timm.models.vision_transformer import VisionTransformer

sys.path.append('GBCNet')
from GBCNet.dataloader import GbDataset, GbRawDataset, GbCropDataset
from GBCNet.models import Resnet50 as Resnet50_GC
from GBCNet.models import GbcNet

sys.path.append('RadFormer')
from RadFormer.models import RadFormer
from RadFormer.dataloader import GbUsgDataSet, GbUsgRoiTestDataSet

from types import SimpleNamespace
sys.path.append('/home/deepankar/scratch/model_stealing_encoders_copy/wise-ft')
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.zeroshot import get_zeroshot_classifier

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor

class Victim(nn.Module):
    """class for victim model

    Args:
        nn (_type_): _description_
    """
    def __init__(self, model, arch):
        super(Victim, self).__init__()
        self.model = model
        self.arch = arch
        
    def forward(self, x):
        # change forward here
        out = self.model(x)
        # if self.arch == 'radformer':
        #     return out[2]
        return out


def load_victim_model(arch, model_path):
    # Define architecture
    if arch == 'resnet50_usucl':
        target_model.net = resnet50(num_classes=3) 
        target_model.net.fc = nn.Sequential(
                          nn.Linear(num_ftrs, 256), 
                          nn.ReLU(inplace=True), 
                          nn.Dropout(0.4),
                          nn.Linear(256, 3)
                        )
    elif arch == 'resnet50_gc':
        target_model = Resnet50_GC(num_cls=3, last_layer=False, pretrain=True) 
    elif arch == 'gbcnet':
        target_model = GbcNet(num_cls=3, pretrain=False)
    elif arch == 'radformer':
        target_model = RadFormer(local_net='bagnet33', \
                        num_cls=3, \
                        global_weight=0.55, \
                        local_weight=0.1, \
                        fusion_weight=0.35, \
                        use_rgb=True, num_layers=4, pretrain=False)
    
    # Load weights
    print('target model keys: ', len(target_model.state_dict().keys()))
    checkpoint_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint_dict:
        checkpoint_dict = checkpoint_dict['state_dict']
    print('checkpoint keys: ', len(checkpoint_dict.keys()))
    target_model.load_state_dict(checkpoint_dict, strict=True)
    # target_model.net = target_model.net.float().cuda()
    target_model = Victim(target_model.float().cuda(), arch)
    
    
    return target_model


def load_thief_model(cfg, arch, n_classes, pretrained_path, load_pretrained=True):

    
    if arch == 'resnet34':
        thief_model = resnet34(num_classes=n_classes)
    elif arch =='medclip_vit':
        # thief_model=MedCLIPModel(vision_cls=MedCLIPVisionModelViT,checkpoint='/home/deepankar/scratch/MSA_Medical/').cuda()
        thief_model=MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        thief_model.from_pretrained()
        thief_model.cuda()
        print(thief_model)
        return thief_model
    elif arch == 'vit_b_16_1k':
        # use this model definition for imagenet1k trained weights
        from torchvision.models import vit_b_16
        thief_model = vit_b_16(num_classes=n_classes)
    elif arch == 'vit_b_16_21k_CLIP':
        args = SimpleNamespace(model= 'ViT-B/16', device= 'cuda',cache_dir= 'cache',classnames='openai',template='simple_template',train_dataset='CIFAR10',data_location='data',batch_size=128)
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
        classifier.save('zeroshotthief.pt')
        thief_model=ImageClassifier.load('zeroshotthief.pt')
        # print("Preprocess_fn",thief_model.train_preprocess)
        thief_model.process_images=True
        thief_model = thief_model.cuda()
        # print(thief_model)
        return thief_model
    elif arch == 'vit_b_16_21k':
        # use this model definition for imagenet21k trained weights
        from vit import vit_base_patch16_224
        thief_model = vit_base_patch16_224(num_classes=n_classes)
    elif arch == 'vit_l_32_21k':
        # use this model definition for imagenet21k trained weights
        from pytorch_pretrained_vit import ViT
        thief_model = ViT('L_32',pretrained=False,num_classes=n_classes)
    elif arch == 'resnet18':
        thief_model = resnet18(num_classes=n_classes)
    elif arch == 'resnet50':
        thief_model = resnet50(num_classes=n_classes)
    elif arch == 'resnet152':
        thief_model = resnet152(num_classes=n_classes)
    elif arch == 'resnet101':
        thief_model = resnet101(num_classes=n_classes)
    elif arch == 'radformer':
        thief_model = RadFormer(local_net='bagnet33', \
                        num_cls=n_classes, \
                        global_weight=0.55, \
                        local_weight=0.1, \
                        fusion_weight=0.35, \
                        use_rgb=True, num_layers=4, pretrain=True,
                        load_local=True)
    elif arch == 'deit':
        thief_model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', 
                                     pretrained=False, num_classes=n_classes)
        pretrained_state = pretrained_state['model']

    elif arch == 'vit':
        thief_model = vit_b_16(num_classes=n_classes)

    # elif arch == 'resnet50_usucl':
        # thief_model.net = resnet50(num_classes=3) 
        # thief_model.net.fc = nn.Sequential(
        #                   nn.Linear(num_ftrs, 256), 
        #                   nn.ReLU(inplace=True), 
        #                   nn.Dropout(0.4),
        #                   nn.Linear(256, 3)
        #                 )

    if load_pretrained == True :
        thief_state = thief_model.state_dict()
        print('thief state: ', print(thief_state.keys()))

        
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in thief_state and v.size() == thief_state[k].size() }
        print('pretrained state: ', pretrained_state.keys())
        thief_state.update(pretrained_state)
        thief_model.load_state_dict(thief_state, strict=True)
    thief_model = thief_model.cuda()
    
    return thief_model


def load_victim_dataset(cfg, dataset_name):
    
    if dataset_name == 'GBC':
        n_classes=3
        set_dir=cfg.VICTIM.DATA_ROOT
        meta_file=os.path.join(set_dir, 'roi_pred.json') 
        img_dir=os.path.join(set_dir, 'imgs') 
        test_set_name="test.txt"

        with open(meta_file, "r") as f:
            df = json.load(f)
        
        val_transforms = transforms.Compose([
                                            transforms.ToPILImage(), 
                                            transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.HEIGHT)),
                                            transforms.ToTensor()
                                            ])
        
        val_labels = []
        v_fname = os.path.join(set_dir, test_set_name)
        with open(v_fname, "r") as f:
            for line in f.readlines():
                val_labels.append(line.strip())
        testset = GbCropDataset(img_dir, df, val_labels, to_blur=False, sigma=0, p=0.15, img_transforms=val_transforms)        

        # val_transforms = transforms.Compose([
        #                                     transforms.Resize((cfg.VICTIM.WIDTH)),
        #                                     transforms.CenterCrop(224),
        #                                     transforms.ToTensor()
        #                                     ])
        # testset = GbUsgRoiTestDataSet(data_dir=img_dir, df=df, image_list_file=os.path.join(set_dir, test_set_name), 
                                    #   to_blur=False, sigma=0, transform=val_transforms)        
        test_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=5)

    
    elif dataset_name == 'gbusg':
        n_classes = 3
        set_dir=cfg.VICTIM.DATA_ROOT
        img_dir=os.path.join(set_dir, 'imgs') 
        list_file = os.path.join(set_dir, 'test.txt')
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        testset = GbUsgDataSet(data_dir=img_dir, 
                                image_list_file=list_file,
                                transform=transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))

    test_loader = DataLoader(dataset=testset, batch_size=1, 
                                shuffle=False, num_workers=0)

    return testset, test_loader, n_classes

   
def load_thief_dataset(cfg, dataset_name, data_root, target_model):
   
    if dataset_name == 'imagenet_full':
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

    elif 'GBUSV' in dataset_name:
        from gbusv_dataset import GbVideoDataset

        # Create an instance of the custom dataset
        if cfg.VICTIM.ARCH == 'gbcnet':
            transforms1 = transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.HEIGHT)),\
                                transforms.ToTensor()])
        elif cfg.VICTIM.ARCH == 'radformer':
            normalize = transforms.Normalize(  
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
            # transforms1 = transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH)),
                                            # transforms.CenterCrop(224),
                                            # transforms.ToTensor(), 
                                            # normalize])
            transforms1 = transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.WIDTH)),
                                            transforms.ToTensor(), 
                                            normalize])
            # transforms2= transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.WIDTH)),
                                            #  transforms.RandomCrop(224, pad_if_needed=True),
                                            #  transforms.RandomRotation(5),
                                            #  transforms.ColorJitter(0.1, 0.1, 0.1, 0), 
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.ToTensor(),
                                            # normalize
                                            # ])
        
        if dataset_name == 'GBUSV':
            thief_data = GbVideoDataset(data_root, transforms1,pickle_root='/home/deepankar/scratch/MSA_Medical/')
            thief_data_aug = GbVideoDataset(data_root, transforms1,pickle_root='/home/deepankar/scratch/MSA_Medical/')
           
        elif dataset_name == 'GBUSV_benign':
            thief_data = GbVideoDataset(data_root, transforms1, data_split='benign',pickle_root='/home/deepankar/scratch/MSA_Medical/')
            thief_data_aug = GbVideoDataset(data_root, transforms1, data_split='benign',pickle_root='/home/deepankar/scratch/MSA_Medical/')
           
        elif dataset_name == 'GBUSV_malignant':
            thief_data = GbVideoDataset(data_root, transforms1, data_split='malignant',pickle_root='/home/deepankar/scratch/MSA_Medical/')
            thief_data_aug = GbVideoDataset(data_root, transforms1, data_split='malignant',pickle_root='/home/deepankar/scratch/MSA_Medical/')
           

        
    else:
        raise AssertionError('invalid thief dataset')
    
    return thief_data, thief_data_aug
        
    
def create_thief_loaders(thief_data, thief_data_aug, labeled_set, val_set, unlabeled_set, batch_size, target_model):
    
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
    # unlabeled_loader = DataLoader(Subset(thief_data_aug, unlabeled_set), batch_size=batch_size, 
    #                                     pin_memory=False, num_workers=4, shuffle=True)
    unlabeled_loader = None
    
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
        

def create_thief_loaders_soft_labels(thief_data, thief_data_aug, labeled_set, val_set, unlabeled_set, batch_size, target_model):
    
    print("replacing labeled set labels with victim labels")
    thiefdataset = Subset(thief_data, labeled_set)
    train_loader = DataLoader(thiefdataset, batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=False)
    
    target_model.eval()
    with torch.no_grad():
        for d, l0, ind0 in tqdm(train_loader):
            d = d.cuda()
            l_soft = target_model(d)
            l_soft = l_soft.detach().cpu()
            # print(l_soft)
            l_hard = l_soft.argmax(axis=1, keepdim=False)
            l_hard = l_hard.detach().cpu().tolist()
            for ii, jj in enumerate(ind0):
                thief_data_aug.samples[jj] = (thief_data_aug.samples[jj][0], l_soft[ii])

        
    train_loader = DataLoader(Subset(thief_data_aug, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=True)
    unlabeled_loader = DataLoader(Subset(thief_data_aug, unlabeled_set), batch_size=batch_size, 
                                        pin_memory=False, num_workers=4, shuffle=True)
    
    print("replacing val labels with victim labels")
    val_loader = DataLoader(Subset(thief_data, val_set), batch_size=batch_size, 
                            pin_memory=False, num_workers=4, shuffle=True)
    target_model.eval()
    with torch.no_grad():
        for d, l0, ind0 in tqdm(val_loader):
            d = d.cuda()
            l_soft = target_model(d)
            l_soft = l_soft.detach().cpu()
            l_hard = l_soft.argmax(axis=1, keepdim=False)
            l_hard = l_hard.detach().cpu().tolist()
            for ii, jj in enumerate(ind0):
                thief_data.samples[jj] = (thief_data.samples[jj][0], l_soft[ii])
           
    return train_loader, val_loader, unlabeled_loader
        
    

    
    