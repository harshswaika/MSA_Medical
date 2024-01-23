import numpy as np
import torch
import os
import torch as torch
import copy
# from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
# from MSA.config3 import ADDENDUM
from models import Simodel
# import lossnet2
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.stats import entropy
import scipy
import math
import sys
sys.path.append('vaal/')
from vaal import model,solver
from tqdm import tqdm
from collections import Counter, defaultdict
import torchattacks

def compute_adjustment(train_loader, tro, num_classes):
    """compute the base probabilities"""

    label_freq = {}
    
    # set all freqs to 0
    for i in range(num_classes):
        label_freq[i] = 0
    
    # compute label frequencies from train data
    for i, data in enumerate(train_loader):
        target = data[1].cuda()
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_uncertainty_mcd(model, unlabeled_loader, k):
    """
    Monte-Carlo dropout with k forward passes. Compute std dev
    """
    model.eval()
    enable_dropout(model)

    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for data in unlabeled_loader:
            inputs = data[0].cuda()
            # labels = labels.cuda()
            z=np.zeros((int(inputs.shape[0]),k,10))
            for i in range(k):
                scores = model(inputs)
                scores=F.softmax(scores)
                z[:,i,:]=scores.detach().cpu().numpy()
            z=np.sum(np.std(z,axis=1),axis=1)
            uncertainty = torch.cat((uncertainty,torch.tensor(z).cuda()), 0)
    
    return uncertainty.cpu()


def get_uncertainty_mcd_entropy(model, unlabeled_loader,k,num_classes):
    """
    Monte-Carlo dropout with k forward passes. Compute entropy
    """
    model.eval()
    enable_dropout(model)
    uncertainty = torch.tensor([]).float().cuda()

    with torch.no_grad():
        for data in tqdm(unlabeled_loader):
            inputs = data[0].cuda()
            # labels = labels.cuda()
            z=np.zeros((int(inputs.shape[0]),k,num_classes))
            for i in range(k):
                scores = model(inputs)
                scores=F.softmax(scores)
                z[:,i,:]=scores.detach().cpu().numpy()
            pred_sum = np.zeros((int(inputs.shape[0]),num_classes))
            for index in range(len(z)):
                pred_sum[index,:] = np.sum(z[index],axis=0)
           
            entropies = np.zeros((int(inputs.shape[0])))
            for index in range(len(pred_sum)):
                entropies[index] = entropy(pred_sum[index])
            uncertainty = torch.cat((uncertainty,torch.tensor(entropies).float().cuda()), 0)

    
    return uncertainty.cpu()

def deepfool1(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        #print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()#[::-1]

    #I = I[0:num_classes]
    label = I[-1]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot  #, loop_i, label, k_i, pert_image



def dfalv2(model, thief_data, unlabeled_idxs, batch_size=1):

    #deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50)
    model.eval()
    uncertainty = torch.tensor([])
    correct = torch.tensor([])
    indexes = torch.tensor([])
    
    unlabeled_loader = DataLoader(Subset(thief_data, unlabeled_idxs), batch_size=batch_size, 
                                        pin_memory=True, num_workers=4, shuffle=True)
    
    # with torch.no_grad():
    for data in tqdm(unlabeled_loader):
        inputs = data[0].cuda()
        labels = data[1]
        ind = data[2]
        l1 = torch.squeeze(inputs,0)
        pert = deepfool1(l1, model, num_classes=10, overshoot=0.02, max_iter=3)
        pert = (pert.flatten())
        pertnorm = np.linalg.norm(pert)
        pertnorm=pertnorm**2
        uncertainty = torch.cat((uncertainty, torch.tensor(pertnorm.reshape(1))), 0)
        indexes = torch.cat((indexes, ind), 0)

    return uncertainty, indexes
 
def dfalv1(model, thief_data, unlabeled_idxs, batch_size=256):

    #deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50)
    model.eval()
    uncertainty = torch.tensor([])
    correct = torch.tensor([])
    indexes = torch.tensor([])
    
    unlabeled_loader = DataLoader(Subset(thief_data, unlabeled_idxs), batch_size=batch_size, 
                                        pin_memory=True, num_workers=4, shuffle=True)
    
    # with torch.no_grad():
    for data in tqdm(unlabeled_loader):
        inputs = data[0].cuda()
        labels = data[1]
        ind = data[2]
        #print(labels, labels.shape)
        # f_image = model(inputs).cpu().detach().numpy().flatten()
        # I = (np.array(f_image)).flatten().argsort()[::-1]
        f_image = model(inputs).cpu().detach().numpy()
        I = (np.array(f_image)).argsort()
        label = I[:,-1]
        l1 = torch.from_numpy(label)
        inputs.requires_grad_(True)
        attack = torchattacks.DeepFool(model, steps=3, overshoot=0.02)
        adv_images = attack(inputs, l1)
        #pert = deepfool(inputs, model, num_classes=10, overshoot=0.02, max_iter=50)
        pert = torch.abs(inputs - adv_images).cpu().detach()
        for i in range(pert.shape[0]):
            pertq = (pert[i].numpy().flatten())
            pertnorm = np.linalg.norm(pertq)
            pertnorm=pertnorm**2
            uncertainty = torch.cat((uncertainty, torch.tensor(pertnorm.reshape(1))), 0)
        indexes = torch.cat((indexes, ind), 0)

    return uncertainty, indexes


def get_uncertainty_entropy(model, thief_data, unlabeled_idxs, batch_size=128):
    
    model.eval()
    uncertainty = torch.tensor([])
    correct = torch.tensor([])
    indexes = torch.tensor([])
    
    unlabeled_loader = DataLoader(Subset(thief_data, unlabeled_idxs), batch_size=batch_size, 
                                        pin_memory=False, num_workers=4, shuffle=True)
    
    with torch.no_grad():
        for data in tqdm(unlabeled_loader):
            inputs = data[0].cuda()
            labels = data[1]
            ind = data[2]
            scores = model(inputs)
            prob_dist = F.softmax(scores).detach().cpu().numpy()
            prbslogs = prob_dist * np.log2(prob_dist + sys.float_info.epsilon)
            numerator = 0 - np.sum(prbslogs, 1)
            denominator = np.log2(prob_dist.shape[1])
            entropy = numerator / denominator
            uncertainty = torch.cat((uncertainty, torch.tensor(entropy)), 0)
            indexes = torch.cat((indexes, ind), 0)

            # also compute score (just for verification, we don't need this for the algo)
            # _, preds = torch.max(scores.data, 1)
            # correct += (preds == labels).sum().item()
            # correct = torch.cat((correct, preds.cpu() == labels), 0)

    return uncertainty, indexes


def get_uncertainty_entropy_slow(model, unlabeled_loader):
    
    model.eval()
    uncertainty = torch.tensor([])
    correct = torch.tensor([])
    
    with torch.no_grad():
        for (inputs, labels, indexes) in tqdm((unlabeled_loader)):
            inputs = inputs.cuda()
            scores = model(inputs)
            prob_dist = F.softmax(scores).detach().cpu().numpy()
            prbslogs = prob_dist * np.log2(prob_dist + sys.float_info.epsilon)
            numerator = 0 - np.sum(prbslogs, 1)
            denominator = np.log2(prob_dist.shape[1])
            entropy = numerator / denominator
            uncertainty = torch.cat((uncertainty, torch.stack([torch.tensor(entropy),labels,indexes],dim=1) ), 0)
            
            # indexes = torch.tensor(list(range(idx*len(labels),idx*len(labels)+len(labels))))
            # uncertainty = torch.cat((uncertainty, torch.stack([torch.tensor(entropy),labels,indexes],dim=1) ), 0)
            
            # also compute score (just for verification, we don't need this for the algo)
            # _, preds = torch.max(scores.data, 1)
            # # correct += (preds == labels).sum().item()
            # correct = torch.cat((correct, preds.cpu() == labels), 0)

    return uncertainty


def get_uncertainty_true_loss(model, unlabeled_loader):
    
    model.eval()
    uncertainty = torch.tensor([])
    correct = torch.tensor([])
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for data in unlabeled_loader:
            inputs = data[0].cuda()
            labels = data[1].cuda()
            scores = model(inputs)
            loss = criterion(scores, labels)
            uncertainty = torch.cat((uncertainty, torch.tensor(loss).cpu()), 0)

    return uncertainty


def get_uncertainty_ll4al(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for data in unlabeled_loader:
            inputs = data[0].cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


def kl_divergence(target_model, thief_model, unlabeled_loader):
    
    target_model.eval()
    thief_model.eval()
    uncertainty = torch.tensor([])
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for data in unlabeled_loader:
            inputs = data[0].cuda()
            labels = data[1].cuda()

            target_scores = target_model(inputs)
            thief_scores = thief_model(inputs)
            pdist = F.softmax(target_scores, dim=1).detach().cpu().numpy()
            qdist = F.softmax(thief_scores, dim=1).detach().cpu().numpy()
            # kl = 0
            # for (p,q) in zip(pdist, qdist):
            #     kl += np.sum(np.where(p != 0, p * np.log(p / q), 0))

            kl = np.sum(np.where(pdist != 0, pdist * np.log(pdist / qdist), 0), 1)

            uncertainty = torch.cat((uncertainty, torch.tensor(kl).cpu()), 0)

    return uncertainty


def kl_divergence_custom(thief_model,labeled_loader, unlabeled_loader):
    
    thief_model.eval()

    labeled_sample_vectors = []
    with torch.no_grad():
        for data in labeled_loader:
            inputs = data[0].cuda()
            scores = thief_model(inputs)
            prob_vec = F.softmax(scores, dim=1).detach().cpu().numpy()
            labeled_sample_vectors.extend(prob_vec)

    unlabeled_sample_vectors = []
    with torch.no_grad():
        for data in unlabeled_loader:
            inputs = data[0].cuda()
            scores = thief_model(inputs)
            prob_vec = F.softmax(scores, dim=1).detach().cpu().numpy()
            unlabeled_sample_vectors.extend(prob_vec)
    
    uncertainty = [0]*len(unlabeled_sample_vectors)

    
    for i in tqdm(range(len(unlabeled_sample_vectors))):
        uncertainty_i = np.inf
        for labeled_vec in labeled_sample_vectors:
            pdist = unlabeled_sample_vectors[i]
            qdist = labeled_vec
            kl = np.sum(scipy.special.rel_entr(pdist,qdist))
            # kl = np.sum(np.where(pdist != 0, pdist * np.log(pdist / qdist), 0), 1)
            uncertainty_i = min(uncertainty_i,kl)
        uncertainty[i] = uncertainty_i 
    return uncertainty


def quries(X,ls,uls,model,k):
    """
    k-center greedy algorithm

    X: all samples from thief dataset (images only)
    ls: indices of labeled set
    uls: indices of unlabeled set
    model: thief model
    k: no. of samples to return
    """
    
    n=len(ls)+len(uls)
    l=ls+uls
    z=X[ls+uls].reshape((n,3,32,32)).cuda()
    # print(n)
    #cdprint(z.shape)
    #b=z[0:2].reshape([2,3,32,32])
    a=n//999
    model.eval()

    # obtain model predictions for all selected (labeled+unlabeled) data points
    for i in range(a+1):
        with torch.no_grad():
            if i==a:
                li=l[999*i:]
                z=X[li].reshape((n%999,3,32,32)).cuda()
            else:
                li=l[999*i:999*(i+1)]
                z=X[li].reshape((999,3,32,32)).cuda()
            label=model(z)
            # del(e)
            labelt=label.detach().cpu().numpy()
            del(label)
            gc.collect()
            if i==0:
                label1=labelt
            else:
                label1=np.concatenate((label1,labelt))
    #print(label1)
    
    # compute distances between all pairs of points
    dist_mat = np.matmul(label1, label1.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(n, 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    #print(dist_mat)

    idxs_lb = np.zeros(n, dtype=bool)
    idxs_tmp = np.arange(n)
    np.random.shuffle(idxs_tmp)
    idxs_lb[:len(ls)] = True
    #print(idxs_lb)
    lb_flag=idxs_lb.copy()
    #print(dist_mat)
    mat = dist_mat[~lb_flag, :][:, lb_flag]
    #print(mat)
    for i in range(k):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(n)[~lb_flag][q_idx_]
        lb_flag[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
    #print(lb_flag)
    #print(ls+uls)
    at=np.arange(n)[(idxs_lb ^ lb_flag)]

    return [(ls+uls)[i] for i in np.arange(n)[(idxs_lb ^ lb_flag)]]
    

def dfal(model,ul,max_iter):
    #model.cpu()
    model.eval()
    #uncertainty = torch.tensor([])
    n=len(ul)
    i=0
    uncertainty=torch.zeros((n),dtype=torch.float)
    for data in ul:
        input = data[0]
        nx=torch.unsqueeze(input[0], 0).cuda()
        nx.requires_grad_()
        eta=torch.zeros(nx.shape).cuda()
        out, e1 = model(nx+eta)
        n_class=out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter=0
        while py == ny and i_iter < max_iter:
            out[0,py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone().cuda()
            value_l = np.inf
            ri = None
            for i in range(n_class):
                if i == py:
                    continue
                nx.grad.data.zero_()
                out[0,i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi= grad_i -grad_np
                fi= out[0,i]-out[0,py]
                value_i=np.abs(fi.cpu().item()) / np.linalg.norm(wi.cpu().numpy().flatten())
                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.cpu().numpy().flatten()) * wi
            eta += ri.clone().cuda()
            nx.grad.data.zero_()
            out, e1 = model(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1
        z=(eta*eta).sum()
        #print(z)
        uncertainty[i] = z.data
        i+=1
        if i%1000==0:
            print(i)
    #model.cuda()
    return uncertainty


def get_samples_vaal(thief_model, unlabeled_loader, labeled_loader, num_labeled,budget):
    num_images = num_labeled
    budget = budget
    initial_budget = num_labeled
    num_classes = 10
    querry_dataloader = labeled_loader
    unlabeled_dataloader = unlabeled_loader
    vae = model.VAE(32)
    discriminator = model.Discriminator(32)
    solve = solver.Solver(num_images)
    vae, discriminator = solve.train(querry_dataloader,vae,discriminator,unlabeled_dataloader)
    # sampled_indices = solve.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
    
    all_preds = []
    all_indices = []
    for data in unlabeled_dataloader:
        images = data[0].cuda()
        with torch.no_grad():
            _,_,mu,_ = vae(images)
            preds = discriminator(mu)
        
        preds = preds.cpu().data
        all_preds.extend(preds)
        all_indices.extend(list(range(len(all_indices),len(all_indices)+256)))
    
    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    all_preds *= -1

    _, querry_indices = torch.topk(all_preds, budget)
    querry_pool_indices = np.asarray(all_indices)[querry_indices]
    return querry_pool_indices


def get_energy(model, unlabeled_loader, temper):
    
    model.eval()
    energy = torch.tensor([])
    
    with torch.no_grad():
        for data in tqdm(unlabeled_loader):
            inputs = data[0].cuda()
            labels = data[1]
            
            scores = model(inputs)
            batch_energy = (temper*torch.logsumexp(scores / temper, dim=1)).cpu()
            energy = torch.cat((energy, batch_energy), 0)

    return energy.numpy()


def get_energy_v2(model, unlabeled_loader, temper):
    
    model.eval()
    energy = torch.tensor([])
    indexes = torch.tensor([])
    
    with torch.no_grad():
        for data in tqdm(unlabeled_loader):
            inputs = data[0].cuda()
            labels = data[1]
            ind = data[2]
            
            scores = model(inputs)
            batch_energy = (temper*torch.logsumexp(scores / temper, dim=1)).cpu()
            energy = torch.cat((energy, batch_energy), 0)
            indexes = torch.cat((indexes, ind), 0)            

    return energy.numpy(), indexes.numpy().astype('int')


def main():
    torch.random.seed()
    X=torch.randn((2,10))
    z=F.softmax(X,dim=1)
    print(z)
    print(torch.log(z))
    z=-z*torch.log(z)
    print(z)
    print(torch.sum(z,axis=1))



if __name__=='__main__':
    main()


