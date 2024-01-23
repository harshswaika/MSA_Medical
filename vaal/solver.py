import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy





class Solver:
    def __init__(self, num_images, budget):
        self.num_images = num_images

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(budget)
        self.train_epochs = 100
        self.batch_size = 128
        self.cuda = True
        self.num_vae_steps = 2
        self.beta = 1
        self.adversary_param = 1
        self.num_adv_steps = 1


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, in dataloader:
                    yield img


    # def train(self, querry_dataloader, val_dataloader, task_model, vae, discriminator, unlabeled_dataloader):
    def train(self, querry_dataloader, vae, discriminator, unlabeled_dataloader):
        self.train_iterations = (self.num_images * self.train_epochs) // self.batch_size
        # self.train_iterations = 200
        print(f"Num iterations: {self.train_iterations}")
        
        lr_change = self.train_iterations // 4
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        # optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)


        vae.train()
        discriminator.train()
        # task_model.train()

        if self.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            # task_model = task_model.cuda()
        
        best_acc = 0
        for iter_count in range(self.train_iterations):
            # if iter_count is not 0 and iter_count % lr_change == 0:
            #     for param in optim_task_model.param_groups:
            #         param['lr'] = param['lr'] / 10
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # # task_model step
            # preds = task_model(labeled_imgs)
            # task_loss = self.ce_loss(preds, labels)
            # optim_task_model.zero_grad()
            # task_loss.backward()
            # optim_task_model.step()

            # VAE step
            for count in range(self.num_vae_steps):
                recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, self.beta)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones((labeled_imgs.size(0),1))
                unlab_real_preds = torch.ones((unlabeled_imgs.size(0),1))
                    
                if self.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.adversary_param * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            # Discriminator step
            for count in range(self.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)
                
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones((labeled_imgs.size(0),1))
                unlab_fake_preds = torch.zeros((unlabeled_imgs.size(0),1))

                if self.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()
                
                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                        self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

                

            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                # print('Current task model loss: {:.4f}'.format(task_loss.item()))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

            # if iter_count % 1000 == 0:
            #     acc = self.validate(task_model, val_dataloader)
            #     if acc > best_acc:
            #         best_acc = acc
            #         best_model = copy.deepcopy(task_model)
                
            #     print('current step: {} acc: {}'.format(iter_count, acc))
            #     print('best acc: ', best_acc)


        # if self.cuda:
        #     best_model = best_model.cuda()

        # final_accuracy = self.test(best_model)
        # return final_accuracy, vae, discriminator
        return vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             self.cuda)

        return querry_indices
                

    def validate(self, task_model, loader):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in loader:
            if self.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
