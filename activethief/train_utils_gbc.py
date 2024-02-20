import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.autograd import Variable


from torch.autograd import Variable


def testz(model, dataloader, no_roi=True, verbose=True):
    
    model.eval()
    y_true, y_pred = [], []
    for i, (inp, target, fname) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.cuda())
            target_var = torch.autograd.Variable(target)

            if len(input_var.shape) == 5:
                images = input_var.squeeze(0)
                outputs =  model(images)
                _, pred = torch.max(outputs, dim=1)
                pred_label = torch.max(pred)
                pred_label = pred_label.unsqueeze(0)
                y_true.append([target_var.tolist()][0][0])
                y_pred.append([pred_label.tolist()])

            else:
                outputs = model(input_var)
                _, pred_label = torch.max(outputs, dim=1)
                y_pred.append(pred_label.tolist()) 
                # if soft labels, extract hard label from it
                if len(target_var.shape) > 1:
                    target_var = target_var.argmax(axis=1)
                y_true.append(target_var.tolist())

    y_pred = np.concatenate(y_pred, 0)
    y_true = np.concatenate(y_true, 0)

    # print('y_pred: ', y_pred)
    # print('y_true: ', y_true)

    acc = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
    sens = cfm[2][2]/np.sum(cfm[2])
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    if verbose == True:
        print('specificity = {}/{}'.format(cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1], np.sum(cfm[0]) + np.sum(cfm[1])))
        print('sensitivity = {}/{}'.format(cfm[2][2], np.sum(cfm[2])))
    
    return acc, f1, spec, sens


def agree(model1, model2, test_loader):
    c=0
    l=0
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images, targets = images.float().cuda(), targets.cuda()
            if len(images.shape) == 5:
                images = images.squeeze(0)
                
                outputs =  model1(images)
                _, pred = torch.max(outputs, dim=1)
                x1 = torch.max(pred)
                
                outputs =  model2(images)
                _, pred = torch.max(outputs, dim=1)
                x2 = torch.max(pred)

                n = images.shape[0]
                c+=n-int((torch.count_nonzero(x1-x2)).detach().cpu())
                l+=n
            else:
                n=images.shape[0]
                x1=model1(images).argmax(axis=-1,keepdims=False)
                x2=model2(images).argmax(axis=-1,keepdims=False)
                c+=n-int((torch.count_nonzero(x1-x2)).detach().cpu())
                l+=n
            # print(c, l)
    # print('Agreement between Copy and source model is ', c/l)
    return c / l


def dist(indices, dataloader):
    "Return label distribution of selected samples" 
    # create dataloader from dataset
    # dl=DataLoader(dz, batch_size=1, sampler=SubsetRandomSampler(indices), pin_memory=False)
    dl = dataloader
    d = {}
    print('Number of samples ', len(indices))
    labels = []
    with torch.no_grad():
        for data in (dl):
            label = data[1]
            # print(data[1].shape)
            # if soft label, extract hard label using argmax
            if len(label.shape) > 1:
                label = label.argmax(axis=1)
            labels.extend(label.cpu().detach().numpy())
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        d[int(lbl)] = 0
    for label in labels:
        d[int(label)]+=1
    return d


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_with_validation(model, criterion, optimizers, scheduler, dataloaders, num_epochs, trial, cycle, out_dir, display_every = 5, early_stop_tolerance=100, la=None):
    print('>> Train a Model.')

    exit = False
    curr_loss = None
    best_f1 = None
    no_improvement = 0

    for epoch in tqdm(range(num_epochs), leave=False):

        t_loss = train_epoch(model, criterion, optimizers, dataloaders, la=la)
        scheduler.step()
        # print(f'epoch {epoch}, loss = {t_loss}')

        if (epoch+1)%2==0:
            val_acc, val_f1, spec, sens = testz(model, dataloaders['val'], verbose=False)
            # test_acc, test_f1 = testz(model, dataloaders['test'])

            if best_f1 is None or val_f1 > best_f1 :
                best_f1 = val_f1
                torch.save({
                    'trial': trial + 1,
                    'cycle': cycle + 1, 
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    },
                    f'{out_dir}/trial_{trial+1}_cycle_{cycle+1}_best.pth')

                # print(f"Epoch {epoch+1} Model saved in path: {os.path.join(out_dir, f'trial_{trial+1}_best.pth')}")
                no_improvement = 0
            else:
                no_improvement += 1
                
                if (no_improvement % early_stop_tolerance) == 0:
                    # if t_loss > 1.5:
                    #     no_improvement = 0
                    #     print('loss = ', t_loss)
                    # else:
                    exit = True

        # Display progress
        if (epoch+1) % display_every == 0:
            train_acc, train_f1, spec1, sens1 = testz(model, dataloaders['train'], verbose=False)
            test_acc, test_f1, spec2, sens2 = testz(model, dataloaders['test'], verbose=False)
            print(f"Epoch {epoch+1}: Train acc/f1 = {train_acc:.4f} / {train_f1:.4f} / {spec1:.4f} / {sens1:.4f} \n\
                Val acc/f1/spec/sens = {val_acc:.4f} / {val_f1:.4f} / {spec:.4f} / {sens:.4f}\n\
                Test acc/f1/spec/sens = {test_acc:.4f} / {test_f1:.4f} / {spec2:.4f} / {sens2:.4f}")

        if exit:
            print(f"Number of epochs processed: {epoch+1} in cycle {cycle+1}") 
            break

    train_acc, train_f1, spec1, sens1 = testz(model, dataloaders['train'])
    val_acc, val_f1, spec2, sens2 = testz(model, dataloaders['val'])
    test_acc, test_f1, spec3, sens3 = testz(model, dataloaders['test'])

    print(f"Trial {trial+1}, Cycle {cycle+1}")
    print(f"Train acc/f1/spec/sens = {train_acc:.4f} / {train_f1:.4f} / {spec1:.4f} / {sens1:.4f}")
    print(f"Val acc/f1/spec/sens = {val_acc:.4f} / {val_f1:.4f} / {spec2:.4f} / {sens2:.4f}")
    print(f"Test acc/f1/spec/sens = {test_acc:.4f} / {test_f1:.4f} / {spec3:.4f} / {sens3:.4f}")

    # Save the last model
    torch.save({'trial': trial + 1,
                'cycle': cycle + 1, 
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                },
                f'{out_dir}/trial_{trial+1}_cycle_{cycle+1}_last.pth')

    print('>> Finished.')


def train_epoch(model, criterion, optimizer, dataloaders, vis=None, plot_data=None, la=None):
    model.train()
    iters = 0
    total_loss = 0

    for data in dataloaders['train']:
        input = data[0].cuda()
        target = data[1].cuda()
        iters += 1

        optimizer.zero_grad()
        output = model(input)

        if la is not None:
            output = output + la
        target_loss = criterion(output, target)

        # scores = model(inputs)
        # target_loss = criterion(scores, labels)

        loss = torch.sum(target_loss) / target_loss.size(0)
        total_loss += torch.sum(target_loss)

        loss.backward()
        optimizer.step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append(loss.item())
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )

    mean_loss = total_loss / iters
    return mean_loss


def train_with_kd(model, criterion, optimizers, scheduler, dataloaders, num_epochs, trial, cycle, out_dir, display_every = 10, early_stop_tolerance=100, temp=1, alpha=1):
    print('>> Train a Model.')

    exit = False
    curr_loss = None
    best_f1 = None
    no_improvement = 0

    for epoch in tqdm(range(num_epochs), leave=False):
        # schedulers['backbone'].step()
        # schedulers['module'].step()

        t_loss = train_epoch_kd(model, criterion, optimizers, dataloaders, temp, alpha)
        scheduler.step()

        if (epoch+1)%2==0:
            val_acc, val_f1, spec, sens = testz(model, dataloaders['val'], verbose=False)
            # test_acc, test_f1 = testz(model, dataloaders['test'])

            if best_f1 is None or val_f1 > best_f1 :
                best_f1 = val_f1
                torch.save({
                    'trial': trial + 1,
                    'cycle': cycle + 1, 
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    },
                    f'{out_dir}/trial_{trial+1}_cycle_{cycle+1}_best.pth')

                # print(f"Epoch {epoch+1} Model saved in path: {os.path.join(out_dir, f'trial_{trial+1}_best.pth')}")
                no_improvement = 0
            else:
                no_improvement += 1
                
                if (no_improvement % early_stop_tolerance) == 0:
                    # if t_loss > 1.5:
                    #     no_improvement = 0
                    #     print('loss = ', t_loss)
                    # else:
                    exit = True

        # Display progress
        if (epoch+1) % display_every == 0:
            train_acc, train_f1, spec1, sens1 = testz(model, dataloaders['train'], verbose=False)
            test_acc, test_f1, spec2, sens2 = testz(model, dataloaders['test'], verbose=False)
            print(f"Epoch {epoch+1}: Train acc/f1 = {train_acc:.4f} / {train_f1:.4f} \n\
                Val acc/f1 = {val_acc:.4f} / {val_f1:.4f} \n\
                Test acc/f1 = {test_acc:.4f} / {test_f1:.4f}")

        if exit:
            print(f"Number of epochs processed: {epoch+1} in cycle {cycle+1}") 
            break

    train_acc, train_f1, spec1, sens1 = testz(model, dataloaders['train'])
    val_acc, val_f1, spec2, sens2 = testz(model, dataloaders['val'])
    test_acc, test_f1, spec3, sens3 = testz(model, dataloaders['test'])

    print(f"Trial {trial+1}, Cycle {cycle+1}")
    print(f"Train acc/f1/spec/sens = {train_acc:.4f} / {train_f1:.4f} / {spec1:.4f} / {sens1:.4f}")
    print(f"Val acc/f1/spec/sens = {val_acc:.4f} / {val_f1:.4f} / {spec2:.4f} / {sens2:.4f}")
    print(f"Test acc/f1/spec/sens = {test_acc:.4f} / {test_f1:.4f} / {spec3:.4f} / {sens3:.4f}")

    # Save the last model
    torch.save({'trial': trial + 1,
                'cycle': cycle + 1, 
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                },
                f'{out_dir}/trial_{trial+1}_cycle_{cycle+1}_last.pth')

    print('>> Finished.')
    
    
def train_epoch_kd(model, criterion, optimizer, dataloaders, temperature, alpha, vis=None, plot_data=None):
    model.train()
    iters = 0
    total_loss = 0

    for data in dataloaders['train']:
        input = data[0].cuda()
        target_soft = data[1].cuda().float()
        target_hard = target_soft.argmax(axis=1)
        # target_hard = data[1].cuda()
        # target_soft = data[2].cuda().float()
        iters += 1

        optimizer.zero_grad()
        output = model(input)
        
        # loss_ce = criterion(output, target_hard)
        # loss_ce = torch.sum(loss_ce) / loss_ce.size(0)
        # loss_kd = torch.nn.KLDivLoss(reduction='sum')(F.log_softmax(output / temperature, dim=1), F.softmax(target_soft, dim=1))
        # target_loss = loss_ce + loss_kd
        
        loss_ce = torch.nn.CrossEntropyLoss()(output, target_hard)
        # loss_kd = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output / temperature, dim=1), # output
        #                                                     F.softmax(target_soft / temperature, dim=1)) # target
        # target_loss = loss_kd * alpha * temperature * temperature + loss_ce * (1. - alpha)
        
        loss_kd = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), # output
                                                            target_soft) # target
        target_loss = loss_kd * alpha + loss_ce * (1. - alpha)

        loss = target_loss
        total_loss += torch.sum(target_loss)

        loss.backward()
        optimizer.step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append(loss.item())
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )

    mean_loss = total_loss / iters
    return mean_loss


def train_cutmix(model, criterion, optimizers, scheduler, dataloaders, num_epochs, trial, cycle, out_dir, beta, cutmix_prob, display_every = 10, early_stop_tolerance=100):
    print('>> Train a Model using CutMix.')
    
    exit = False
    curr_loss = None
    best_f1 = None
    no_improvement = 0

    for epoch in tqdm(range(num_epochs), leave=False):

        # if epoch < 10:
        #     train_epoch(model, criterion, optimizers, dataloaders)
        # else:    
        #     train_epoch_cutmix(model, criterion, optimizers, dataloaders, beta, cutmix_prob, vis, plot_data)

        t_loss = train_epoch_cutmix(model, criterion, optimizers, dataloaders, beta, cutmix_prob)
        scheduler.step()

        if (epoch+1) % 2 == 0:
            val_acc, val_f1 = testz(model, dataloaders['val'])
            
            if best_f1 is None or val_f1 > best_f1 :
                best_f1 = val_f1
                torch.save({
                    'trial': trial + 1,
                    'cycle': cycle + 1, 
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    },
                    f'{out_dir}/trial_{trial+1}_cycle_{cycle+1}_best.pth')

                # print(f"Epoch {epoch+1} Model saved in path: {os.path.join(out_dir, f'trial_{trial+1}_best.pth')}")
                no_improvement = 0
            else:
                no_improvement += 1
                
                if (no_improvement % early_stop_tolerance) == 0:
                   
                    exit = True
        
        # Display progress
        if (epoch+1) % display_every == 0:
            train_acc, train_f1 = testz(model, dataloaders['train'])
            test_acc, test_f1 = testz(model, dataloaders['test'])
            print(f"Epoch {epoch+1}: Train acc/f1 = {train_acc:.4f} / {train_f1:.4f} \n\
                Val acc/f1 = {val_acc:.4f} / {val_f1:.4f} \n\
                Test acc/f1 = {test_acc:.4f} / {test_f1:.4f}")

        if exit:
            print(f"Number of epochs processed: {epoch+1} in cycle {cycle+1}") 
            break

    train_acc, train_f1 = testz(model, dataloaders['train'])
    val_acc, val_f1 = testz(model, dataloaders['val'])
    test_acc, test_f1 = testz(model, dataloaders['test'])

    print(f"Trial {trial+1}, Cycle {cycle+1}")
    print(f"Train acc/f1 = {train_acc:.4f} / {train_f1:.4f}")
    print(f"Val acc/f1 = {val_acc:.4f} / {val_f1:.4f}")
    print(f"Test acc/f1 = {test_acc:.4f} / {test_f1:.4f}")

    # Save the last model
    torch.save({'trial': trial + 1,
                'cycle': cycle + 1, 
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                },
                f'{out_dir}/trial_{trial+1}_cycle_{cycle+1}_last.pth')

    print('>> Finished.')


def train_epoch_cutmix(model, criterion, optimizer, dataloaders, beta, cutmix_prob, vis=None, plot_data=None):
    model.train()
    # global iters
    iters = 0
    total_loss = 0

    # for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    for data in dataloaders['train']:
        input = data[0].cuda()
        target = data[1].cuda()
        iters += 1

        optimizer.zero_grad()

        # CutMix
        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # only for analysis, compute output on original input
            # output_simple = model(input)[0]
            # target_loss_simple = criterion(output_simple, target)

            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            
            # compute output
            output = model(input)
            target_loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            
            # if lam > 0.5:
            #     target_loss = criterion(output, target_a)
            # else:
            #     target_loss = criterion(output, target_b)
            # z = 1
        else:
            # compute output
            output = model(input)
            target_loss = criterion(output, target)

        # scores = model(inputs)
        # target_loss = criterion(scores, labels)

        loss = torch.sum(target_loss) / target_loss.size(0)
        total_loss += torch.sum(target_loss)

        loss.backward()
        optimizer.step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append(loss.item())
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )

    mean_loss = total_loss / iters
    return mean_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_mixup(model, criterion, optimizers, scheduler, dataloaders, num_epochs, alpha, vis, plot_data):
    print('>> Train a Model using Mixup.')
    # checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    for epoch in tqdm(range(num_epochs), leave=False):
        # schedulers['backbone'].step()
        # schedulers['module'].step()

        train_epoch_mixup(model, criterion, optimizers, dataloaders, alpha, vis, plot_data)
        scheduler.step()

        if (epoch+1) % 10 == 0:
            train_acc = test(model, dataloaders['train'])
            test_acc = test(model, dataloaders['test'])
            print(f'Train acc = {train_acc}, Test acc = {test_acc}')

        # Save a checkpoint
        if False and epoch % 10 == 0:
            acc = test(models, dataloaders, 'test')
            acc2= test(models, dataloaders, 'train')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
            print('Train Acc: {:.3f}'.format(acc2))
            agree(model,models['backbone'],dataloaders['test'])
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')


def train_epoch_mixup(model, criterion, optimizer, dataloaders, alpha, vis=None, plot_data=None):
    model.train()
    # global iters
    iters = 0

    # for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        targets = data[1].cuda()
        iters += 1

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, use_cuda=True)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        outputs = model(inputs)[0]
        target_loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # scores = model(inputs)
        # target_loss = criterion(scores, labels)

        loss = torch.sum(target_loss) / target_loss.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append(loss.item())
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )


def train_augmix(model, criterion, optimizers, scheduler, dataloaders, num_epochs, preprocess, vis, plot_data):
    print('>> Train a Model using Augmix.')
    # checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    for epoch in tqdm(range(num_epochs), leave=False):
        # schedulers['backbone'].step()
        # schedulers['module'].step()

        train_epoch_augmix(model, criterion, optimizers, dataloaders, preprocess, vis, plot_data)
        scheduler.step()

        if (epoch+1) % 10 == 0:
            train_acc = test(model, dataloaders['train'])
            test_acc = test(model, dataloaders['test'])
            print(f'Train acc = {train_acc}, Test acc = {test_acc}')

    print('>> Finished.')


def train_epoch_augmix(model, criterion, optimizer, dataloaders, preprocess, vis=None, plot_data=None, all_ops=False, mixture_width=3, mixture_depth=1, 
        aug_severity=3):
    model.train()
    # global iters
    iters = 0

    # for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    for data in dataloaders['train']:
        image= data[0].cuda()
        target = data[1].cuda()
        iters += 1

        aug_list = augmentations.augmentations
        if all_ops:
            aug_list = augmentations.augmentations_all

        ws = np.float32(np.random.dirichlet([1] * mixture_width))
        m = np.float32(np.random.beta(1, 1))

        # mix = torch.zeros_like(preprocess(image))
        for i in range(mixture_width):
            image_aug = image.copy()
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * preprocess(image_aug)
            # mix = ws[i] * image_aug

        mixed = (1 - m) * preprocess(image) + m * mix
        # mixed = (1 - m) * image + m * mix

        output = model(input)[0]
        target_loss = criterion(output, target)

        loss = torch.sum(target_loss) / target_loss.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append(loss.item())
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )

