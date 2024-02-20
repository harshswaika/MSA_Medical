# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import logging
import random
import warnings
import math

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from semilearn.algorithms import get_algorithm, name2alg
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg
from semilearn.core.utils import get_net_builder, get_logger, get_port, send_model_cuda, count_parameters, over_write_args_from_file, TBLog

from utils_train import create_model, test, test_thief, dist, load_victim_dataset, agree
import torch.multiprocessing

def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description='Semi-Supervised Learning (USB)')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb to plot and save curves')
    parser.add_argument('--use_aim', action='store_true', help='Use aim to plot and save curves')

    '''
    Training Configuration of FixMatch
    '''
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=20,
                        help='total number of training iterations')
    parser.add_argument('--num_warmup_iter', type=int, default=0,
                        help='cosine linear warmup iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10,
                        help='evaluation frequency')
    parser.add_argument('--num_log_iter', type=int, default=5,
                        help='logging frequencu')
    # parser.add_argument('-nl', '--num_labels', type=int, default=400)
    parser.add_argument('-bsz', '--batch_size', type=int, default=8)
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=1.0, help='layer-wise learning rate decay, default to 1.0 which means no layer decay')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--use_pretrain', default=False, type=str2bool)
    parser.add_argument('--pretrain_path', default='', type=str)

    '''
    Algorithms Configurations
    '''  

    ## core algorithm setting
    parser.add_argument('-alg', '--algorithm', type=str, default='fixmatch', help='ssl algorithm')
    parser.add_argument('--use_cat', type=str2bool, default=True, help='use cat operation in algorithms')
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip_grad', type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument('-imb_alg', '--imb_algorithm', type=str, default=None, help='imbalance ssl algorithm')

    '''
    Data Configurations
    '''

    ## standard setting configurations
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--include_lb_to_ulb', type=str2bool, default='True', help='flag of including labeled data into unlabeled data, default to True')

    ## imbalanced setting arguments
    parser.add_argument('--lb_imb_ratio', type=int, default=1, help="imbalance ratio of labeled data, default to 1")
    parser.add_argument('--ulb_imb_ratio', type=int, default=1, help="imbalance ratio of unlabeled data, default to 1")
    parser.add_argument('--ulb_num_labels', type=int, default=None, help="number of labels for unlabeled data, used for determining the maximum number of labels in imbalanced setting")

    ## cv dataset arguments
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=float, default=0.875)

    ## nlp dataset arguments 
    parser.add_argument('--max_length', type=int, default=512)

    ## speech dataset algorithms
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--c', type=str, default='')

    # add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)

    # add imbalanced algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if args.imb_algorithm is not None:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args



def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''

    assert args.num_train_iter % args.epoch == 0, \
        f"# total training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading paths are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    if args.gpu == 'None':
        args.gpu = None
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.distributed = False
    args.device = torch.device('cuda')
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''
    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")
    
    # Load victim dataset
    logger.info('\n')
    test_dataset = load_victim_dataset(args)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=128,
        num_workers=args.num_workers, 
        # pin_memory=True
        )

    # Load victim model
    victim_model = create_model(args.victim_arch, args.num_classes)
    state_dict = torch.load(args.victim_model_path)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    victim_model.load_state_dict(state_dict, strict=True)
    victim_model.cuda()

    victim_acc, spec, sens = test(args, test_loader, victim_model, 0)
    logger.info(f"Victim model accuracy = {victim_acc:.4f}, spec = {spec:.2f}, sens = {sens:.2f}")

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the synchronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Load labeled and val set from supervised model
    print('\n')
    logger.info('Loading thief dataset ...')
    cudnn.benchmark = True
    if args.rank != 0 and args.distributed:
        torch.distributed.barrier()

    if args.load_labeled_set is True:
        labeled_set = np.load(args.labeled_set_path)
        val_set = np.load(args.val_set_path)
        print('Loaded supervised labeled set of length ', len(labeled_set))

        if args.expand_labels or args.num_labels < args.batch_size:
            num_expand_x = math.ceil(
                args.batch_size * args.eval_step / args.num_labels)
            labeled_set = np.hstack([labeled_set for _ in range(num_expand_x)])
            np.random.shuffle(labeled_set)
            print('Labeled set expanded to length ', len(labeled_set), ', unique indices ', len(set(labeled_set)))

    else:
        labeled_set = None
        val_set = None

    # Initialize network for training
    logger.info('\nInitializing thief model ...')
    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders will be set in algorithms
    if args.imb_algorithm is not None:
        model = get_imb_algorithm(args, _net_builder, tb_log, logger, 
                                victim_model=victim_model, labeled_set=labeled_set, 
                                val_set=val_set, test_loader=test_loader)
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger,  
                              victim_model=victim_model, labeled_set=labeled_set, 
                              val_set=val_set, test_loader=test_loader)
    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # Load pretrained imagenet model for coldstart
    if args.use_pretrain == True:
        print("Load pretrained model for initializing the thief from ", args.pretrained_dir)
        
        # initialize from an imagenet pretrained model: cold start
        thief_state = model.model.state_dict()
        pretrained_state = torch.load(args.pretrained_dir)
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        imagenet_pretrained_state_common = {}
        for k, v in pretrained_state.items():
            if k in thief_state and v.size() == thief_state[k].size():
                imagenet_pretrained_state_common[k] = v
            elif 'backbone.'+k in thief_state and v.size() == thief_state['backbone.'+k].size():
                imagenet_pretrained_state_common['backbone.'+k] = v
            # remove 'module.' from pretrained state dict
            if k[7:] in thief_state and v.size() == thief_state[k[7:]].size():
                imagenet_pretrained_state_common[k[7:]] = v
    
    print('Common keys pretrained model: ', len(imagenet_pretrained_state_common.keys()))
    thief_state.update(imagenet_pretrained_state_common)
    model.model.load_state_dict(thief_state, strict=True)
    model.ema_model.load_state_dict(thief_state, strict=True)

    # Load anchor model
    print('Load anchor model', args.warmstart_dir)
    thief_state = model.model.state_dict()
    print('thief state: ', list(thief_state.keys())[:5])
    pretrained_state = torch.load(args.warmstart_dir)['state_dict']
    print('pretrained state: ', list(pretrained_state.keys())[:5])
    supervised_pretrained_state_common = {}
    for k, v in pretrained_state.items():
        if k in thief_state and v.size() == thief_state[k].size():
            supervised_pretrained_state_common[k] = v
        elif 'backbone.'+k in thief_state and v.size() == thief_state['backbone.'+k].size():
            supervised_pretrained_state_common['backbone.'+k] = v
        # remove 'module.' from pretrained state dict
        elif k[7:] in thief_state and v.size() == thief_state[k[7:]].size():
            supervised_pretrained_state_common[k[7:]] = v
        # remove 'base_model.' from pretrained state dict
        elif k[11:] in thief_state and v.size() == thief_state[k[11:]].size():
            supervised_pretrained_state_common[k[11:]] = v
        else:
            print('key not found', k)


    assert(len(thief_state.keys()) == len(supervised_pretrained_state_common.keys()))
    print('Common keys anchor model: ', len(thief_state.keys()))
    thief_state.update(supervised_pretrained_state_common)

    if args.algorithm in ['selfkd', 'selfkdaugment', 'selfkdcontrastive', 'pseudolabel_with_anchor']:
        model.anchor_init(thief_state)
        # init_acc, spec, sens = test(args, test_loader, model.anchor_model, 0)
        # init_agr = agree(victim_model, model.anchor_model, test_loader)
        # print(f'Anchor model on target dataset: acc = {init_acc:.4f}, agreement = {init_agr:.4f}')
        acc, agr, spec, sens = test_thief(model, model.anchor_model, victim_model, test_loader, ema=False)
        logger.info(f'Anchor model on target dataset: acc = {acc:.4f}, spec = {spec:.2f}, sens = {sens:.2f}')

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")
    
    # _, acc = test2(args, model.loader_dict['train_lb'], victim_model, 0)
    # print("Checking label data accuracy: ", acc)

    logger.info('Validation set distribution: ')
    val_dist = dist(val_set, model.loader_dict['eval'])
    logger.info(val_dist)

    logger.info('Labeled set distribution: ')
    label_dist = dist(labeled_set, model.loader_dict['train_lb'])
    logger.info(label_dist)

    # if args.algorithm in ['comatch', 'comatch2']:
    #     model.compute_adjustment(args.la, tro=args.tro)
    #     model.num_weak_augs = args.num_weak_augs

    # if args.algorithm == 'remixmatch':
    #     model.compute_label_dist(True)

    # Calculate initial accuracy and agreement
    acc, agr, spec, sens = test_thief(model, model.model, victim_model, test_loader, ema=False)
    logger.info(f'Initial model on target dataset (without EMA): acc = {acc:.4f}, agreement = {agr:.4f}, spec = {spec:.2f}, sens = {sens:.2f}')

    # if args.algorithm in ['comatchkd', 'fixmatchkd', 'selfkd', 'selfkdaugment', 'selfkdcontrastive', 'pseudolabel_with_anchor']:
    #     init_acc, spec, sens = test(args, test_loader, model.anchor_model, 0)
    #     init_agr = agree(victim_model, model.anchor_model, test_loader)
    #     logger.info(f'Anchor model on target dataset: acc = {init_acc:.4f}, agreement = {init_agr:.4f}, spec = {spec:.2f}, sens = {sens:.2f}')

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))    
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    if hasattr(model, 'warmup'):
        logger.info(("Warmup stage"))
        model.warmup()

    # START TRAINING of FixMatch
    logger.info("\nModel training")
    model.train()

    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    if hasattr(model, 'finetune'):
        logger.info("Finetune stage")
        model.finetune()
    
    acc, agr, spec, sens = test_thief(model, model.model, victim_model, test_loader, ema=False)
    logger.info(f'Best model on target dataset (without EMA): acc = {acc:.4f}, agreement = {agr:.4f}, spec = {spec:.2f}, sens = {sens:.2f}')
    acc, agr, spec, sens = test_thief(model, model.model, victim_model, test_loader, ema=True)
    logger.info(f'Best model on target dataset (with EMA): acc = {acc:.4f}, agreement = {agr:.4f}, spec = {spec:.2f}, sens = {sens:.2f}')

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)
