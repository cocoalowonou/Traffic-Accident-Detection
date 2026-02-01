import os
import argparse
import numpy as np
import random
import json
import time
import itertools
# torch
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from sklearn.preprocessing import LabelEncoder
# misc

from opt import parse_opt
from dataset import Dataset_DoTA, dota_collate_fn
from models.TAD_main import ModelMain
import sys
sys.path.append('../')
from utils import *

torch.set_printoptions(profile="full")
from torch.utils.tensorboard import SummaryWriter

'''
Old data Vs New Model
'''
seed = 213
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
scaler = GradScaler()
start_from = ''
le = LabelEncoder()
opts = parse_opt()

def get_dataset():
    
    meta_data = json.load(open(opts.meta_data))
    train_X = read_txt(opts.train_X)
    test_X = read_txt(opts.test_X)
    print("Train : {} Test: {}".format(len(train_X), len(test_X)))
    train_X = train_X[:50]
    test_X = test_X[:50]
    classes_dict = generate_classes(meta_data)
    le.fit(list(classes_dict.keys()))
    print(classes_dict)
    
    train_set = Dataset_DoTA(opts.frames_path, train_X, classes_dict, meta_data, opts.kernel_list, opts.window_size, opts.stride_factor, opts.pos_thresh, opts.neg_thresh, opts.n_classes)
    valid_set = Dataset_DoTA(opts.frames_path, test_X, classes_dict, meta_data, opts.kernel_list, opts.window_size, opts.stride_factor, opts.pos_thresh, opts.neg_thresh, opts.n_classes)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)    
    train_loader = DataLoader(train_set,
                              batch_size=opts.batch_size,
                              num_workers=opts.num_workers,
                              shuffle=True,
                              collate_fn=dota_collate_fn,
                              pin_memory=True)

    valid_loader = DataLoader(valid_set,
                              batch_size=opts.batch_size,
                              num_workers=opts.num_workers,
                              collate_fn=dota_collate_fn,
                              pin_memory=True)
    
    return train_loader, valid_loader, train_sampler


def get_model(gpu):
    model = ModelMain(opts.d_model, opts.n_classes, opts.stride_factor, opts.kernel_list).to(gpu)
    
    if len(opts.pretrained_model) > 0:
        print("Initializing weights from {}".format(opts.pretrained_model))
        model.load_state_dict(torch.load(opts.pretrained_model,
                                              map_location=lambda storage, location: storage))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    return model


def train(gpu, world_size):
    rank = opts.nr * opts.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)
    print('loading dataset')
    train_loader, valid_loader, train_sampler = get_dataset()
    print("Training set : {} Testing set: {}".format(len(train_loader.dataset), len(valid_loader.dataset)))

    print('building model')
    model = get_model(gpu)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opts.learning_rate, weight_decay=1e-5, momentum=opts.alpha, nesterov=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=opts.reduce_factor,
                                               patience=opts.patience_epoch,
                                               verbose=True)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    reg_loss = torch.nn.SmoothL1Loss()
    
    best_loss = 0.0

    all_eval_losses = []
    all_cls_losses = []
    all_reg_losses = []
    all_sent_losses = []
    all_training_losses = []

    writer = SummaryWriter(opts.logs_path)
    for curr_epoch in range(opts.max_epoch):
        t_epoch_start = time.time()
        print('Epoch: {}'.format(curr_epoch))
        train_sampler.set_epoch(curr_epoch)
        epoch_loss, train_cls, train_reg, train_label, train_acc = train_epoch(curr_epoch, model, optimizer, train_loader, bce_loss, reg_loss, gpu)
        (valid_loss, val_cls_loss, val_reg_loss, val_sent_loss, val_acc) = valid_epoch(model, valid_loader, bce_loss, reg_loss, gpu)
        
        all_training_losses.append(epoch_loss)

        all_eval_losses.append(valid_loss)
        all_cls_losses.append(val_cls_loss)
        all_reg_losses.append(val_reg_loss)
        all_sent_losses.append(val_sent_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.module.state_dict(), os.path.join(opts.checkpoint_path, 'best_model.t7'))
            print('*'*5)
            print('Better validation loss {:.4f} found, save model'.format(valid_loss))

        scheduler.step(valid_loss)

        # validation/save checkpoint every a few epochs
        if curr_epoch%opts.save_every_epoch == 0 or curr_epoch == opts.max_epoch:
            
            torch.save({
            'epoch': curr_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(opts.checkpoint_path, 'model_epoch_{}.t7'.format(curr_epoch)))

        print('-'*80)
        print('Epoch {} summary'.format(curr_epoch))
        print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, valid_loss, time.time()-t_epoch_start
        ))
        print("Train_cls: {:.4f}, "
        "Train_reg: {:.4f}, Train_label: {:.4f}".format(train_cls, train_reg, train_label))

        print('val_cls: {:.4f}, '
              'val_reg: {:.4f}, val_label: {:.4f}, '.format(val_cls_loss, val_reg_loss, val_sent_loss))
        print('-'*80)
        
        writer.add_scalars('Loss', {
            'Train loss': epoch_loss,
            'Valid loss': valid_loss
        }, curr_epoch)

        writer.add_scalars('Binary Entropy Loss', {
            'Train loss': train_cls,
            'Valid loss': val_cls_loss
        }, curr_epoch)

        writer.add_scalars('Regression Loss', {
            'Train loss': train_reg,
            'Valid loss': val_reg_loss
        }, curr_epoch)

        writer.add_scalars('Classification Loss', {
            'Train loss': train_label,
            'Valid loss': val_sent_loss
        }, curr_epoch)

        writer.add_scalars('Top1 Accuracy', {
            'Train': train_acc,
            'Valid': val_acc
        }, curr_epoch)

    writer.close()


### Training the network ###
def train_epoch(epoch, model, optimizer, train_loader, bce_crt, reg_crt, gpu):
    model.train() # training mode
    train_loss = []
    train_cls_loss = []
    train_reg_loss = []
    train_sent_loss = []
    correct = 0
    total = 0
    nbatches = len(train_loader)
    t_iter_start = time.time()

    for train_iter, data in enumerate(train_loader):
        (img_batch, mask_batch, tempo_seg_pos, tempo_seg_neg, label_batch) = data
        img_batch = img_batch.cuda()
        mask_batch = mask_batch.cuda()
        tempo_seg_neg = tempo_seg_neg.cuda()
        tempo_seg_pos = tempo_seg_pos.cuda()
        label_batch = label_batch.cuda()

        t_model_start = time.time()

        optimizer.zero_grad()
        with autocast():
            
            (pred_score, gt_score,
            pred_offsets, gt_offsets,
            pred_class) = model(img_batch, mask_batch, tempo_seg_pos, tempo_seg_neg)
            cls_loss = bce_crt(pred_score, gt_score) * opts.cls_weight
            reg_loss = reg_crt(pred_offsets, gt_offsets) * opts.reg_weight
            
            lable_loss = F.cross_entropy(pred_class, label_batch.long()) * opts.label_weight

            total_loss = cls_loss + reg_loss + lable_loss

        _, predicted = torch.max(pred_class.data, 1)
        total += label_batch.size(0)
        correct += (predicted == label_batch).sum().item()
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        total_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), opts.grad_norm)

        scaler.step(optimizer)
        scaler.update()

        train_loss.append(total_loss.data.item())
        train_cls_loss.append(cls_loss.data.item())
        train_reg_loss.append(reg_loss.data.item())
        train_sent_loss.append(lable_loss.data.item())

        t_model_end = time.time()
        if train_iter%50 == 0:
            print('iter: [{}/{}], training total loss: {:.4f}, '
                'class: {:.4f}, '
                'reg: {:.4f}, label: {:.4f}, '
                'grad norm: {:.4f} '
                'data time: {:.4f}s, total time: {:.4f}s'.format(
                train_iter, nbatches, total_loss.data.item(), cls_loss.data.item(),
                reg_loss.data.item(), lable_loss.data.item(),
                total_grad_norm,
                t_model_start - t_iter_start,
                t_model_end - t_iter_start
            ), end='\r')

            t_iter_start = time.time()
    acc = 100 * correct // total
    print(f'Training Accuracy : {acc} %')

    return np.mean(train_loss), np.mean(train_cls_loss), np.mean(train_reg_loss), np.mean(train_sent_loss), acc


### Validation ##
def valid_epoch(model, loader, bce_crt, reg_crt, gpu):
    model.eval()
    valid_loss = []
    val_cls_loss = []
    val_reg_loss = []
    val_sent_loss = []
    correct = 0
    total = 0

    for iter, data in enumerate(loader):
        (img_batch, mask_batch, tempo_seg_pos, tempo_seg_neg, label_batch) = data
        with torch.no_grad():
            
            img_batch = img_batch.cuda()
            mask_batch = mask_batch.cuda()
            tempo_seg_neg = tempo_seg_neg.cuda()
            tempo_seg_pos = tempo_seg_pos.cuda()
            label_batch = label_batch.cuda()
            with autocast():
                (pred_score, gt_score,
                pred_offsets, gt_offsets,
                pred_class) = model(img_batch,mask_batch, tempo_seg_pos, tempo_seg_neg)

                cls_loss = bce_crt(pred_score, gt_score) * opts.cls_weight
                reg_loss = reg_crt(pred_offsets, gt_offsets) * opts.reg_weight

                label_loss = F.cross_entropy(pred_class, label_batch.long()) * opts.label_weight
                total_loss = cls_loss + reg_loss + label_loss

            _, predicted = torch.max(pred_class.data, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()
            
            valid_loss.append(total_loss.data.item())
            val_cls_loss.append(cls_loss.data.item())
            val_reg_loss.append(reg_loss.data.item())
            val_sent_loss.append(label_loss.data.item())
    acc = 100 * correct // total
    print(f'Validation Accuracy : {acc} %')

    return (np.mean(valid_loss), np.mean(val_cls_loss), np.mean(val_reg_loss), np.mean(val_sent_loss), acc)

def main():
    parser = argparse.ArgumentParser()
    world_size = opts.gpus * opts.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train, nprocs=opts.gpus, args=(world_size,))
if __name__ == '__main__':
    main()