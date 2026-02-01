import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from models.model_main import ModelMain
from dataset import Dataset_DoTA_Test
from eval.eval_detect import EvalDetect
from eval.eval_proposal import EvalProposal
from opt import parse_opt
sys.path.append('../')
from utils import *
torch.set_printoptions(profile="full")
from torch.utils.tensorboard import SummaryWriter

le = LabelEncoder()

opts = parse_opt()

def get_dataset():
    print('loading dataset')
    
    meta_data = json.load(open(opts.meta_data))
    test_X = read_txt(opts.test_X)
    
    classes_dict = generate_classes(meta_data)
    le.fit(list(classes_dict.keys()))
    test_set = Dataset_DoTA_Test(opts.frames_path, test_X, meta_data, opts.window_size, opts.n_classes )
    
    test_loader = DataLoader(test_set,
                              batch_size=1,
                              num_workers=opts.num_workers,
                              pin_memory=True)

    return test_loader, test_X

def eval_results(epoch, densecap_result, prop_result, test_file_list):
    if not os.path.exists(f'{opts.results_path}_{epoch}'):
        os.makedirs(f'{opts.results_path}_{epoch}')
    
    dense_cap_all = {'version':'VERSION 1.0', 'results':densecap_result,
                     'external_data':{'used':'true',
                      'details':'Anomaly Accident Type Classification Result with Temporal Learning with stride factor 50'}}
    
    with open(os.path.join(f'{opts.results_path}_{epoch}', 'recog_testing.json'), 'w') as f:
        json.dump(dense_cap_all, f)

    # write proposals to json file for evaluation (proposal)
    prop_all = {'version':'VERSION 1.0', 'results':prop_result,
                'external_data':{'used':'true',
                'details':'Anomaly Accident Proposal Generation Result with Temporal Learning with Stride facotr 50 '}}
    
    with open(os.path.join(f'{opts.results_path}_{epoch}', 'prop_testing.json'), 'w') as f:
        json.dump(prop_all, f)

    anet_proposal = EvalProposal(opts.meta_data, test_file_list, 
                                 os.path.join(f'{opts.results_path}_{epoch}', 'prop_testing.json'),
                                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                                 max_avg_nr_proposals=100,
                                 subset="testing", verbose=True, check_status=True)

    auc = anet_proposal.evaluate()

    anet_detection = EvalDetect(opts.meta_data, test_file_list, os.path.join(f'{opts.results_path}_{epoch}', 'recog_testing.json'), tiou_thresholds=np.linspace(0.5, 0.95, 10))
    mAP = anet_detection.evaluate()


def test(epoch, model, loader, test_list):
    model.eval()
    densecap_result = defaultdict(list)
    prop_result = defaultdict(list)
    avg_prop_num = 0
    best_model = torch.load(opts.pretrained_model)
    model.load_state_dict(best_model['model_state_dict'])
    meta_data = json.load(open(opts.meta_data))
    pred_out = defaultdict(list)
    gt_out = defaultdict(list)
    with torch.no_grad():

        for iter, (img_batch, mask, vid_prefix, vid_len) in enumerate(loader):
            img_batch = img_batch.cuda()
            mask = mask.cuda()
            all_proposal_results = model.module.inference(img_batch,
                                        mask,
                                        vid_len,
                                        opts.min_prop,
                                        opts.max_prop,
                                        opts.min_prop_before_nms,
                                        opts.pos_thresh
                                        )
            print("Video prefix > ", vid_prefix, " len > ", len(vid_prefix))
            for b in range(len(vid_prefix)):
                vid = vid_prefix[b]
                print('Write results for video: {}'.format(vid))
                for pred_start, pred_end, pred_s, sent in all_proposal_results[b]:
                    
                    ann_class = cat2labels(le, [sent])[0]
                    pred_out[vid].append(ann_class)
                    gt_out[vid].append(meta_data[vid]['anomaly_class'])
                    densecap_result[vid].append(
                        {'ann_class':ann_class,
                        'segment':[pred_start,
                                    pred_end],
                        'score':pred_s})

                    prop_result[vid].append(
                        {'segment':[pred_start,
                                    pred_end],
                        'score':pred_s})

                avg_prop_num += len(all_proposal_results[b])
    with open(f'./pred_out_testonly_{epoch}.json', 'w')as f:
        json.dump(pred_out, f)
    with open(f'./gt_out_testonly_{epoch}.json', 'w')as f:
        json.dump(gt_out, f)
    print("average proposal number: {}".format(avg_prop_num/len(loader.dataset)))

    eval_results(epoch, densecap_result, prop_result, test_list)
    
def test_dota(gpu, world_size):
    rank = opts.nr * opts.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)
    test_loader, test_list = get_dataset()
    ################### Evaluate Only #####################
    # eval_results(test_list)
    ################### Testing + Evaluate ######################
    
    model = ModelMain(d_model=opts.d_model, n_class=opts.n_classes, stride_factor=opts.stride_factor, kernel_list=opts.kernel_list).to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # for epoch in range(max_epoch):
    test(49, model, test_loader, test_list)

def main():
    
    world_size = opts.gpus * opts.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(test_dota, nprocs=opts.gpus, args=(world_size,))
if __name__ == '__main__':
    main()