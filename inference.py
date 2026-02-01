import os
import sys
import json
import numpy as np
import subprocess
from collections import defaultdict
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from models.model_main import ModelMain
from dataset import Dataset_DoTA_Inf
from opt_dense import parse_opt
sys.path.append('../')
from utils import *
torch.set_printoptions(profile="full")

le = LabelEncoder()

opts = parse_opt()

def setup_dataset():
    print('loading dataset')
    if not os.path.exists(f'{opts.output}'):
        os.makedirs(f'{opts.output}')
        subprocess.call(f"ffmpeg -i {opts.video} -r 10 {opts.output}/$img_%06d.jpg", shell=True)
    if not os.path.exists(f'{opts.results_path}'):
        os.makedirs(f'{opts.results_path}')
        
    meta_data = json.load(open(opts.meta_data))
    classes_dict = generate_classes(meta_data)
    le.fit(list(classes_dict.keys()))

    test_set = Dataset_DoTA_Inf(opts.output, opts)
    
    test_loader = DataLoader(test_set,
                              batch_size=1,
                              num_workers=opts.num_workers,
                              pin_memory=True)

    return test_loader


def test(model, loader):
    model.eval()
    densecap_result = defaultdict(list)
    prop_result = defaultdict(list)
    avg_prop_num = 0
    best_model = torch.load(opts.pretrained_model)
    model.load_state_dict(best_model['model_state_dict'])
    
    pred_out = defaultdict(list)
    with torch.no_grad():

        for iter, (img_batch, mask, vid_len) in enumerate(loader):
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

            for pred_start, pred_end, pred_s, sent in all_proposal_results[0]:
                
                ann_class = cat2labels(le, [sent])[0]
                pred_out[0].append(ann_class)
                densecap_result[opts.video].append(
                    {'ann_class':ann_class,
                    'segment':[pred_start,
                                pred_end],
                    'score':pred_s})

                prop_result[opts.video].append(
                    {'segment':[pred_start,
                                pred_end],
                    'score':pred_s})

            avg_prop_num += len(all_proposal_results[0])
    with open(f'./pred_out_testonly.json', 'w')as f:
        json.dump(pred_out, f)

    dense_cap_all = {'version':'VERSION 1.0', 'results':densecap_result,
                     'external_data':{'used':'true',
                      'details':'Anomaly Accident Type Classification Result with Temporal Learning with stride factor 50'}}
    
    with open(os.path.join(f'{opts.results_path}', 'recog_testing.json'), 'w') as f:
        json.dump(dense_cap_all, f)

    # write proposals to json file for evaluation (proposal)
    prop_all = {'version':'VERSION 1.0', 'results':prop_result,
                'external_data':{'used':'true',
                'details':'Anomaly Accident Proposal Generation Result with Temporal Learning with Stride facotr 50 '}}
    
    with open(os.path.join(f'{opts.results_path}', 'prop_testing.json'), 'w') as f:
        json.dump(prop_all, f)

if __name__ == '__main__':
    
    test_loader = setup_dataset()

    model = ModelMain(d_model=opts.d_model, n_class=opts.n_classes, stride_factor=opts.stride_factor, kernel_list=opts.kernel_list).cuda()
    model = torch.nn.DataParallel(model)

    # for epoch in range(max_epoch):
    test(model, test_loader)