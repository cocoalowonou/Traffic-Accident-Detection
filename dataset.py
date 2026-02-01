import os
import numpy as np
from PIL import Image
import math
from tqdm import tqdm
from collections import defaultdict
from random import shuffle

import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

from torchvision import transforms
import sys
sys.path.append('../')
from utils import *

            
class Dataset_DoTA(data.Dataset):
    def __init__(self, frames_path, data_list, class_dict, meta_data, kernel_list, slide_window_size, stride_factor, pos_thresh, neg_thresh, nclasses ):
        super(Dataset_DoTA, self).__init__()

        self.slide_window_size = slide_window_size
        self.frames_path = frames_path
        self.nclasses = nclasses
        ########## Get Video meta data #######################
        
        segment_data = generate_segment_exp(meta_data, data_list, class_dict)
        self.meta_data = meta_data

        '''
            'v_id' : [num_frames, start, end, class]
            {'UbDCiQVuzPY_002416': [100, 35, 77, 11], '88LcRU7uEFE_001348': [40, 35, 40, 6]}
        '''

        ############### All the anchors ###############
        self.segment_list = []
        anc_len_lst = []
        anc_cen_lst = []
        for i in range(0, len(kernel_list)):
            kernel_len = kernel_list[i]
            anc_cen = np.arange(float((kernel_len) / 2.), float(
                slide_window_size + 1 - (kernel_len) / 2.), math.ceil(kernel_len/stride_factor))
            anc_len = np.full(anc_cen.shape, kernel_len)
            anc_len_lst.append(anc_len)
            anc_cen_lst.append(anc_cen)
        anc_len_all = np.hstack(anc_len_lst)
        anc_cen_all = np.hstack(anc_cen_lst)

        results = []
        for vid in data_list:
            vid_res = get_pos_neg(vid, segment_data[vid], slide_window_size, anc_len_all, anc_cen_all, pos_thresh, neg_thresh)
            results.append(vid_res)
        for vid_idx, res in enumerate(results):
            vid_prefix, total_frame, pos_seg, neg_seg = res
            proplist_per_seg = []
            class_prop = -1
            
            for proposal in pos_seg: #Each pos_seg has list of proposals 
                class_prop = proposal[-1]
                proplist_per_seg.append(proposal[:-1])
            self.segment_list.append((vid_prefix, proplist_per_seg, class_prop, neg_seg, total_frame))
            
    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, index):
        
        video_prefix, pos_seg, class_id, neg_seg, total_frame = self.segment_list[index]        
        image_out_list = []
        img_path = os.path.join(self.frames_path, video_prefix, 'images')
        img_path_list = sorted(os.listdir(img_path))
        for img_idx in range(total_frame):

            img = img_path_list[img_idx]
            image_out_list.append(self._read_img_and_transform(os.path.join(img_path, img)))
        
        image_out_list = torch.stack(image_out_list)
        
        mask = torch.ones(self.slide_window_size, self.nclasses, dtype=torch.float)
        
        return (pos_seg, class_id, neg_seg, image_out_list, mask)


    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((192,192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

    def _read_img_and_transform(self, img_path):
        return self.transform(Image.open(img_path).convert('RGB'))

def dota_collate_fn(batch_lst):

    sample_each = 10
    batch_size = len(batch_lst)
    input_batch, mask_batch = [], []
    tempo_proposal_pos = torch.FloatTensor(np.zeros((batch_size, sample_each, 4)))
    tempo_proposal_neg = torch.FloatTensor(np.zeros((batch_size, sample_each, 2)))
    gt_class_batch = torch.empty(batch_size, sample_each)
    for batch_indx, (pos_seg, gt_class, neg_seg, input_feat, mask) in enumerate(batch_lst):
        input_batch.append(input_feat)
        pos_seg_tensor = torch.FloatTensor(pos_seg)
        gt_class_batch[batch_indx].fill_(gt_class)
        mask_batch.append(mask)
        
        if len(pos_seg) >= sample_each:
            perm_idx = torch.randperm(len(pos_seg))
            tempo_proposal_pos[batch_indx,:,:] = pos_seg_tensor[perm_idx[:sample_each]]
        else:
            tempo_proposal_pos[batch_indx,:len(pos_seg),:] = pos_seg_tensor
            idx = torch.multinomial(torch.ones(len(pos_seg)), sample_each-len(pos_seg), True)
            tempo_proposal_pos[batch_indx,len(pos_seg):,:] = pos_seg_tensor[idx]

        neg_seg_tensor = torch.FloatTensor(neg_seg)
        
        if len(neg_seg) >= sample_each:
            perm_idx = torch.randperm(len(neg_seg))
            tempo_proposal_neg[batch_indx, :, :] = neg_seg_tensor[perm_idx[:sample_each]]
        else:
            tempo_proposal_neg[batch_indx, :len(neg_seg), :] = neg_seg_tensor
            idx = torch.multinomial(torch.ones(len(neg_seg)), sample_each - len(neg_seg),True)
            tempo_proposal_neg[batch_indx, len(neg_seg):, :] = neg_seg_tensor[idx]
    
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value = 0)
    gt_class_batch = gt_class_batch.view(-1)
    mask_batch = pad_sequence(mask_batch, batch_first=True, padding_value=0)
    
    return (input_batch, mask_batch, tempo_proposal_pos, tempo_proposal_neg, gt_class_batch)

def get_pos_neg(vid, segment_vInfo, slide_window_size, anc_len_all, anc_cen_all, pos_thresh, neg_thresh, sampling_sec=1):
    
    window_start = 0
    window_end = slide_window_size

    window_start_t = window_start * sampling_sec
    window_end_t = window_end * sampling_sec

    total_frame = segment_vInfo[0]
    gt_start = segment_vInfo[1] / sampling_sec
    gt_end = segment_vInfo[2] / sampling_sec
    
    neg_overlap = [0] * anc_len_all.shape[0]
    pos_seg = []
    for j in range(anc_len_all.shape[0]):
    
        if gt_start > gt_end:           # None Sense condition ...
            gt_start, gt_end = gt_end, gt_start
        if anc_cen_all[j] + anc_len_all[j] / 2. <= total_frame:
            if window_start_t <= segment_vInfo[1] and window_end_t + sampling_sec * 2 >= segment_vInfo[2]:
                overlap = segment_iou(np.array([gt_start, gt_end]), np.array([[
                    anc_cen_all[j] - anc_len_all[j] / 2.,
                    anc_cen_all[j] + anc_len_all[j] / 2.]]))
                neg_overlap[j] = max(overlap, neg_overlap[j])

                if overlap >= pos_thresh:
                    len_offset = math.log((gt_end - gt_start) / anc_len_all[j])
                    cen_offset = ((gt_end + gt_start) / 2. - anc_cen_all[j]) / anc_len_all[j]
                    
                    pos_seg.append([j, overlap, len_offset, cen_offset, segment_vInfo[3]])
                    
    neg_seg = []
    for oi, overlap in enumerate(neg_overlap):
        if overlap < neg_thresh:
            neg_seg.append((oi, overlap))

    return vid, total_frame, pos_seg, neg_seg
    

##################### Test ##########################

class Dataset_DoTA_Test(data.Dataset):
    def __init__(self, frames_path, data_list, meta_data, slide_window_size, nclasses ):
        super(Dataset_DoTA_Test, self).__init__()

        self.slide_window_size = slide_window_size
        self.frames_path = frames_path
        self.nclasses = nclasses
        ########## Get Video meta data #######################
        
        self.data_list = data_list
        self.meta_data = meta_data

        ############### All the anchors ###############
            
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        video_prefix = self.data_list[index]
        total_frame = self.meta_data[video_prefix]['num_frames']     
        image_out_list = []
        img_path = os.path.join(self.frames_path, video_prefix, 'images')
        img_path_list = sorted(os.listdir(img_path))
        for img_idx in range(total_frame):

            img = img_path_list[img_idx]
            image_out_list.append(self._read_img_and_transform(os.path.join(img_path, img)))
        
        image_out_list = torch.stack(image_out_list)
        
        mask = torch.ones(self.slide_window_size, self.nclasses, dtype=torch.float)
        
        return (image_out_list, mask, video_prefix, total_frame)


    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((192,192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

    def _read_img_and_transform(self, img_path):
        return self.transform(Image.open(img_path).convert('RGB'))
    

    
#################### Inference #################


class Dataset_DoTA_Inf(data.Dataset):
    def __init__(self, frames_path, opts):
        super(Dataset_DoTA_Inf, self).__init__()
        self.frames_path = frames_path
        self.opts = opts
        ############### All the anchors ###############
            
    def __len__(self):
        return 1 # Always return 1 because the dataset load 1 video

    def __getitem__(self, index):
     
        image_out_list = []
        img_path_list = sorted(os.listdir(self.frames_path))
        total_frame = len(img_path_list)
        for img_idx in range(total_frame):

            img = img_path_list[img_idx]
            image_out_list.append(self._read_img_and_transform(os.path.join(self.frames_path, img)))
        
        image_out_list = torch.stack(image_out_list)
        
        mask = torch.ones(self.opts.window_size, self.opts.n_classes, dtype=torch.float)
        
        return (image_out_list, mask, total_frame)


    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((192,192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

    def _read_img_and_transform(self, img_path):
        return self.transform(Image.open(img_path).convert('RGB'))
    
