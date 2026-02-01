
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import sys
sys.path.append('../')
from models.Tadtempo import Tadtempo
from models.TAD_transformer import TransformerClassifer
from utils import *
torch.set_printoptions(profile="full")
class SegmentProposal(nn.Module):
    def __init__(self, d_model, stride_factor, kernel_list, soi_size = 50):
        super().__init__()
        '''
        Input: video_features from temporal convolutional learnt network
        Output: Anchors proposals 

        '''
        self.kernel_list = kernel_list
        self.stride_factor = stride_factor
        self.soi_size = soi_size

        self.regress_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, 2, 1, bias=False)
        )

        self.bn_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, 1, 1, bias=False)
        )

        self.prop_out = nn.ModuleList(
            [nn.Sequential(
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model,
                          kernel_list[i],
                          stride=math.ceil(kernel_list[i]/stride_factor),
                          groups=d_model,
                          bias=False),
                nn.BatchNorm1d(d_model),
                )
                for i in range(len(kernel_list))
            ])
    def forward(self, vis_feat, s_pos, s_neg ):
        B, ch, T = vis_feat.size()
        dtype = vis_feat.data.type()
        prop_lst = []
        for i, kernel in enumerate(self.prop_out):
            kernel_size = self.kernel_list[i]
            if kernel_size <= vis_feat.size(-1):
                pred_o = kernel(vis_feat)
                action_ness = self.bn_conv(pred_o)
                action_location = self.regress_conv(pred_o)
                
                pred_out = torch.cat((action_ness, action_location), 1)
                
                anchor_c = torch.FloatTensor(np.arange(
                    float(kernel_size)/2.0,
                    float(T+1-kernel_size/2.0),
                    math.ceil(kernel_size/self.stride_factor)
                )).type(dtype)

                if anchor_c.size(0) != pred_o.size(-1):
                    raise Exception("size mismatch!")

                anchor_c = anchor_c.expand(B, 1, anchor_c.size(0))
                anchor_l = torch.FloatTensor(anchor_c.size()).fill_(kernel_size).type(dtype)

                pred_final = torch.cat((pred_out, anchor_l, anchor_c), 1)
                prop_lst.append(pred_final)
            else:
                print('skipping kernel sizes greater than {}'.format(
                    self.kernel_list[i]))
                break

        prop_all = torch.cat(prop_lst, 2) # ( Batch, 5 , n_prop_outputs)

        if B != s_pos.size(0) or B != s_neg.size(0):
            raise Exception('feature and ground-truth segment do not match!')

        pred_len = prop_all[:, 3, :] * torch.exp(prop_all[:, 1, :]) #(Batch, n_prop_outputs)
        pred_cen = prop_all[:, 4, :] + prop_all[:, 3, :] * prop_all[:, 2, :] #(Batch, n_prop_outputs)

        sample_each = 10
        pred_score = torch.FloatTensor(np.zeros((sample_each*B, 2))).type(dtype) 
        gt_score = torch.FloatTensor(np.zeros((sample_each*B, 2))).type(dtype)
        pred_offsets = torch.FloatTensor(np.zeros((sample_each*B,2))).type(dtype)
        gt_offsets = torch.FloatTensor(np.zeros((sample_each*B,2))).type(dtype)
        
        soi = torch.FloatTensor(np.zeros((B, sample_each, T, 1))).type(dtype)

        for b in range(B):
            pos_anchor = s_pos[b] #ground truth
            neg_anchor = s_neg[b] #background
            if pos_anchor.size(0) != sample_each or neg_anchor.size(0) != sample_each:
                raise Exception("# of positive or negative samples does not match")

            for i in range(len(pos_anchor)):
                # sample pos anchors
                pos_sam = pos_anchor[i].data # j, overlap_score, len_offset, cen_offset, anc_len
                pos_sam_ind = int(pos_sam[0])

                pred_score[b*sample_each+i, 0] = prop_all[b, 0, pos_sam_ind]
                gt_score[b*sample_each+i, 0] = 1

                pred_offsets[b*sample_each+i] = prop_all[b, 1:3, pos_sam_ind]
                gt_offsets[b*sample_each+i] = pos_sam[2:4]

                # sample neg anchors
                neg_sam = neg_anchor[i].data
                neg_sam_ind = int(neg_sam[0])
                pred_score[b*sample_each+i, 1] = prop_all[b, 0, neg_sam_ind]
                gt_score[b*sample_each+i, 1] = 0

                crt_pred_cen = pred_cen[b, pos_sam_ind]
                crt_pred_len = pred_len[b, pos_sam_ind]

                pred_start_w = crt_pred_cen - crt_pred_len / 2.0
                pred_end_w = crt_pred_cen + crt_pred_len/2.0

                prop_start = math.floor(max(0, min(T-1, pred_start_w.data.item())))
                prop_end = math.ceil(max(1, min(T, pred_end_w.data.item())))

                soi[b, i, prop_start : prop_end, :] = 1
                
        return pred_score, gt_score, pred_offsets, gt_offsets, soi


    def inference(self, vis_feat, actual_frame_length, min_prop_num, max_prop_num, min_prop_num_before_nms, pos_thresh):
        B, ch, T = vis_feat.size()
        dtype = vis_feat.data.type()

        prop_lst = []
        for i, kernel in enumerate(self.prop_out):

            kernel_size = self.kernel_list[i]
            if kernel_size <= actual_frame_length[0]: # no need to use larger kernel size in this case, batch size is only 1
                pred_o = kernel(vis_feat)
                action_ness = self.bn_conv(pred_o)
                action_location = self.regress_conv(pred_o)
                
                pred_out = torch.cat((action_ness, action_location), 1)
                
                anchor_c = torch.FloatTensor(np.arange(
                    float(kernel_size)/2.0,
                    float(T+1-kernel_size/2.0),
                    math.ceil(kernel_size/self.stride_factor)
                )).type(dtype)

                if anchor_c.size(0) != pred_o.size(-1):
                    raise Exception("size mismatch!")

                anchor_c = anchor_c.expand(B, 1, anchor_c.size(0))
                anchor_l = torch.FloatTensor(anchor_c.size()).fill_(kernel_size).type(dtype)

                pred_final = torch.cat((pred_out, anchor_l, anchor_c), 1)
                prop_lst.append(pred_final)
            else:
                print('skipping kernel sizes greater than {}'.format(
                    self.kernel_list[i]))
                break

        prop_all = torch.cat(prop_lst, 2) # ( Batch, 5 , n_prop_outputs)
        prop_all[:,0,:] = F.sigmoid(prop_all[:,0,:])

        pred_len = prop_all[:, 3, :] * torch.exp(prop_all[:, 1, :])
        pred_cen = prop_all[:, 4, :] + prop_all[:, 3, :] * prop_all[:, 2, :]
        nms_thresh_set = np.arange(0.9, 0.95, 0.05).tolist()
        
        for b in range(B):
            soi_result = []
            crt_pred = prop_all.data[b]
            crt_pred_cen = pred_cen.data[b]
            crt_pred_len = pred_len.data[b]
            
            crt_nproposal = 0
            nproposal = torch.sum(torch.gt(prop_all.data[b, 0, :], pos_thresh)) # total number of proposals that confidence score (actioness score at index 0) is actually higher than threshold 0.7
            
            nproposal = min(max(nproposal, min_prop_num_before_nms), prop_all.size(-1))
            
            pred_results = np.empty((nproposal, 3))
            _, sel_idx = torch.topk(crt_pred[0], nproposal) # Get the index of 10 proposals with highest score 
            
            '''
            Non-Maximum Suppression Algorithm:

            1. Select the proposal with highest confidence score, remove it from B and add it to the final proposal list D. (Initially D is empty).
            2. Now compare this proposal with all the proposals — calculate the IOU (Intersection over Union) of this proposal with every other proposal. If the IOU is greater than the threshold N, remove that proposal from B.
            3. Again take the proposal with the highest confidence from the remaining proposals in B and remove it from B and add it to D.
            4. Once again calculate the IOU of this proposal with all the proposals in B and eliminate the boxes which have high IOU than threshold.
            5. This process is repeated until there are no more proposals left in B.
            '''
            
            for nms_thresh in nms_thresh_set:
                
                # Loop the proposals with hightest scores in descending order
                for prop_idx in range(nproposal):
                    original_frame_len = actual_frame_length[b].item()
                    pred_start_w = crt_pred_cen[sel_idx[prop_idx]] - crt_pred_len[sel_idx[prop_idx]] / 2.0
                    pred_end_w = crt_pred_cen[sel_idx[prop_idx]] + crt_pred_len[sel_idx[prop_idx]] / 2.0
                    pred_start = pred_start_w
                    pred_end = pred_end_w
                    if pred_start >= pred_end:
                        continue
                    if pred_end >= original_frame_len or pred_start < 0:
                        continue

                    hasoverlap = False
                    if crt_nproposal > 0:
                        
                        pred_start = pred_start.cpu().data.numpy()
                        pred_end = pred_end.cpu().data.numpy()
                        if np.max(segment_iou(np.array([pred_start, pred_end]), pred_results[:crt_nproposal])) > nms_thresh:
                            hasoverlap = True

                    if not hasoverlap:
                        prop_soi = torch.zeros(1, T, 1).type(dtype)
                        win_start = math.floor(max(min(pred_start, min(original_frame_len, T)-1), 0))
                        win_end = math.ceil(max(min(pred_end, min(original_frame_len, T)), 1))
                        
                        pred_results[crt_nproposal] = np.array([win_start,
                                                                win_end,
                                                                crt_pred[0, sel_idx[prop_idx]]])


                        prop_soi[:, win_start: win_end, : ] =1
                        
                        soi_result.append(prop_soi)
                        crt_nproposal += 1

                    if crt_nproposal >= max_prop_num:
                        break

                if crt_nproposal >= min_prop_num:
                    break

            if len(soi_result) == 0: # append all-one window if no window is proposed
                soi_result.append(torch.ones(1, T, 1).type(dtype))
                pred_results[0] = np.array([0, min(original_frame_len, self.soi_size), pos_thresh])
                crt_nproposal = 1

            pred_results = pred_results[:crt_nproposal]
            soi_result = torch.cat(soi_result, 0)
            
            soi_result = soi_result.view(crt_nproposal, T, 1)
            return pred_results, soi_result

class ModelMain(nn.Module):
    def __init__(self, d_model, n_class, stride_factor, kernel_list, slide_window_size=280, vis_dropout = 0.3  ):
        super().__init__()
        self.n_class = n_class
        self.d_model = d_model
        self.slide_window_size = slide_window_size
        self.cnn_encoder = ResNet_Encoder(cnn_out_dim=d_model, drop_prob=vis_dropout)    
        self.tempo_learn = TempoLearnModel(4, d_model, d_model)
        self.segment_prop = SegmentProposal(d_model, stride_factor, kernel_list)
        self.multi_classifier = TransformerClassifer(d_model, n_class, drop_ratio = vis_dropout)

    def forward(self, x, mask, pos_anc, neg_anc):
        
        x = self.cnn_encoder(x)
        B, T, ch = x.size()
        
        fix_x = torch.zeros(B, self.slide_window_size, ch).type(x.data.type())
        fix_x[:, :T, : ] = x[:, :self.slide_window_size, :]

        vis_feat = fix_x.transpose(1,2).contiguous()
        mask = mask.transpose(1,2).contiguous()

        vis_feat = self.tempo_learn(vis_feat, mask)
        pred_score, gt_score, pred_offsets, gt_offsets, soi = self.segment_prop(vis_feat, pos_anc, neg_anc)
        
        B, n_sample, T, gate = soi.size()
        ano_class_out = torch.from_numpy(np.empty((B, n_sample, self.n_class))).type(x.data.type())
        
        for s in range(n_sample):
            
            out = self.multi_classifier(fix_x, soi[:, s, :, :])
            
            ano_class_out[:, s, :] = out
        ano_class_out = ano_class_out.view(-1, self.n_class)
        return pred_score, gt_score, pred_offsets, gt_offsets, ano_class_out


    def inference(self, x, mask, vid_len, min_prop_num, max_prop_num, min_prop_before_nms, pos_thresh):
        
        x = self.cnn_encoder(x)
        B, T, ch = x.size()
        
        fix_x = torch.zeros(B, self.slide_window_size, ch).type(x.data.type())
        fix_x[:, :T, : ] = x[:, :self.slide_window_size, :]

        vis_feat = fix_x.transpose(1,2).contiguous()
        mask = mask.transpose(1,2).contiguous()

        vis_feat = self.tempo_learn(vis_feat, mask)
        pred_results, soi_results = self.segment_prop.inference(vis_feat, vid_len, min_prop_num, max_prop_num, min_prop_before_nms, pos_thresh)
        
        n_sample, T, gate = soi_results.size()
        ano_class_out = torch.from_numpy(np.empty((n_sample, self.n_class))).type(x.data.type())
        
        for s in range(n_sample):
            
            out = self.multi_classifier(fix_x, soi_results[s, :, :])
            ano_class_out[s, :] = out
        _, predicted_anomaly_class = torch.max(ano_class_out.data, 1)
        predicted_anomaly_class = predicted_anomaly_class.data.cpu().numpy()
        
        # use cap_batch as caption batch siz    
        all_proposal_results = []
        batch_result = []
        for idx in range(len(pred_results)):
            batch_result.append((pred_results[idx][0],
                                    pred_results[idx][1],
                                    pred_results[idx][2],
                                    predicted_anomaly_class[idx]))
        all_proposal_results.append(tuple(batch_result))

        return all_proposal_results

class SelfAttention(nn.Module):
    """ self attention module"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query = nn.Conv2d(in_channels=in_dim,
                             out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim,
                           out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim,
                             out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query(x).reshape(
            m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).reshape(m_batchsize, -1, width*height)
        energy = torch.matmul(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).reshape(m_batchsize, -1, width*height)

        out = torch.matmul(proj_value, attention.permute(0, 2, 1))
        out = out.reshape(m_batchsize, C, height, width)

        return out

class ResNet_Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, cnn_out_dim=256, drop_prob=0.1, bn_momentum=0.1):
        super(ResNet_Encoder, self).__init__()
    
        self.enc_image_size = encoded_image_size
        self.cnn_out_dim = cnn_out_dim
        self.bn_momentum = bn_momentum
        resnet = models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.attention_block = SelfAttention(in_dim=2048)
        
        self.fine_tune()
        
        self.fc = nn.Sequential(
                *[
                    self._build_fc(2048, cnn_out_dim, True),
                    nn.ReLU(),
                    nn.Dropout(p=drop_prob),
                ]
            )

    def _build_fc(self, in_features, out_features, with_bn=True):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features, momentum=self.bn_momentum)
        ) if with_bn else nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        b, t, ch, h, w = x.size()
        x = x.view(-1, ch, h, w)
        out = self.resnet(x)
        out = self.attention_block(out)
        out = out.mean(dim=[-1,-2]).squeeze()
        out = self.fc(out)
        out = out.view(b,t,out.size(-1))
        
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        for c in list(self.resnet.children())[-1:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

