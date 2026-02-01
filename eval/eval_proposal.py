
import os
import json
import numpy as np

import pandas as pd
import sys
sys.path.append('../')
from utils import *

class EvalProposal(object):
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    GROUND_TRUTH_FIELDS = ['database']
    PROPOSAL_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename, test_list, proposal_filename,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 proposal_fields=PROPOSAL_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 max_avg_nr_proposals=None,
                 subset='validation', verbose=False,
                 check_status=True):
        
        
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.max_avg_nr_proposals = max_avg_nr_proposals
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = proposal_fields
        self.recall = None
        self.avg_recall = None
        self.proposals_per_video = None
        self.check_status = check_status
        
        self.ground_truth, self.activity_index = self._import_ground_truth(ground_truth_filename, test_list)
        self.proposal = self._import_proposal(proposal_filename)

    def _import_ground_truth(self, ground_truth_filename, testlist):
        
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []

        data = json.load(open(ground_truth_filename))
        classes = generate_classes(data)
        
        for videoid in testlist:
            videoid = videoid.split('.txt')[0]
            vinfo = data[videoid]
            video_lst.append(videoid)
            t_start_lst.append(vinfo['anomaly_start'])
            t_end_lst.append(vinfo['anomaly_end'])
            label_lst.append(classes[vinfo['anomaly_class']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        return ground_truth, activity_index

    def _import_proposal(self, proposal_filename):
        
        with open(proposal_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid proposal file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        score_lst = []
        for videoid, v in data['results'].items():
            for result in v:
                video_lst.append(videoid)
                t_start_lst.append(result['segment'][0])
                t_end_lst.append(result['segment'][1])
                score_lst.append(result['score'])
        proposal = pd.DataFrame({'video-id': video_lst,
                                 't-start': t_start_lst,
                                 't-end': t_end_lst,
                                 'score': score_lst})
        return proposal

    def evaluate(self):
        """Evaluates a proposal file. To measure the performance of a
        method for the proposal task, we computes the area under the
        average recall vs average number of proposals per video curve.
        """
        recall, avg_recall, proposals_per_video = average_recall_vs_avg_nr_proposals(
            self.ground_truth, self.proposal,
            max_avg_nr_proposals=self.max_avg_nr_proposals,
            tiou_thresholds=self.tiou_thresholds)

        area_under_curve = np.trapz(avg_recall, proposals_per_video)

        if self.verbose:
            print('[RESULTS] Performance on DoTA proposal task.')
            print('\tAUC score for Average Recall - Average Number of Proposals: {}%'.format(
                100. * float(area_under_curve) / proposals_per_video[-1]))

        self.area = 100. * float(area_under_curve) / proposals_per_video[-1]
        self.recall = recall
        self.avg_recall = avg_recall
        self.proposals_per_video = proposals_per_video
        return self.area


def average_recall_vs_avg_nr_proposals(ground_truth, proposals,
                                       max_avg_nr_proposals=None,
                                       tiou_thresholds=np.linspace(0.5, 0.95,
                                                                   10)):
    """ Computes the average recall given an average number
        of proposals per video.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    proposal : df
        Data frame containing the proposal instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        array with tiou thresholds.

    Outputs
    -------
    recall : 2darray
        recall[i,j] is recall at ith tiou threshold at the jth average number of average number of proposals per video.
    average_recall : 1darray
        recall averaged over a list of tiou threshold. This is equivalent to recall.mean(axis=0).
    proposals_per_video : 1darray
        average number of proposals per video.
    """

    # Get list of videos.
    video_lst = ground_truth['video-id'].unique()

    if not max_avg_nr_proposals:
        max_avg_nr_proposals = float(proposals.shape[0]) / video_lst.shape[0]

    ratio = max_avg_nr_proposals * float(video_lst.shape[0]) / proposals.shape[
        0]

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    proposals_gbvn = proposals.groupby('video-id')

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    total_nr_proposals = 0
    for videoid in video_lst:
        # Get ground-truth instances associated to this video.
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        this_video_ground_truth = ground_truth_videoid.loc[:,
                                  ['t-start', 't-end']].values

        # Get proposals for this video.
        try:
            proposals_videoid = proposals_gbvn.get_group(videoid)
        except:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        this_video_proposals = proposals_videoid.loc[:,
                               ['t-start', 't-end']].values

        if this_video_proposals.shape[0] == 0:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        # Sort proposals by score.
        sort_idx = proposals_videoid['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)
        if this_video_ground_truth.ndim != 2:
            this_video_ground_truth = np.expand_dims(this_video_ground_truth,
                                                     axis=0)

        nr_proposals = np.minimum(int(this_video_proposals.shape[0] * ratio),
                                  this_video_proposals.shape[0])
        total_nr_proposals += nr_proposals
        this_video_proposals = this_video_proposals[:nr_proposals, :]

        # Compute tiou scores.
        tiou = wrapper_segment_iou(this_video_proposals,
                                   this_video_ground_truth)
        score_lst.append(tiou)

    pcn_lst = np.arange(1, 101) / 100.0 * (
    max_avg_nr_proposals * float(video_lst.shape[0]) / total_nr_proposals)
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]
            # Find proposals that satisfies minimum tiou threshold.
            true_positives_tiou = score >= tiou
            # Get number of proposals as a percentage of total retrieved.
            pcn_proposals = np.minimum(
                (score.shape[1] * pcn_lst).astype(np.int), score.shape[1])

            for j, nr_proposals in enumerate(pcn_proposals):
                # Compute the number of matches for each percentage of the proposals
                matches[i, j] = np.count_nonzero(
                    (true_positives_tiou[:, :nr_proposals]).sum(axis=1))

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    avg_recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (
    float(total_nr_proposals) / video_lst.shape[0])

    return recall, avg_recall, proposals_per_video
