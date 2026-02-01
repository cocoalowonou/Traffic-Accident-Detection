import numpy as np

def generate_segment_exp(meta_data, data_list, classes_dict):
    segment = {}
    
    for video in data_list:
        vinfo = meta_data[video]
        total_frames = vinfo['num_frames']
        start_time = vinfo['anomaly_start']
        end_time = vinfo['anomaly_end']
        label = classes_dict[vinfo['anomaly_class']]
        segment[video] = [total_frames, start_time, end_time, label]

    return segment

def generate_classes(data):
    class_list = []
    for vid, vinfo in data.items():
        class_list.append(vinfo['anomaly_class'])
    class_list = list(set(class_list))
    class_list = sorted(class_list)
    classes = {}
    for i,cls in enumerate(class_list):
        classes[cls] = i
    return classes


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def interpolated_prec_rec(prec, rec):
    
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in range(m):
        tiou[:, i] = segment_iou(target_segments[i,:], candidate_segments)

    return tiou

# from NBT
def update_values(dict_from, dict_to):
    
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def cat2labels(label_encoder, list):
    return label_encoder.inverse_transform(list)

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def count_per_class(classes_dict, lable_list, le):
    result = {}
    for cls_name, idx in classes_dict.items():
        result[cls_name] = list(cat2labels(le, lable_list)).count(cls_name)
    return result


def read_txt(path):
    data_list = open(path, 'r').readlines()
    result_list = []
    for data in data_list:
        result_list.append(data.rstrip())
    return result_list


def write_txt(path, list):
    with open(path, 'w') as f:
        for vid in list:
            f.write(f'{vid}\n')