"""
The metrics and accuracy for evaluting the segmentaions


"""


import numpy as np
from seg_util import *


def segs_eval(pred_segms , gt_segms , mode="pixel_acc"):
    """
    Average metrics of a batch of predict and ground truth segmentation with given mode.

    params:
        pred_segms [batch_size, height, width ]
        gt_segms [batch_size, height, width ]
        mode: The metrics used
    """
    check_df_size(pred_segms, gt_segms)
    df_size = pred_segms.shape[0]
    sum_acc=0
    if mode== "pixel_acc":

        for i in np.arange(df_size):
            sum_acc+=pixel_accuracy(pred_segms[i,...],gt_segms[i,...])
    elif mode =="miou":
        for i in np.arange(df_size):
            sum_acc+= mean_IU(pred_segms[i,...],gt_segms[i,...])
    elif mode =="mean_acc":
        for i in np.arange(df_size):
            sum_acc+= mean_accuracy(pred_segms[i,...],gt_segms[i,...])
    elif mode =="correct_pred":
        for i in np.arange(df_size):
            sum_acc+= correct_pred(pred_segms[i,...],gt_segms[i,...])


    return round(sum_acc/df_size , 4)

def correct_pred(eval_segm, gt_segm):
    '''
    i>=1
    sum_i(n_ii) / sum_i(pred_i_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0
    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        if c !=0:
            curr_eval_mask = eval_mask[:, :, i]
            curr_gt_mask = gt_mask[:, :, i]
     
            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            n_ij = np.sum(curr_eval_mask)
            IU[i] = n_ii / n_ij
    mean_IU_ = np.sum(IU) / (n_cl_gt - 1)
    return mean_IU_


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[:, :, i]
        curr_gt_mask = gt_mask[:, :, i]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[:, :, i]
        curr_gt_mask = gt_mask[:, :, i]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    IOU = true_positive / (true_positive + false_positive + false_negative)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[:, :, i]
        curr_gt_mask = gt_mask[:, :, i]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        # print(n_ii,t_i , n_ij )
        IU[i] = n_ii / (t_i + n_ij - n_ii)
        # print(IU[i] )
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_

def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[:, :, i]
        curr_gt_mask = gt_mask[:, :, i]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = get_pixel_area(eval_segm)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_