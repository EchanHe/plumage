#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/11/30

Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.
'''

import numpy as np


#segs [batch_size , height, width]
def segs_to_masks(segs):
    assert len(segs.shape) ==3 , "Make sure input is [batch_size , height, width]"
    df_size = segs.shape[0]
    cl, n_cl = extract_classes(segs[0,...] )

    masks = np.zeros((segs.shape[0] , segs.shape[1] , segs.shape[2] , n_cl))
    for i in np.arange(df_size):
        masks[i,...] = extract_masks(segs[i,...] , cl, n_cl)

    return masks

#seg [height , width]
def seg_to_mask(seg):
    assert len(segs.shape) ==2 , "Make sure input is [height, width]"
    cl, n_cl = extract_classes(seg)
    return extract_masks(seg , cl, n_cl)


#pred_segms [batch_size, height, width ]
#gt_segms [batch_size, height, width ]
#mode: sum_accurary
def segs_eval(pred_segms , gt_segms , mode="pixel_acc"):
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


    return sum_acc/df_size

def correct_pred(eval_segm, gt_segm):
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

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((h, w , n_cl))

    for i, c in enumerate(cl):
        masks[:, : , i] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def segs_size(segs):
    try:
        df_size = segs.shape[0]
        height = segs.shape[1]
        width  = segs.shape[2]
    except IndexError:
        raise

    return df_size , height, width

def check_df_size(pred_segms, gt_segms):
    p_a,p_b,p_c = segs_size(pred_segms)
    g_a, g_b, g_c = segs_size(gt_segms)
    if (p_a != g_a) or (p_b != g_b) or (p_c!=g_c):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")
'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)