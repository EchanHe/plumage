"""
Use for visualize and save image of landmarks

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
import os
import cv2

##### Segmentation part ##########

from seg_util import segs_to_masks,extract_classes


color_list = [np.array([255,0,0]),
              np.array([0,255,0]) ,
              np.array([0,0,255]),
              np.array([255,255,0]),
              np.array([0,255,255]) ,
              np.array([255,0,255])]


def save_masks_on_image(images, pred_segs, save_path, fig_names=None):
    """
    Save predction on top of images
    
    params:
        images [batch, height, width, 3]
        pred_segs [batch, height, width]  
        save_path directory of saving figures
        fig_names: a list of img names that used to name the figure
    """ 
    _check_shape_NHW(images ,pred_segs)  
    # Set the figure params
    if images.shape[0] < 10:
        batch = images.shape[0]
    else:
        batch = 10
    nrows = 2
    ncols =5
    alpha = 0.3
    
    result_size = pred_segs.shape[0]
    
    cl, n_cl = extract_classes(pred_segs)
    
    for i_start in range(0,result_size,batch):
        fig  = plt.figure(figsize=(100,40))       
        for i_row in range(i_start, i_start+batch):
            plt.subplot(nrows, ncols,(i_row)%10+1)
            image = images[i_row,...].copy()
            ori_image= images[i_row,...]
            for i_cl in range(1,n_cl):
                for c in range(3):
                    image[:, :, c] = np.where(pred_segs[i_row,...] == i_cl,
                                      ori_image[:, :, c] *
                                      (1 - alpha) + alpha * color_list[i_cl-1][c] ,
                                      image[:, :, c])
            if fig_names is not None:
                plt.title(fig_names[i_row] , fontsize = 20)   
            plt.imshow(image)
        fig.savefig(save_path+"seg_{}.jpg".format(i_start))
        # fig.savefig("/home/yichenhe/plumage/result_visualization/segment_contour/" + "image_and_mask{}.jpg".format(i_start))
        plt.close(fig)
inter_color_list = [np.array([255,0,0]),
                    np.array([0,255,0]),
                    np.array([0,0,255]) ]
def save_pred_diff_on_image(images, gt_segs, pred_segs , c , save_path , fig_names=None):
    """
    Save ground truth, prediction and their intersection of a certain class on image into figure
    
    params:
        images [batch, height, width, 3]
        pred_segs [batch, height, width] 
        gt_segs [batch, height, width] 
        c, the class of the segmentation result.
        save_path directory of saving figures
        fig_names: a list of img names that used to name the figure
    """ 

    _check_shape_NHW(images ,pred_segs, gt_segs)

    # Set the figure params
    if images.shape[0] < 10:
        batch = gt_segs.shape[0]
    else:
        batch = 10
    nrows = 2
    ncols =5
    alpha = 0.3
    
    result_size = gt_segs.shape[0]
    
    gt_mask = gt_segs == c
    pred_mask = pred_segs == c

    
    intersection = np.logical_and(gt_mask , pred_mask)
    
    color_result = np.zeros((intersection.shape))
    color_result[pred_mask==True] = 1
    color_result[gt_mask==True] = 2
    color_result[intersection==True] = 3
    

    
    for i_start in range(0,result_size,batch):
        fig  = plt.figure(figsize=(100,40)) 
        
        for i_row in range(i_start, i_start+batch):
            plt.subplot(nrows, ncols,(i_row)%10+1)
            image = images[i_row,...].copy()
            ori_image = images[i_row,...]
            for i_color in range(1,4):
                for c in range(3):
                    image[:, :, c] = np.where(color_result[i_row,...] == i_color,
                                      ori_image[:, :, c] * (1 - alpha) + alpha * inter_color_list[i_color-1][c] ,
                                      image[:, :, c])
            # Save the title of figure        
            if fig_names is not None:
                plt.title(fig_names[i_row] , fontsize = 20)          
            plt.imshow(image)
        fig.savefig(save_path+"segdiff_{}.jpg".format(i_start))
        plt.close(fig)

def save_masks(gt_segs, pred_segs, save_path, fig_names=None):
    """
    Save segmention of ground truth, prediction and their intersection into figure.
    
    params:
        pred_segs [batch, height, width] 
        gt_segs [batch, height, width] 
        save_path directory of saving figures
        fig_names: a list of img names that used to name the figure
    """ 

    _check_shape_NHW(pred_segs, gt_segs)

    # Set the figure params
    if gt_segs.shape[0] < 10:
        batch = gt_segs.shape[0]
    else:
        batch = 10
    nrows = 10
    ncols =3
    
    result_size = gt_segs.shape[0]
    intersection = (gt_segs == pred_segs)
    plot_list = [pred_segs,gt_segs,intersection]
    
    for i_start in range(0,result_size,batch):
        fig  = plt.figure(figsize=(30,100)) 
        for i_row in range(i_start, i_start+batch):
            for i_plot in range(0,ncols):    
                plt.subplot(nrows, ncols, (i_row*ncols+i_plot)%30+1)
                plt.imshow(plot_list[i_plot][i_row])

            # Save the title of figure        
            if fig_names is not None:
                plt.title(fig_names[i_row] , fontsize = 20)         
        fig.savefig(save_path+"segs_{}.jpg".format(i_start))
        plt.close(fig)


##### Segmentation part ##########


#### Coords part ######




#### Coords part ######s


##small util##
def _check_shape_NHW(a,b,*args):
    cond = (a.shape[:3] == b.shape[:3])
    for count, thing in enumerate(args): 
        assert type(a) == type(thing) , "All the args should be the same class"
        cond = cond and (a.shape[:3] == thing.shape[:3])

    if not cond:
        raise ShapeErr("NHW shape is different")
    
def _check_shape_NHWC(a,b,*args):
    cond = (a.shape == b.shape)
    for count, thing in enumerate(args): 
        assert type(a) == type(thing) , "All the args should be the same class"
        cond = cond and (a.shape == thing.shape)
    if not cond:
        raise ShapeErr("NHWC shape is different")


'''
Exceptions
'''
class ShapeErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)