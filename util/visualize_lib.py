"""
Use for visualize and save image of landmarks

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
import os
import cv2
import re

##### Segmentation part ##########

from seg_util import segs_to_masks,extract_classes


color_list = [np.array([255,0,0]),
              np.array([0,255,0]) ,
              np.array([0,0,255]),
              np.array([255,255,0]),
              np.array([0,255,255]) ,
              np.array([255,0,255]),
              np.array([255,0,0]),
              np.array([0,255,0]) ,
              np.array([0,0,255]),
              np.array([255,255,0]),
              np.array([0,255,255]) ,
              np.array([255,0,255])
              ]


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
    result_size = images.shape[0]
    if result_size < 10:
        batch = result_size
    else:
        batch = 10
    nrows = 2
    ncols =5
    alpha = 0.3
    
    
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
        if not os.path.exists(save_path):
            os.makedirs(save_path)
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
    result_size = gt_segs.shape[0]
    if result_size < 10:
        batch = result_size
    else:
        batch = 10
    nrows = 2
    ncols =5
    alpha = 0.3
    
    
    
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
    result_size = gt_segs.shape[0]
    if result_size < 10:
        batch = result_size
    else:
        batch = 10
    nrows = 10
    ncols =3
    
    
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

# LM_CNT = 16

def show_markups(imgs , pred_coords = None , pred_patches =None , pred_contours=None,
    gt_coords =None , gt_patches=None, gt_contours=None, pck_threshold = None, 
    save_path = None, image_name ="markup" , sub_fig_names = None , LM_CNT=16):
    """
    Plot the markups and images
    
    params:
        img: [batch, height, width, 3]
        gt / pred_coords: coordinations [batch , landmark *2 ]    
        gt / pred_patches: a list of patches coords  [batch][patches count][patch points *2 ].
        gt / pred_contours: a list of contours coords  [batch][contours count][contour points *2 ].
        save_path directory of saving figures
        fig_name: name of the fig was saved
        sub_fig_names: a list of img names that used to name the figure
    """
    result_size = imgs.shape[0]
    if result_size < 10:
        batch = result_size
    else:
        batch = 10

    ncols = batch
    # batch_size = ncols = imgs.shape[0]
    # assert batch_size <=20, "The batch size is larger than 20, bad for plotting"
    nrows = 1

    

    for i_start in range(0,result_size,batch):
        fig  = plt.figure(figsize=(ncols * 10, nrows *10))
        for i_row in range(i_start, i_start+batch):
            plt.subplot(nrows, ncols,(i_row)%(ncols*nrows)+1)

            show_one_markup(plt, img = imgs[i_row,...] ,
            pred_coord = _none_or_element(pred_coords ,i_row) , pred_patch =_none_or_element(pred_patches ,i_row),
            pred_contour = _none_or_element(pred_contours ,i_row),
            gt_coord = _none_or_element(gt_coords ,i_row), gt_patch =   _none_or_element(gt_patches ,i_row),
            gt_contour = _none_or_element(gt_contours ,i_row),pck_threshold = pck_threshold,
            fig_name = _none_or_element(sub_fig_names ,i_row) , LM_CNT=LM_CNT)

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(save_path+image_name+"_{}.jpg".format(i_start))
        plt.close("all")
        fig.clf()


def show_one_markup(plt, img, pred_coord , pred_patch , pred_contour,
    gt_coord  , gt_patch, gt_contour,pck_threshold, fig_name = "" , LM_CNT =16, save_path = None):
    """
    Plot one image and markup using show_coords and show_patches

    params:
        plt : pyplot
        img: [height, width, 3]
        gt / pred_coord: coordinations [landmark *2 ]    
        gt / pred_patch: a list of patches coords [patches count][patch points *2 ].
        gt / pred_contour: a list of contours coords [contours count][contour points *2 ].
        fig_name: The img name that used to name current figure
    """

    plt.title(fig_name , fontsize = 20)
    plt.imshow(img)
    # Show the predict mark and pred_patch
    show_patches(plt, pred_patch)
    show_patches(plt, pred_contour , is_patch = False)
    show_coords(plt, pred_coord , LM_CNT = LM_CNT)

    show_patches(plt, gt_patch, 'red')
    show_patches(plt, gt_contour , 'red', is_patch = False)
    show_coords(plt, gt_coord, pck_threshold, 'red' , LM_CNT = LM_CNT)
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + fig_name)
        plt.close("all")
        plt.clf()

patches_names = ['s1','s2','s3','s4','s5', 'crown', 'nape','mantle', 'rump', 'tail',
'throat', 'breast', 'belly', 'tail\nunderside',
  'wing\ncoverts',   'wing\nprimaries\nsecondaries',]


patches_names = ['s1','s2','s3','s4','s5', 'crown', 'nape','mantle', 'rump', 'tail',
'throat', 'breast', 'belly', 
  'wing\ncoverts',   'wing\nprimaries\nsecondaries',]

def show_coords(plt, coord ,pck_threshold = None, color = 'cyan' , LM_CNT = 16):
    """
    Plot the 2D points on figure

    params:
        plt : pyplot
        coord [landmark *2 ]. eg [x1,y1, ... , xn, yn]
        lm_cnt: The landmark count
        color: The color of the point

    """
    if coord is None:
        return 0
    lm_cnt = LM_CNT
    cols_num_per_coord = 2
    if pck_threshold is not None:
        x = coord[0]
        y = coord[1] 
        plt.plot([x,x], [y,y+pck_threshold],color = 'white' , linewidth = 2.0)
        # plt.text(x* (1.01) , y * (1.01)  , str(pck_threshold)+" pixels", 
        #                      fontsize=10 , bbox=dict(facecolor='white', alpha=0.4))
    for i_col in range(lm_cnt):
        x = coord[ i_col * cols_num_per_coord]
        y = coord[ i_col * cols_num_per_coord +1] 
        if x >= 0 and ~np.isnan(x):
            plt.plot(x, y, 'x' , alpha=0.8 , mew = 3 , mec = color )

            plt.text(x , y-200 ,
                patches_names[i_col], 
                fontdict={'color': color,'size':12 })
    # Write the labels at the text. 


def show_patches(plt,patch, color = 'cyan' , is_patch = True ):
    """
    Plot the patches on figure

    params:
        plt : pyplot
        patch [patches count][patch points *2 ]. eg [x1,y1, ... , xn, yn]
    """ 
    if patch is None:
        return 0

    for id_p, p_coord in enumerate(patch):
        # print(p_coord)
        length = p_coord.shape[0]
        if length>=2:
            for i in range(0,length,2):
                x = (p_coord[i%length] , p_coord[(i+2)%length])
                y = (p_coord[(i+1)%length], p_coord[(i+3)%length])
                plt.plot(x,y,color = color,lw =2)
            # if is_patch:
            #     plt.text(p_coord[0] , p_coord[1]-20 ,
            #         patches_names[id_p][5:], 
            #         fontdict={'color': color,'size':12 })
                
                

#### Coords part ######


#### Result analysis part ####

def analyse_time(time_series, plt, seg_duration = 100):
    """
    Goal: analyse the time
    
    params:
        time_series: the series of with time segs [ts_1, ..., ts_n] as value and configs as index
        plt: plot
        seg_duration, the iters for one time segmentions. Default value is 100
    """
    fig=plt.figure()
    ax = fig.add_subplot(211)  
        #Draw the barplot
    ax2 = fig.add_subplot(212)  
    
    result = pd.DataFrame()
    
    for i, v in time_series.items():
        
        times = str_to_array(v)

        segs = times.shape[0]
        total_time = segs * seg_duration
        segs = np.arange(0, total_time, seg_duration )
        
        # Draw the labels of the configs
        draw_line_plot(ax, segs, times, i)
        
        result.loc['time',i] = times[-1]
        
        ax2.barh(i, times[-1], align='center')
        
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_title('Time', fontsize = 'x-large')  # Set the pokemon's name as the title
    plt.ylabel("Sec")
    plt.xlabel("Iterations")

    
    return result


def draw_radar(ax,vals,labels , config_name , title):
    """
    Goal: draw radar plot
    
    params:
        ax: axis from matplot
        vals
    """
    
    angles=np.linspace(0, 2*np.pi, len(vals), endpoint=False) # Set the angle
    angles=np.concatenate((angles,[angles[0]]))

    vals=np.concatenate((vals,[vals[0]])) 

    ax.plot(angles, vals, 'o-' , label=config_name)  # Draw the plot (or the frame on the radar chart)
    ax.fill(angles, vals, alpha=0.25)  #Fulfill the area  
    if "diff" not in title:    
        ax.set_ylim([0,1.0])    

#     angles=np.linspace(np.pi/2, 2*np.pi+np.pi/2, len(labels), endpoint=False) # Set the angle
    if labels is not None:
        ax.set_thetagrids(angles * 180/np.pi, labels)  # Set the label for each axis
        
def radar_for_df(plt, df , metric, labels, exclude_index =None , include_index = None):
    """
    Goal: draw radar plot for all configs of the dataframe
    
    params:
        plt: plt
        df: data frame of the stat table format
        metric: metric to draw
        labels: a list of label for every class of value
        exclude_index: default is None, and exclude the index of value list.
    """
    
    config_names = df["name"]

    #Plot for list value for every patches or labels.
    fig = plt.figure()
    ax = plt.gca(projection = "polar")
    
    if exclude_index is not None:
        labels = np.delete(labels,exclude_index)
    if include_index is not None:
        labels = np.take(labels, include_index)
    for idx,confid_name in enumerate(config_names):
        value = df.loc[idx ,metric]
        value = str_to_array(value)
        
        if exclude_index is not None:
            value = np.delete(value,exclude_index)
        if include_index is not None:
            value = np.take(value, include_index)
        draw_radar(ax,value, labels, confid_name, metric)
    
    ax.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.) #write the labels box  
    fig.suptitle(metric , y = 1.0, fontsize = 20)

def draw_barplot(ax, val, x, config_name, horizontal = True):
    """
    Goal: draw bar plot
    
    params:
        ax: axis from matplot
        val: numerical value
        x: the name of x value
        config_name: the configuration name
        horizontal: whether draw the horizontal plot or not
    """
    if horizontal:
        ax.barh(x,val, align='center',label = config_name)
    else:
        ax.bar(x,val, align='center',label = config_name)
    

def barplot_for_df(plt, df, metric, horizontal= True):
    """
    Goal: draw bar plot for all configs of the dataframe
    
    params:
        plt: plt
        df: data frame of the stat table format
        metric: metric to draw
        horizontal: whether draw the horizontal plot or not
    """
    
    config_names = df["name"]

    #Plot for list value for every patches or labels.
    fig = plt.figure()
    ax = plt.gca()
    for idx,confid_name in enumerate(config_names):
        value = float(df.loc[idx ,metric])
        draw_barplot(ax,value, idx, confid_name, horizontal = horizontal)
    
    ax.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.) #write the labels box  
    fig.suptitle(metric)
    

##small util##

def str_to_array(s):
    """
    Goal: Cast string array to np array
    
    params:
        s, string
    
    return np.array
    """
    
    #Check whether string is '[....] format'
    m = re.match('\[[\S*\s*]*\]', s)
    assert m is not None, "The string is not in \'[....]\'"
    #transer string to array
    if ',' not in s:
        s = re.sub( '\[\s+', '[', s )
        s = re.sub( '\s+\]', ']', s )
        s = re.sub( '\s+', ',', s ).strip()
    result = np.array(eval(s))
    return result


def _none_or_element(array , id_batch):
    if array is not None:
        return array[id_batch]
    else:
        return None


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