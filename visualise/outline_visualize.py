"""
Write labels and imagees
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os, sys
import psutil

dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config
from visualize_lib import *
from points_util import pred_coords_to_patches,create_rect_on_coords_proportion_length
from points_metrics import pck_accuracy

def _help_func_dict(config,key, default_value = None):
    if key in config:
        return config[key]
    else:
        return default_value

def turn_str_to_patch_coord(patches):
    patches_coords = [0]* patches.shape[0]
    for row in np.arange(patches.shape[0]):
        if type(patches[row]) is int:
            patches_coords[row] =np.array([-1])
        else:
            patches_coords[row] = np.array([float(s) for s in patches[row].split(",")])

    return patches_coords
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
    result = eval(s)
    if type(result) == tuple:
        result = list(result)
    else:
        result = [result]
    return result

args = sys.argv
if len(args)==2:
    config_name = args[1]
else:
    config_name = 'outline.cfg'
print('--Parsing Config File: {}'.format(config_name))

params = process_config(os.path.join(dirname, config_name))



## Visualize training images
gt_file = params["gt_file"]
pred_file = params["pred_file"]

img_path = params["img_folder"]
save_path = params["save_path"]

scale = params['scale']
batch_size =  params['batch_size']
file_col = params['file_col']



output_format = params['output_format']

gt_df = pd.read_csv(gt_file)
pred_df = pd.read_csv(pred_file)
pred_df = pred_df[:1]

# 
gt_df = pd.merge(pred_df[[file_col]],gt_df, on=[file_col])
gt_df = gt_df.sort_values(by=[file_col])
pred_df = pred_df.sort_values(by=[file_col])


gt_outlines = gt_df['outline'].values
pred_outlines = pred_df['obstruction'].values
file_names = gt_df[file_col].values

for i in range(0, gt_df.shape[0] , batch_size):

    pred_outline = str_to_array(pred_outlines[i])
    gt_outline = str_to_array(gt_outlines[i])



    filename = file_names[i]
    img = cv2.imread(img_path + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    if scale!=1:
        img = cv2.resize(img, dsize=(img.shape[1]//scale,img.shape[0]//scale), interpolation=cv2.INTER_CUBIC)

    #scale outline coordinates
    # pred_coord[gt_coord ==-1]=-1
    # pred_coord[gt_coord !=-1]=pred_coord[gt_coord !=-1]//scale
    # gt_coord[gt_coord !=-1]=gt_coord[gt_coord !=-1]//scale

    if i %100 ==0:
        process = psutil.Process(os.getpid())
        print("memory usage: ", i, process.memory_info().rss /1e6)
    

    fig  = plt.figure(figsize=(15, 8))
    
    ###############
    if "region" in pred_df.columns:
        new_save_path = save_path+pred_df.region.values[i] +"/"
    else:
        new_save_path = save_path

    if ("intensity_diff" in pred_df.columns) and ("pixel_diff" in pred_df.columns):

        plt.text(0.5,0.95, "{} intensity_diff: {}".format(pred_df.region.values[i],pred_df.intensity_diff.values[i]),
            fontdict={'color': "white",'size':12 },
            transform=plt.gca().transAxes)

        plt.text(0.5,0.90, "{} pixel_diff: {}".format(pred_df.region.values[i],pred_df.pixel_diff.values[i]),
            fontdict={'color': "white",'size':12 },
            transform=plt.gca().transAxes)
    ############


    show_one_markup(plt, img, pred_coord = None, pred_patch = pred_outline, pred_contour = None,
    gt_coord =None, gt_patch = gt_outline, gt_contour = None, pck_threshold =None, 
     fig_name = filename , save_path =new_save_path ,
      show_patch_labels = False , show_colour_labels = False , show_fig_title = False,
      linewidth = 3, format = output_format)



     
        
    plt.clf()
    plt.close()
