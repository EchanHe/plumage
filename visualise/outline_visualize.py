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

def contour_to_mask(contour_coords , scale , img):
    height,width, _ = img.shape
    mask = np.zeros((height, width))
    for contour_coord in contour_coords:
        mask_temp = np.zeros((height, width))
        outline_coord = [[x//scale,y//scale] for (x,y) in zip(contour_coord[::2], contour_coord[1::2])]
        outline_coord = np.expand_dims(outline_coord , axis = 0)
        cv2.fillPoly(mask_temp, outline_coord, 1)
        mask = np.logical_xor(mask , mask_temp)
    return mask.astype('uint8')



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


save_mask = params['save_mask']

output_format = params['output_format']

gt_df = pd.read_csv(gt_file)
pred_df = pd.read_csv(pred_file)


# 
gt_df = pd.merge(pred_df[[file_col]],gt_df, on=[file_col])
gt_df = gt_df.sort_values(by=[file_col])
pred_df = pred_df.sort_values(by=[file_col])


gt_outlines = gt_df['outline'].values
pred_outlines = pred_df['outline'].values
file_names = gt_df[file_col].values

for i in range(0, gt_df.shape[0] , batch_size):

    print(i)
    pred_outline = str_to_array(pred_outlines[i])
    gt_outline = str_to_array(gt_outlines[i])



    filename = file_names[i]

    try:
        img = cv2.imread(img_path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print("file not exist: " + img_path+filename)
        continue 
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

    if save_mask:
        gt_mask = contour_to_mask(gt_outline, scale , img)
        pred_mask = contour_to_mask(pred_outline, scale, img)
        show_one_masks(plt, img, pred_mask = pred_mask, gt_mask = gt_mask
            , fig_name = filename,  save_path = save_path, 
            show_fig_title = True , format = output_format)
    else:
        show_one_markup(plt, img, pred_coord = None, pred_patch = pred_outline, pred_contour = None,
        gt_coord =None, gt_patch = gt_outline, gt_contour = None, pck_threshold =None, 
         fig_name = filename , save_path =save_path ,
          show_patch_labels = False , show_colour_labels = False , show_fig_title = False,
          linewidth = 3, format = output_format)


     
        
    plt.clf()
    plt.close()
