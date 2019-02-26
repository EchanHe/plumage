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

def turn_str_to_patch_coord(patches):
    patches_coords = [0]* patches.shape[0]
    for row in np.arange(patches.shape[0]):
        if type(patches[row]) is int:
            patches_coords[row] =np.array([-1])
        else:
            patches_coords[row] = np.array([float(s) for s in patches[row].split(",")])

    return patches_coords


args = sys.argv
if len(args)==2:
    config_name = args[1]
else:
    config_name = 'plot.cfg'
print('--Parsing Config File: {}'.format(config_name))

params = process_config(os.path.join(dirname, config_name))



## Visualize training images

#for each view
gt_file = params["gt_file"]
pred_file = params["pred_file"]
# view = "side"

img_path = params["img_folder"]
save_path = params["save_path"]

scale = params['scale']
batch_size =  params['batch_size']

gt_df = pd.read_csv(gt_file)
pred_df = pd.read_csv(pred_file)
gt_df = pd.merge(pred_df[['file.vis']],gt_df, on=['file.vis'])
gt_df = gt_df.sort_values(by=['file.vis'])
pred_df = pred_df.sort_values(by=['file.vis'])
# gt_df = gt_df.loc[gt_df.view == view].reset_index()

gt_data = data_input.plumage_data_input(gt_df, batch_size ,scale = scale, state = 'coords',
                                         is_train=True ,
                                           pre_path = img_path,is_aug=False )

pred_data  = data_input.plumage_data_input(pred_df, batch_size ,scale = scale, state = 'coords',
                                         is_train=True ,
                                           pre_path = img_path,is_aug=False )


gt_coords = gt_df[gt_data.coords_cols].values
pred_coords = pred_df[pred_data.coords_cols].values
file_names = gt_df["file.vis"].values
for i in range(0, gt_df.shape[0] , batch_size):

    pred_coord = pred_coords[i]
    gt_coord = gt_coords[i]
    filename = file_names[i]
    img = cv2.imread(img_path + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 


    pred_coord[gt_coord ==-1]=-1
    pred_coord[gt_coord ==-1]=pred_coord[gt_coord ==-1]//scale
    gt_coord[gt_coord ==-1]=gt_coord[gt_coord ==-1]//scale

    gt_patches =  create_rect_on_coords_proportion_length(np.expand_dims(gt_coord,axis=0).astype(float) , width =50,height=50 , ignore_coords =5 , circular_indexing = True)
    pred_patches =  create_rect_on_coords_proportion_length(np.expand_dims(pred_coord,axis=0).astype(float)  , width =50,height=50 , ignore_coords =5, circular_indexing = True)

    if i %100 ==0:
        process = psutil.Process(os.getpid())
        print("memory usage: ", i, process.memory_info().rss /1e6)
    

    fig  = plt.figure(figsize=(15, 8))
    

    if params['plot_result']:
        pred_patch =turn_str_to_patch_coord(np.array(pred_patches[0]))
        gt_patch =turn_str_to_patch_coord(np.array(gt_patches[0]))

        if "region" in pred_df.columns:
            new_save_path = save_path+pred_df.region.values[i] +"/"
        else:
            new_save_path = save_path

        if ("intensity_diff" in pred_df.columns) and ("pixel_diff" in pred_df.columns):
            plt.text(2500,100, "{} intensity_diff: {}".format(pred_df.region.values[i],pred_df.intensity_diff.values[i]),
                fontdict={'color': "white",'size':12 })
            plt.text(2500,200, "{} pixel_diff: {}".format(pred_df.region.values[i],pred_df.pixel_diff.values[i]),
                fontdict={'color': "white",'size':12 })
        

        show_one_markup(plt, img, pred_coord = None, pred_patch = pred_patch, pred_contour = None,
        gt_coord =None, gt_patch = gt_patch, gt_contour = None, pck_threshold =100, 
         fig_name = filename , LM_CNT =15, save_path =new_save_path , show_patch_labels = True)
    else:    
        diff_coord,_ = pck_accuracy(np.expand_dims(pred_coord,axis=0) , np.expand_dims(gt_coord,axis=0) ,15,100)
        for idx, coord in enumerate(diff_coord):
            distances = [100,200,300,400]
            for i in range(len(distances) - 1):
                lower = distances[i]
                upper = distances[i+1]
                if coord>=lower and coord<upper:
                    new_save_path =save_path + "{}_{}/{}/".format(lower , upper , gt_data.points_names[idx] ) 

                    show_one_markup(plt, img, pred_coord = pred_coord, pred_patch = None, pred_contour = None,
                        gt_coord =gt_coord, gt_patch = None, gt_contour = None, pck_threshold =100, 
                            fig_name = filename , LM_CNT =15, save_path =new_save_path )

            if coord >= distances[-1]:
                new_save_path =save_path + "{}_/{}/".format( distances[-1] , gt_data.points_names[idx]  ) 

                show_one_markup(plt, img, pred_coord = pred_coord, pred_patch = None, pred_contour = None,
                        gt_coord =gt_coord, gt_patch = None, gt_contour = None, pck_threshold =100, 
                            fig_name = filename , LM_CNT =15, save_path =new_save_path )

     
        
    plt.clf()
    plt.close()
