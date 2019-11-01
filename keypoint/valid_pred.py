from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf

slim = tf.contrib.slim

import numpy as np
import pandas as pd
import cv2
import sys, os
import network
from datetime import date

from sklearn.model_selection import train_test_split, KFold

dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config, generate_grid_params
from points_util import heatmap_to_coord
from points_metrics import *
from points_io import write_pred_dataframe,plot_heatmaps,save_heatmaps




print('--Parsing Config File')
config_name = 'config_pred_cpm.cfg'

params = process_config(os.path.join(dirname, config_name))

##reading data###


if 'split' not in params or params['split'] == False:
    df_pred = pd.read_csv(params['pred_file'])
else:
    df_all = pd.read_csv(params['pred_file'])
    
    # Split training and valdiation data, via category.
    if 'category_col' in params and params['category_col'] is not None:
        category_col = params['category_col']
    else:
        category_col = "view"

    if category_col not in df_all.columns:
        df_all[category_col] = 1

    gb = df_all.groupby("view")
    if params['kfold'] ==None:
        print("train_test_split with seed:",params['split_seed'] )
        # train_test_split split option
        split_list = [t for x in gb.groups for t in train_test_split(gb.get_group(x),
         test_size=params['test_size'], random_state =params['split_seed'])]
        
        df_train = pd.concat(split_list[0::2],sort = False)
        df_valid = pd.concat(split_list[1::2],sort = False)
    else:
        # Kfold option
        print("Kfold with seed: {} and {} th fold".format(params['split_seed'] ,params['kfold'] ))
        kf = KFold(n_splits=5 ,shuffle = True, random_state=params['split_seed'])
        train_list = []
        valid_list = []
        for key in gb.groups:
            data_view = gb.get_group(key)
            for idx, (train_index, valid_index) in enumerate(kf.split(data_view)):
                if idx ==params['kfold']:
                    train_list.append(data_view.iloc[train_index,:])         
                    valid_list.append(data_view.iloc[valid_index,:]) 

        df_train = pd.concat(train_list,sort = False)
        df_valid = pd.concat(valid_list,sort = False)    
    
    df_pred = df_valid
#df_pred = df_pred.sample(n=50,random_state=3)
# Create the name using some of the configuratation.

if params['category'] is not None and params['category'] !='all':
    params['name'] +='_' + params['category']
    df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
else:
    params['name'] +='_' + 'all'


print("Read training data ....")
pred_data = data_input.plumage_data_input(df_pred,batch_size=params['batch_size'],is_train =False,
                           pre_path =params['img_folder'],state=params['data_state'],file_col = params['file_col'],
                           scale=params['scale'] ,is_aug = params['img_aug'],
                           heatmap_scale = params['output_stride'])
params['point_names'] = pred_data.points_names
# Whether validation or prediction
if 'is_valid' in params:
    is_valid = params['is_valid']
else:
    is_valid = False

pred_data_size = pred_data.df_size
one_epoch_steps = pred_data_size//params['batch_size']
params["one_epoch_steps"] = one_epoch_steps

##### Create the network using the hyperparameters. #####
tf.reset_default_graph()
model = network.Pose_Estimation(params,pred_data.img_width, pred_data.img_height)
#Get prediction.
network_to_use = getattr(model, params['network_name'])
predict = network_to_use()

#File name and paths
restore_file = params['restore_param_file']
initialize = params['init']

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # Load the checkpoint
    if initialize:
        print ("Initializing Network")
        sess.run(init_op)
    else:
        print("Restore file from: {}".format(restore_file))
        sess.run(init_op)
        saver.restore(sess, restore_file)

    # 
    #make prediction from data
    pred_coords = np.zeros((0, 2*model.points_num))
    pred_names = df_pred[params['file_col']].values

    for start_id in range(0, pred_data.df.shape[0], pred_data.batch_size):
        #Generate heatmaps by batchs, so the memory won't overflow.
        # img_mini , heatmap_mini,_ , _ = pred_data.get_next_batch_no_random()
        img_mini = pred_data.get_next_batch_no_random()
        
        feed_dict = {model.images: img_mini}
        predict_mini = sess.run(predict, feed_dict=feed_dict)

        if 'summary_heatmaps' in params and params['summary_heatmaps']:
            plot_heatmaps(dir = params['heatmap_dir'],
             heatmaps = predict_mini[0,...], img = img_mini[0,...],
              file_name = pred_names[start_id],  names = pred_data.points_names, img_per_row = 5)

            # plot_heatmaps(dir = "/home/yichenhe/plumage/result/heatmap_test/heatmap_per_point/gt_sigma2/",
            #  heatmaps = heatmap_mini[0,...], img = img_mini[0,...],
            #   file_name = pred_names[start_id],  names = pred_data.points_names, img_per_row = 5)

        if 'save_heatmaps' in params and params['save_heatmaps']:
            save_heatmaps(dir = params['heatmap_dir'], heatmaps= predict_mini[0,...], 
                file_name = pred_names[start_id],  pt_names =  pred_data.points_names)

            # save_heatmaps(dir = "/home/yichenhe/plumage/result/heatmap_test/heatmap_per_point/gt_sigma2/",
            #  heatmaps= heatmap_mini[0,...], 
            #     file_name = pred_names[start_id],  pt_names =  pred_data.points_names)

        pred_coord_mini = heatmap_to_coord(predict_mini , pred_data.img_width , pred_data.img_height)

        pred_coords = np.vstack((pred_coords, pred_coord_mini))
    pred_coords = pred_coords[:pred_data.df_size,...]     
    if start_id % 1000==0:
        print(start_id, "steps")
    pred_df = write_pred_dataframe(pred_data, pred_coords,
        folder = params['pred_result_dir']+"pred/",
        file_name = str(date.today()) + params['name'], file_col_name = params['file_col'],
        patches_coord=None, write_index = False , is_valid = is_valid)

# If validation, print or save the metrics between ground-truth and predictions
if is_valid:
    gt_coords = pred_data.df[pred_data.coords_cols].values
        # Calculate metrics for points only
    diff_per_pt ,pck= pck_accuracy(pred_coords , gt_coords,
        lm_cnt = pred_data.lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)
    print(diff_per_pt ,pck)

    #Write the Evaluation result