from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf

slim = tf.contrib.slim

import numpy as np
import pandas as pd

import sys, os
import network
from datetime import date

dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config, generate_grid_params
from points_util import heatmap_to_coord
from points_metrics import *
from points_io import write_pred_dataframe

print('--Parsing Config File')
config_name = 'config_pred.cfg'

params = process_config(os.path.join(dirname, config_name))

##reading data###
df_pred = pd.read_csv(params['pred_file'])

# df_pred = df_pred.sample(n=30,random_state=3)
# Create the name using some of the configuratation.
print(params['category'])
if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_pred = df_pred.loc[df_pred.view==params["category"],:].reset_index(drop = True)
elif params['category'] is None or params['category'] =='all':
    params['name'] +='_' + 'all'


print("Read training data ....")
pred_data = data_input.plumage_data_input(df_pred,batch_size=params['batch_size'],is_train =params['is_train'],
                           pre_path =params['img_folder'],state=params['data_state'],
                           scale=params['scale'] ,is_aug = params['img_aug'],
                           heatmap_scale = params['output_stride'])

# Whether validation or prediction
if 'is_valid' in params:
    is_valid = params['is_valid']
else:
    is_valid = False
##### Create the network using the hyperparameters. #####
tf.reset_default_graph()
model = network.CPM(params,pred_data.img_width, pred_data.img_height)

#Get prediction.
predict = model.inference_pose_vgg_l2()

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

    for start_id in range(0, pred_data.df.shape[0], pred_data.batch_size):
        #Generate heatmaps by batchs, so the memory won't overflow.
        img_mini = pred_data.get_next_batch_no_random()
        feed_dict = {model.pred_images: img_mini}
        predict_mini = sess.run(predict, feed_dict=feed_dict)
        pred_coord_mini = heatmap_to_coord(predict_mini , pred_data.img_width , pred_data.img_height)

        pred_coords = np.vstack((pred_coords, pred_coord_mini))
    pred_coords = pred_coords[:pred_data.df_size,...]     
    pred_df = write_pred_dataframe(pred_data, pred_coords,
        folder = params['pred_result_dir']+"pred/",
        file_name = str(date.today()) + params['name'],
        patches_coord=None, write_index = False)

if is_valid:
    gt_coords = pred_data.df[pred_data.coords_cols].values
        # Calculate metrics for points only
    diff_per_pt ,pck= pck_accuracy(pred_coords , gt_coords,
        lm_cnt = pred_data.lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)
    print(diff_per_pt ,pck)
    # Do metrics evaluation

    #Write the Evaluation result