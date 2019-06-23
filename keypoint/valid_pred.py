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
config_name = 'config_pred_cpm.cfg'

params = process_config(os.path.join(dirname, config_name))

##reading data###
df_pred = pd.read_csv(params['pred_file'])

#df_pred = df_pred.sample(n=50,random_state=3)
# Create the name using some of the configuratation.
print(params['category'])
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
        img_mini = pred_data.get_next_batch_no_random()
        feed_dict = {model.images: img_mini}
        predict_mini = sess.run(predict, feed_dict=feed_dict)


        if 'save_heatmaps' in params and params['save_heatmaps']:
            # Save images.
            if not os.path.exists(params['heatmap_dir']):
                os.makedirs(params['heatmap_dir'])
            for i in range(params['points_num']):
                img_temp = predict_mini[0,...,i:i+1]
                img_temp = np.interp(img_temp,[np.min(img_temp),np.max(img_temp)],[0,255]).astype(np.uint8)
                img_temp = cv2.applyColorMap(img_temp, cv2.COLORMAP_JET)

                img_temp = cv2.resize(img_temp, dsize=(img_mini.shape[2], img_mini.shape[1]),
                    interpolation = cv2.INTER_NEAREST)

                dst = cv2.addWeighted(img_mini[0,...],0.3,img_temp,0.7,0)

                cv2.imwrite(params['heatmap_dir'] + "{}_{}.png".format(pred_names[start_id],pred_data.points_names[i]),
                            dst)


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