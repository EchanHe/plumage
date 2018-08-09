
"""
Use for validation.
Read a dataframe with the labels
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import pandas as pd

import sys
import os
import network
## Add dir to path
dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config , save_config
from visualize_lib import *
from seg_io import masks_to_json
from seg_util import  segs_to_masks
import seg_metrics


print('--Parsing Config File')
params = process_config('config_valid.cfg')

model = network.Network(params)
predict = model.deeplab_v3()


valid_csv = pd.read_csv(params['valid_file'])
print("Read csv data from: ",params['valid_file'], "in folder:", params['img_folder'])
valid_data = data_input.plumage_data_input(valid_csv,
                                    batch_size=params['batch_size'],is_train =True,
                                    pre_path =params['img_folder'],state=params['data_state'],
                                    scale=params['scale'] ,is_aug = params['img_aug'])


restore_file = params['restore_param_file']
initialize = params['init']
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

# Generate the result from model
with tf.Session() as sess:
    if initialize:
        print ("Initializing Network")
        sess.run(init_op)
    else:
        assert os.path.exists(restore_file + ".index") , "Ckpt file is wrong, please check the config file!"
        print("Read checkpoint from: ", restore_file)
        sess.run(init_op)
        saver.restore(sess, restore_file)

    # Validation step
    result_valid = np.zeros((valid_data.df_size , predict.shape[1] , predict.shape[2]))
    gt_valid = np.zeros((valid_data.df_size , predict.shape[1] , predict.shape[2]))
    if params['save_img']:
        img_all = np.zeros((valid_data.df_size , predict.shape[1] , predict.shape[2] ,3 ))

    for i_df_valid in np.arange(0,valid_data.df_size,params["batch_size"]):
        x, y_valid= valid_data.get_next_batch_no_random()
        y_valid_segs = np.argmax(y_valid,axis = 3)
        feed_dict = {
            model.images: x
            }            
        result_mini = np.argmax(sess.run(predict, feed_dict=feed_dict) , axis = 3)
       
        result_valid[i_df_valid:i_df_valid+params["batch_size"],...] = result_mini
        gt_valid[i_df_valid:i_df_valid+params["batch_size"],...] = y_valid_segs

        if params['save_img']:
            img_all[i_df_valid:i_df_valid+params["batch_size"],...] = x

result_valid = result_valid.astype('uint8')

#Write the images
if params['save_img']:
    print("save into images into: ", params['visualize_img'])
    img_all = img_all.astype('uint8')
    save_masks_on_image(img_all ,result_valid ,
        save_path =params['visualize_img'], 
        fig_names= valid_csv['file.vis'].values)
    save_pred_diff_on_image(img_all ,gt_valid, result_valid , 1 , 
        save_path =params['visualize_img'], 
        fig_names= valid_csv['file.vis'].values)

json_file = params['result_dir']+"{}_result_{}.json".format(params['name'] , params['valid_file'].split('/')[-1])
masks_to_json(segs_to_masks(result_valid),json_file , valid_csv["file.vis"].values)


# Print the accuracy:
acc_iou = seg_metrics.segs_eval(result_valid,gt_valid,mode="miou")
acc_cor_pred = seg_metrics.segs_eval(result_valid,gt_valid,mode="correct_pred")
print("acc_iou",acc_iou )
print("acc_cor_pred : {}\n".format(acc_cor_pred))
for view in valid_csv['view'].unique():
    # valid_csv.loc[valid_csv['view'] == 'back' , :].i
    index = np.where(valid_csv['view'] == view)[0]
    pred_result = result_valid[index,...]
    gt_result = gt_valid[index,...]
    acc_iou = seg_metrics.segs_eval(pred_result,gt_result,mode="miou")
    acc_cor_pred = seg_metrics.segs_eval(pred_result,gt_result,mode="correct_pred")
    print("{} accuracy".format(view))
    print("acc_iou",acc_iou )
    print("acc_cor_pred : {}\n".format(acc_cor_pred))