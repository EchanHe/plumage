
"""
Use to predict
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



print('--Parsing Config File')
params = process_config('config_pred.cfg')


tf.reset_default_graph()
model = network.Network(params)
predict = model.deeplab_v3()


valid_csv = pd.read_csv(params['valid_file'])
# valid_csv = valid_csv[:100]
valid_data = data_input.plumage_data_input(valid_csv,
                                           batch_size=params['batch_size'],is_train =params['is_train'],
                           pre_path =params['img_folder'],state=params['data_state'],
                           scale=params['scale'] ,is_aug = params['img_aug'])

param_dir = params['saver_directory']
logdir = params['log_dir']
restore_file = params['restore_param_file'] #"/home/yichenhe/plumage/params/contour/2018-07-31_deep_lab_v3_all-20"
initialize = params['init']
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
import time
start_time = time.time()
with tf.Session() as sess:
    if initialize:
        print ("Initializing Network")
        sess.run(init_op)
    else:
        assert os.path.exists(restore_file + ".index") , "Ckpt file is wrong, please check the config file!"
        print("Read checkpoint from: ".format(restore_file))
        sess.run(init_op)
        saver.restore(sess, restore_file)

    # Validation step
    miou_list = np.array([])
    cor_pred_list =np.array([])
    for i_df_valid in np.arange(0,valid_csv.shape[0],params["batch_size"]):
        x = valid_data.get_next_batch_no_random()
        feed_dict = {
            model.images: x
            }            
        result_mini = np.argmax(sess.run(predict, feed_dict=feed_dict) , axis = 3)
        
        ##### Action to do in every batch ######
        ##save the result visualization
       
        
        #### END     
        if i_df_valid == 0:
            result_valid = result_mini
            x_valid_all = x
        else:
            result_valid = np.vstack((result_valid , result_mini))
            x_valid_all = np.vstack((x_valid_all , x))

print("generate result--- {} mins ---".format((time.time() - start_time)/60))
start_time = time.time()
if params['save_img']:
    print("save into images into: ", params['visualize_img'])
    save_masks_on_image(x_valid_all ,result_valid ,save_path =params['visualize_img'] , 
        fig_names= valid_csv['file.vis'])


json_file = params['result_dir']+"result_{}.json".format(params['valid_file'].split('/')[-1])
print("Save result in :", json_file)
masks_to_json(segs_to_masks(result_valid),json_file , valid_csv["file.vis"])
print("Write result--- {} mins ---".format((time.time() - start_time)/60))