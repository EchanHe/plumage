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

from datetime import date
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# read Arguments from console input.`
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c","--cate", default = None, choices=["back" ,"belly","side" ],
                    help="The type of viewing")
args = parser.parse_args()

dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config
from points_util import heatmap_to_coord
from points_metrics import *

print('--Parsing Config File')
config_name = 'config_train.cfg'

params = process_config(os.path.join(dirname, config_name))


##reading data###
df_train = pd.read_csv(params['train_file'])
df_valid = pd.read_csv(params['valid_file'])

# params['category'] = args.cate
print(params['category'])
if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
elif params['category'] is None or params['category'] =='all':
    params['name'] +='_' + 'all'


# df_train = df_train.sample(n=500,random_state=3)
# df_valid = df_valid.sample(n=50,random_state=3)


print("Read training data ....")
train_data = data_input.plumage_data_input(df_train,batch_size=params['batch_size'],is_train =params['is_train'],
                           pre_path =params['img_folder'],state=params['data_state'],
                           scale=params['scale'] ,is_aug = params['img_aug'],heatmap_scale = 8)
print("Read valid data ....")
valid_data = data_input.plumage_data_input(df_valid,batch_size=params['batch_size'],is_train =params['is_train'],
                           pre_path =params['img_folder'],state=params['data_state'],
                           scale=params['scale'] ,is_aug = params['img_aug'],heatmap_scale = 8)

tf.reset_default_graph()
model = network.CPM(params,train_data.img_width, train_data.img_height )

#Get prediction, loss and train_operation.
predict = model.inference_pose_vgg_l2()
loss = model.loss()
train_op = model.train_op(loss, model.global_step)


#Calculate the training steps
train_data_size = train_data.df_size
total_steps = (params['nepochs'] * train_data_size) //params['batch_size']
summary_steps = total_steps // params['summary_interval']
valid_steps = total_steps // params['valid_interval']
saver_steps = total_steps // params['saver_interval']

#File name and paths
param_dir = params['saver_directory']
logdir = params['log_dir']
restore_file = params['restore_param_file']
save_filename = "{}_{}".format(str(date.today()) ,params['name'])
initialize = params['init']



print('Total steps: {}\nSum steps: {}, Valid steps: {}, Save steps: {}'.format(total_steps,
 summary_steps,valid_steps,saver_steps))

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    if os.listdir(param_dir) == [] or initialize:
        print ("Initializing Network")
        sess.run(init_op)
    else:
        print("Restore file from: {}".format(restore_file))
        sess.run(init_op)
        saver.restore(sess, restore_file)
#         model.restore(sess, saver, restore_file)
    sess.run(tf.local_variables_initializer())
    # merged = tf.summary.merge_all()
    train_summary = tf.summary.merge_all('train')
    

    writer = tf.summary.FileWriter(logdir, sess.graph)



        
    for i in range(total_steps):
        #Training part
        #以batch_size随机选择数据 迭代max_iteration次 
        tmp_global_step = model.global_step.eval()
        img_mini, heatmap_mini,coords_mini , vis_mini = train_data.get_next_batch()
        feed_dict = {
                    model.images: img_mini,
                    model.labels:heatmap_mini,
                    model.vis_mask: vis_mini
                    }
        sess.run(train_op, feed_dict=feed_dict)

        ###### Write training detail#####
        if (i+1) % summary_steps == 0:
            print("{} steps Loss: {}".format(i+1,sess.run(loss, feed_dict=feed_dict)))
            lear = model.learning_rate.eval()
#             print("\tGlobal steps and learning rates: {}  {}".format(tmp_global_step,lear))

            result_train=sess.run(predict, feed_dict=feed_dict)

            summary = sess.run(train_summary, feed_dict=feed_dict)    
            writer.add_summary(summary, tmp_global_step)
            
        ######Validating the result part#####    
        if (i+1) % valid_steps ==0:
            #Validation part
            #write the validation result

            loss_list = np.array([])

            _prediction = np.zeros((0,predict.shape[1], predict.shape[2], predict.shape[3]))
            for i_df_valid in np.arange(0,df_valid.shape[0],params["batch_size"]):
                img_mini, heatmap_mini,coords_mini , vis_mini = valid_data.get_next_batch_no_random()

                feed_dict = {
                    model.images: img_mini,
                    model.labels: heatmap_mini,
                    model.vis_mask: vis_mini
                    }            
                _loss = sess.run(loss, feed_dict=feed_dict)
                loss_list = np.append(loss_list,_loss)
                _prediction = np.vstack([_prediction,sess.run(predict, feed_dict=feed_dict) ])
            gt_coords = valid_data.df[valid_data.coords_cols].values
            pred_coord = heatmap_to_coord(_prediction , valid_data.img_width , valid_data.img_height)
            
            diff_per_pt ,pck= pck_accuracy(pred_coord , gt_coords,
             lm_cnt = valid_data.lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)
            print(diff_per_pt,pck)
            print(loss_list , np.mean(loss_list))


            summary = sess.run(model.valid_summary,
                feed_dict = {model.point_acc:diff_per_pt,model.valid_loss:np.mean(loss_list)})
            writer.add_summary(summary , tmp_global_step)  

            if (i + 1) % saver_steps == 0:
                #Write the parameters
                tmp_global_step = model.global_step.eval()
                epochs = (tmp_global_step*params["batch_size"])//train_data_size
                model.save(sess, saver, save_filename,epochs)


######## Create polygon patches, calculate all metrics and write metrics

# #Create the polygon coords around the predicted centroids.
# patch_coord = pred_coords_to_patches(pred_coord, half_width =10, half_height=10, ignore_coords =10)
# #Save the predicted dataframe to file.
# df_pred = write_pred_dataframe(valid_data , pred_coord , 
#     params['valid_result_dir']+"grid_temp/",
#      file_name = str(date.today()) + col_name,
#       patches_coord= patch_coord )
#     # write_seg_result(mean_miou, params , params['result_dir'] ,cor_pred = mean_cor_pred)