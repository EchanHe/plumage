from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import pandas as pd

import sys, os
import network
import itertools

from datetime import date
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config, generate_grid_params
from points_util import heatmap_to_coord
from points_metrics import *
from points_io import write_pred_dataframe, build_result_dict

print('--Parsing Config File')
config_name = 'config_train.cfg'

params = process_config(os.path.join(dirname, config_name))
grid_params = generate_grid_params(params)
print(grid_params)

##reading data###
df_train = pd.read_csv(params['train_file'])
df_valid = pd.read_csv(params['valid_file'])

df_train = df_train.sample(n=500,random_state=3)
df_valid = df_valid.sample(n=50,random_state=3)

# Create the name using some of the configuratation.
print(params['category'])
if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
elif params['category'] is None or params['category'] =='all':
    params['name'] +='_' + 'all'

# lr_list = np.empty([0])
################
if bool(grid_params):

    keys, values = zip(*grid_params.items())
    final_grid_df = pd.DataFrame()
    
    for id_grid,v_pert in enumerate(itertools.product(*values)):
        col_name = ""
        for key, value in zip(keys, v_pert):
            params[key] = value
            col_name += "{}-{};".format(key,value)

        ### Read the training data and validation data ###
        print("Read training data ....")
        train_data = data_input.plumage_data_input(df_train,batch_size=params['batch_size'],is_train =params['is_train'],
                                   pre_path =params['img_folder'],state=params['data_state'],
                                   scale=params['scale'] ,is_aug = params['img_aug'],
                                   heatmap_scale = params['output_stride'])
        print("Read valid data ....\n")
        valid_data = data_input.plumage_data_input(df_valid,batch_size=params['batch_size'],is_train =params['is_train'],
                                   pre_path =params['img_folder'],state=params['data_state'],
                                   scale=params['scale'] ,is_aug = params['img_aug'],
                                   heatmap_scale = params['output_stride'])

        ##### Create the network using the hyperparameters. #####
        tf.reset_default_graph()
        model = network.Pose_Estimation(params,train_data.img_width, train_data.img_height )

        network_to_use = getattr(model, params['network_name'])
        predict = network_to_use()
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
        logdir = os.path.join(params['log_dir'], col_name+str(date.today()))
        restore_file = params['restore_param_file']
        save_filename = "{}_{}".format(str(date.today()) ,params['name']) +col_name
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

            #### Get the summary of training and weight.
            train_summary = tf.summary.merge_all('train')
            weight_summary = tf.summary.merge_all('weight')
            writer = tf.summary.FileWriter(logdir, sess.graph)

                
            for i in range(total_steps):
                ####### Training part ########
                # Get input data and label from Training set, randomly.
                tmp_global_step = model.global_step.eval()
                img_mini, heatmap_mini,coords_mini , vis_mini = train_data.get_next_batch()
                feed_dict = {
                            model.images: img_mini,
                            model.labels:heatmap_mini,
                            model.vis_mask: vis_mini
                            }
                sess.run(train_op, feed_dict=feed_dict)

                ###### Train Summary part #####
                if (i+1) % summary_steps == 0 or i == 0:
                    print("{} steps Loss: {}".format(i+1,sess.run(loss, feed_dict=feed_dict)))
                    lear = model.learning_rate.eval()
        #             print("\tGlobal steps and learning rates: {}  {}".format(tmp_global_step,lear))

                    result_train=sess.run(predict, feed_dict=feed_dict)

                    summary,weight_s = sess.run([train_summary,weight_summary], feed_dict=feed_dict)    
                    writer.add_summary(summary, tmp_global_step)
                    writer.add_summary(weight_s, tmp_global_step)

                    # lr_list = np.append(lr_list, loss.eval(feed_dict=feed_dict))
                    
                ######Validating the result part#####    
                if (i+1) % valid_steps ==0 or i == 0:
                    #Validation part
                    #write the validation result

                    loss_list = np.array([])
                    # _prediction = np.zeros((0,predict.shape[1], predict.shape[2], predict.shape[3]))
                    pred_coords = np.zeros((0, 2*model.points_num))
                    for i_df_valid in np.arange(0,valid_data.df.shape[0],valid_data.batch_size):
                        img_mini, heatmap_mini,coords_mini , vis_mini = valid_data.get_next_batch_no_random()
                        feed_dict = {
                            model.images: img_mini,
                            model.labels: heatmap_mini,
                            model.vis_mask: vis_mini
                            }            
                        _loss, _prediction_mini = sess.run([loss,predict], feed_dict=feed_dict)
                        loss_list = np.append(loss_list,_loss)

                        pred_coord_mini = heatmap_to_coord(_prediction_mini , valid_data.img_width, valid_data.img_height)
                        pred_coords = np.vstack((pred_coords, pred_coord_mini))    
                        # _prediction = np.vstack([_prediction,sess.run(predict, feed_dict=feed_dict)])
                    pred_coords = pred_coords[:valid_data.df_size,...]    
                    gt_coords = valid_data.df[valid_data.coords_cols].values
                    # pred_coord = heatmap_to_coord(_prediction , valid_data.img_width , valid_data.img_height)
                    
                    diff_per_pt ,pck= pck_accuracy(pred_coords , gt_coords,
                        lm_cnt = valid_data.lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)

                    summary = sess.run(model.valid_summary,
                        feed_dict = {model.point_acc:diff_per_pt,model.valid_loss:np.mean(loss_list)})
                    writer.add_summary(summary , tmp_global_step)  
                ####### Save the parameters to computers.
                if (i + 1) % saver_steps == 0:        
                    tmp_global_step = model.global_step.eval()
                    epochs = (tmp_global_step*params["batch_size"])//train_data_size
                    model.save(sess, saver, save_filename,epochs)

        # Evaluate all and write in a dataframe.
        # The csv of both Ground truth and validation.
        gt_coords = valid_data.df[valid_data.coords_cols].values
        # pred_coord = heatmap_to_coord(_prediction , valid_data.img_width , valid_data.img_height)
        # Calculate metrics for points only
        diff_per_pt ,pck= pck_accuracy(pred_coords , gt_coords,
            lm_cnt = valid_data.lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)
        # Write the validation result to csv
        write_pred_dataframe(valid_data , pred_coords ,
            folder = params['valid_result_dir']+"grid_temp/",
            file_name = str(date.today()) + col_name,
            patches_coord=None, write_index = False )

        result_dict = params
        result_dict = build_result_dict(result_dict = params,
            pck = np.round(pck, 4), mean_pck = round(np.nanmean(pck), 4), pck_threshold = params['pck_threshold'],
            diff_per_pt=np.round(diff_per_pt, 4), mean_diff_per_pt = round(np.nanmean(diff_per_pt), 4))

        # result_dict['mean_pck'] = round(np.nanmean(pck), 4)
        # result_dict['mean_diff_per_pt'] = round(np.nanmean(diff_per_pt), 4)
        # result_dict['pck{}'.format(params['pck_threshold'])] = np.round(pck, 4)
        # result_dict['diff_per_pt'] = np.round(diff_per_pt, 4)

        # result_dict = {str(k):str(v) for k,v in result_dict.items()}
        final_grid_df = final_grid_df.append(pd.DataFrame(result_dict, index=[id_grid]))

    final_grid_df.to_csv(params['valid_result_dir']+ "{}grid_search.csv".format(str(date.today())), index = False)    

# lr_list = np.round(lr_list,4)
# N=5
# print(lr_list)
# print(np.diff(lr_list))
# print(np.convolve(lr_list, np.ones((N,))/N, mode='valid'))