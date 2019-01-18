from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import pandas as pd

import sys, os
import network
from network import _help_func_dict
import itertools
from sklearn.model_selection import train_test_split

from datetime import date
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config, generate_grid_params, extract_config_name
from points_util import heatmap_to_coord
from points_metrics import *
from points_io import write_pred_dataframe, build_result_dict

def read_csv(params):
    ##reading data
    df_all = pd.read_csv(params['input_file'])
    gb = df_all.groupby("view")
    split_list = [t for x in gb.groups for t in train_test_split(gb.get_group(x),
     test_size=params['test_size'], random_state =params['split_seed'])]
    
    df_train = pd.concat(split_list[0::2],sort = False)
    df_valid = pd.concat(split_list[1::2],sort = False)



    # ##reading data###
    # df_train = pd.read_csv(params['train_file'])
    # df_valid = pd.read_csv(params['valid_file'])

    #Sampling a sub set
    if 'small_data' in params and params['small_data']:
        df_train = df_train.sample(n=100,random_state=3)
        df_valid = df_valid.sample(n=20,random_state=3)

    # Create the name using some of the configuratation.
    print(params['category'])
    if params['category'] is not None and params['category'] !='all':
        params['name'] +='_' + params['category']
        df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
        df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
    else:
        params['name'] +='_' + 'all'
    return df_train, df_valid

def create_data(params, df_train, df_valid):

    ### Read the training data and validation data ###
    print("Read training data ....")
    train_data = data_input.plumage_data_input(df_train,batch_size=params['batch_size'],is_train =params['is_train'],
                               pre_path =params['img_folder'],state=params['data_state'],
                               scale=params['scale'] ,is_aug = params['img_aug'],
                               heatmap_scale = params['output_stride'] ,
                               view = params['category'],no_standard = _help_func_dict(params,"no_standard",False))
    print("Read valid data ....\n")
    valid_data = data_input.plumage_data_input(df_valid,batch_size=params['batch_size'],is_train =params['is_train'],
                               pre_path =params['img_folder'],state=params['data_state'],
                               scale=params['scale'] ,is_aug = False,
                               heatmap_scale = params['output_stride'],
                                view = params['category'],no_standard = _help_func_dict(params,"no_standard",False) )
    params['points_num'] = train_data.lm_cnt
    extract_config_name(params)
    return train_data, valid_data

def trainining(params, train_data, valid_data):

    config_name = params['config_name']

    #Calculate the training steps
    train_data_size = train_data.df_size
    one_epoch_steps = train_data_size//params['batch_size']
    params["one_epoch_steps"] = one_epoch_steps
    total_steps = (params['nepochs'] * train_data_size) //params['batch_size']
    summary_steps =  total_steps // params['summary_interval']
    valid_steps = total_steps // params['valid_interval']
    saver_steps = total_steps // params['saver_interval']
    print('Total steps: {}\nOne epoch: {}\nSum steps: {}, Valid steps: {}, Save steps: {}'.format(total_steps,one_epoch_steps,
        summary_steps,valid_steps,saver_steps))

    tf.reset_default_graph()
    model = network.Pose_Estimation(params,train_data.img_width, train_data.img_height )

    network_to_use = getattr(model, params['network_name'])
    predict = network_to_use()
    loss = model.loss()
    train_op = model.train_op(loss, model.global_step)

    #File name and paths
    param_dir = params['saver_directory']
    logdir = os.path.join(params['log_dir'], config_name+str(date.today()))
    restore_file = params['restore_param_file']
    save_filename = "{}_{}".format(str(date.today()) ,params['name']) + config_name
    initialize = params['init']

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
        # weight_summary = tf.summary.merge_all('weight')
        writer = tf.summary.FileWriter(logdir, sess.graph)

            
        for i in range(total_steps):
            ####### Training part ########
            # Get input data and label from Training set, randomly.
            tmp_global_step = model.global_step.eval()
            img_mini, heatmap_mini,coords_mini , vis_mini = train_data.get_next_batch()
            # print(np.count_nonzero(vis_mini==0))
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

                temp_summary = sess.run(train_summary, feed_dict=feed_dict)    
                writer.add_summary(temp_summary, tmp_global_step)
                # lr_list = np.append(lr_list, loss.eval(feed_dict=feed_dict))
                
            ######Validating the result part#####    
            if (i+1) % valid_steps ==0 or i == 0:
                #Validation part
                #write the validation result

                loss_list = np.array([])
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
                pred_coords = pred_coords[:valid_data.df_size,...]    
                gt_coords = valid_data.df[valid_data.coords_cols].values

                diff_per_pt ,pck= pck_accuracy(pred_coords , gt_coords,
                    lm_cnt = valid_data.lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)
                ave_diff = np.nanmean(diff_per_pt)
                summary = sess.run(model.valid_summary,
                    feed_dict = { model.point_acc:diff_per_pt, 
                                    model.valid_loss:np.mean(loss_list),
                                    model.ave_pts_diff:ave_diff})
                writer.add_summary(summary , tmp_global_step)  
            ####### Save the parameters to computers.
            if (i + 1) % saver_steps == 0:        
                tmp_global_step = model.global_step.eval()
                epochs = (tmp_global_step*params["batch_size"])//train_data_size
                model.save(sess, saver, save_filename,epochs)  
        params['restore_param_file'] = "{}-{}".format(save_filename, epochs)
    return model, predict    

def get_and_eval_result(params, valid_data):
    params_valid = params.copy()
    params_valid['is_train'] = False
    params_valid['l2'] = 0.0

    tf.reset_default_graph()
    model = network.Pose_Estimation(params_valid,
        valid_data.img_width, valid_data.img_height )

    network_to_use = getattr(model, params_valid['network_name'])
    predict = network_to_use()

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    # Get the predictions:
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, params_valid['saver_directory'] + params_valid["restore_param_file"])

        pred_coords = np.zeros((0, 2*valid_data.lm_cnt))
        for i_df_valid in np.arange(0,valid_data.df.shape[0],valid_data.batch_size):
            img_mini, heatmap_mini,coords_mini , vis_mini = valid_data.get_next_batch_no_random()
            feed_dict = {
                model.images: img_mini,
                model.labels: heatmap_mini,
                model.vis_mask: vis_mini
                }            
            _prediction_mini = sess.run(predict, feed_dict=feed_dict)

            pred_coord_mini = heatmap_to_coord(_prediction_mini , valid_data.img_width, valid_data.img_height)
            pred_coords = np.vstack((pred_coords, pred_coord_mini))    

        pred_coords = pred_coords[:valid_data.df_size,...]    

    gt_coords = valid_data.df[valid_data.coords_cols].values
    diff_per_pt ,pck= pck_accuracy(pred_coords , gt_coords,
            lm_cnt = valid_data.lm_cnt , pck_threshold = params_valid['pck_threshold'],scale = 1)




    write_pred_dataframe(valid_data , pred_coords ,
        folder = params_valid['valid_result_dir']+"grid_temp/",
        file_name = str(date.today()) + params_valid['config_name'],
        patches_coord=None, write_index = False )

    result_dict = build_result_dict(result_dict = params_valid,
        pck = np.round(pck, 4), mean_pck = round(np.nanmean(pck), 4), pck_threshold = params_valid['pck_threshold'],
        diff_per_pt=np.round(diff_per_pt, 4), mean_diff_per_pt = round(np.nanmean(diff_per_pt), 4))

    return result_dict, pred_coords



args = sys.argv
if len(args)==2:
    config_name = args[1]
else:
    config_name = 'config_train_hg.cfg'
print('--Parsing Config File: {}'.format(config_name))

params = process_config(os.path.join(dirname, config_name))
grid_params = generate_grid_params(params)
print(grid_params)



# Trainning and validation
################
if bool(grid_params):

    keys, values = zip(*grid_params.items())
    final_grid_df = pd.DataFrame()
    
    #Generate parameters for grid search    
    for id_grid,v_pert in enumerate(itertools.product(*values)):
        config_name = ""
        for key, value in zip(keys, v_pert):
            params[key] = value
            config_name += "{}-{};".format(key,value)
        df_train,df_valid = read_csv(params)
        train_data, valid_data = create_data(params, df_train,df_valid)    

        ##### Create the network using the hyperparameters. #####
        model , predict = trainining(params, train_data, valid_data)

        # produce the result;
        result_dict , pred_coords = get_and_eval_result(params , valid_data)


        final_grid_df = final_grid_df.append(pd.DataFrame(result_dict, index=[id_grid]))

    final_grid_df.to_csv(params['valid_result_dir']+ "{}grid_search.csv".format(str(date.today())), index = False)    


# training seperating models for different view, or combinations of keypoings
train_by_views = False
views = ['back','belly','side']
no_standards = [True, True, True]
if train_by_views:
    final_grid_df = pd.DataFrame()
    for view, no_standard in zip(views, no_standards):
        # print(view, no_standard)
        params['category'] = view
        params['no_standard'] = no_standard
        train_data, valid_data = read_data(params)

        model , predict = trainining(params, train_data, valid_data)
        # produce the result;
        pred_coords = get_result(model,predict, valid_data)

        # Evaluate all and write in a dataframe.
        # The csv of both Ground truth and validation.
        gt_coords = valid_data.df[valid_data.coords_cols].values

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
        final_grid_df = final_grid_df.append(pd.DataFrame(result_dict, index=[id_grid]))

    final_grid_df.to_csv(params['valid_result_dir']+ "{}by_view.csv".format(str(date.today())), index = False)    
# lr_list = np.round(lr_list,4)
# N=5
# print(lr_list)
# print(np.diff(lr_list))
# print(np.convolve(lr_list, np.ones((N,))/N, mode='valid'))

