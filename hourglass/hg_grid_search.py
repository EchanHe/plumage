"""
TRAIN LAUNCHER 

"""

import tensorflow as tf

from hourglass_tiny import HourglassModel

import os
import pandas as pd
import sys
import itertools
from datetime import date
# import hg_data_input and other library
dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import *
from points_io import write_point_result ,write_pred_dataframe
from points_util import heatmap_to_coord , pred_coords_to_patches
from points_metrics import *
from seg_metrics import segs_eval
#read the params..
print('--Parsing Config File')
config_name = 'config_grid.cfg'

params = process_config(os.path.join(dirname, config_name))


grid_params = generate_grid_params(params)

print(grid_params)

# Get the name of the parameters
if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
elif params['category'] is None or params['category'] =='all':
    params['name'] +='_' + 'all'
ori_name = params['name']
if bool(grid_params):

    keys, values = zip(*grid_params.items())

    df_lists= [] 
    # v_result_matrix = np.zeros(())

    for v_pert in itertools.product(*values):
        #Updates the values of parameters each time to do a grid search training.
        col_name = ""
        for key, value in zip(keys, v_pert):
            params[key] = value
            col_name +=  "_{}_{}".format(key,value)

        print(col_name)
        params['name'] = ori_name + col_name
        print([params[k] for k in keys])

        # train process

        print("Read training set data: ...")

        df_train = pd.read_csv(params['train_file'])
        df_valid = pd.read_csv(params['valid_file'])

        # sampling the training data for faster testing.
        # df_train = df_train.sample(n=20,random_state=3)
        # df_valid = df_valid.sample(n=5,random_state=3)

        # df=df[:1]

        input_data = data_input.plumage_data_input(df_train,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                                 is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )
        print("Read valid set data: ...")
        # df_valid = df_valid[:10]
        valid_data = data_input.plumage_data_input(df_valid,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                                 is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )
        
        epochSize = input_data.df_size // params["batch_size"]
        total_steps = (epochSize * params['nepochs']) //params["batch_size"]
        summary_steps = epochSize // params['summary_interval']


        img_height = input_data.img_height // input_data.scale
        img_width = input_data.img_width // input_data.scale
        tf.reset_default_graph()
        model = HourglassModel(img_width = img_width,img_height=img_height,img_scale = params['scale'],
                                 nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                               nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
                               attention = params['mcam'],training=True, drop_rate= params['dropout_rate'],
                               lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],decay_step=params['decay_step'], name=params['name'],
                               data_stream_train=input_data, data_stream_valid=valid_data, data_stream_test=None, is_grey = params['is_grey'],
                               logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],saver_directory=params['saver_directory'],
                               tiny= params['tiny'], w_loss=params['weighted_loss'],w_summary=True , joints= params['joint_list'],modif=False)
        model.generate_model()
        load_file = None
        print("epochsize: {}".format(epochSize))
        model.training_init(nEpochs=params['nepochs'], epochSize=epochSize, saveStep=summary_steps, load = load_file)

        heatmaps = model.get_heatmaps(load = load_file)
        model.Session.close()

        print("Output heatmaps result. Shape: "+ str(heatmaps.shape))


        df_file_names = valid_data.df[valid_data.file_name_col]
        gt_coords = valid_data.df[valid_data.coords_cols].as_matrix()
        lm_cnt = valid_data.lm_cnt


        pred_coord = heatmap_to_coord(heatmaps , valid_data.img_width , valid_data.img_height)
        patch_coord = pred_coords_to_patches(pred_coord)

        df_pred = write_pred_dataframe(valid_data , pred_coord , 
            params['valid_result_dir'], file_name = None , patches_coord= patch_coord )


        ##### Accuracy test:
                #PCK accuracy:
        diff_per_pt ,pck= pck_accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)


        valid_data = data_input.plumage_data_input(df_valid, df_valid.shape[0] ,scale = 10, state = 'patches',
                                             is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )

        pred_data = data_input.plumage_data_input(df_pred, df_pred.shape[0] ,scale = 10, state = 'patches',
                                             is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )

        _, gt_mask = valid_data.get_next_batch_no_random_all()
        _, pred_mask = pred_data.get_next_batch_no_random_all()

        #Inside polygon metrics
        _, gt_centroid, _ , _ = valid_data.get_next_batch_no_random_all_labels()
        _, pred_centroid, _ , _ = pred_data.get_next_batch_no_random_all_labels()

        #Only use the centroids of polygons
        pred_centroid = pred_centroid[:,10:]
        gt_centroid = gt_centroid[:,10:]
        gt_patches = gt_mask[...,1:]

        #Check whether centroids are inside patch 
        in_area = in_area_rate(pred_centroid ,gt_centroid, gt_patches)

        # Check mean iou and correct prediction

        gt_segms = np.argmax(gt_mask, -1)
        pred_segms = np.argmax(pred_mask, -1)

        miou = segs_eval(pred_segms , gt_segms , mode="miou")
        m_precision =  segs_eval(pred_segms , gt_segms , mode="correct_pred" , background = 0)

        iou =  segs_eval(pred_segms , gt_segms , mode="miou" , per_class=True)
        precision =  segs_eval(pred_segms , gt_segms , mode="correct_pred" , background = 0 , per_class=True)

        accuracy = {'pck{}'.format(params['pck_threshold']) : pck,
                    'diff_per_pt': diff_per_pt,
                    'in_poly': in_area,
                    'mean_pck': np.nanmean(pck),
                    'mean_diff_per_pt': np.nanmean(diff_per_pt),
                    'mean_in_poly': np.nanmean(in_area),
                    'iou': iou,
                    'precision': precision,
                    'mean_iou': miou,
                    'mean_precision': m_precision,
                    'running time':model.time_segments
                     }
        df_accuracy = pd.DataFrame.from_dict(accuracy, orient='index')      
        df_accuracy.columns = [col_name] 
        print(df_accuracy.iloc[:,0]) 
        df_lists.append(df_accuracy)
pd.concat(df_lists,axis = 1).to_csv(params['valid_result_dir'] + "{}grid_search.csv".format(str(date.today())))


#generate the dictionary for grid search

#try different configurations

#   training models

#   write validation results in one csv (times, precision).


# print("Read training set data: ...")
# rootdir = params['work_dir']


# df_train = pd.read_csv(params['train_file'])
# df_valid = pd.read_csv(params['valid_file'])

# if params['category'] is not None:
#     params['name'] +='_' + params['category']
#     df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
#     df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
# elif params['category'] is None or params['category'] =='all':
#     params['name'] +='_' + 'all'
# # df=df[:1]

# # df_train=df_train[:25]
# input_data = data_input.plumage_data_input(df_train,params['batch_size'],scale = params['scale'], state = params['data_state'],
#                                          is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )
# # df_valid = df_valid[:1]
# print("Read valid set data: ...")
# # df_valid = df_valid[:10]
# valid_data = data_input.plumage_data_input(df_valid,params['batch_size'],scale = params['scale'], state = params['data_state'],
#                                          is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )
# epochSize = input_data.df_size
# total_steps = (epochSize * params['nepochs']) //params["batch_size"]
# summary_steps = total_steps // params['summary_interval']


# model = HourglassModel(img_width = params['img_width'],img_height=params['img_height'] ,img_scale = params['scale'],
#                          nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
#                        nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
#                        attention = params['mcam'],training=True, drop_rate= params['dropout_rate'],
#                        lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],decay_step=params['decay_step'], name=params['name'],
#                        data_stream_train=input_data, data_stream_valid=valid_data, data_stream_test=None, is_grey = params['is_grey'],
#                        logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],saver_directory=params['saver_directory'],
#                        tiny= params['tiny'], w_loss=params['weighted_loss'],w_summary=True , joints= params['joint_list'],modif=False)
# model.generate_model()
# load_file = None
# model.training_init(nEpochs=params['nepochs'], epochSize=total_steps, saveStep=summary_steps, load = load_file)


# heatmaps = model.get_heatmaps(load = load_file)
# print("Output heatmaps result. Shape: "+ str(heatmaps.shape))


# df_file_names = valid_data.df[valid_data.file_name_col]
# gt_coords = valid_data.df[valid_data.coords_cols].as_matrix()
# lm_cnt = valid_data.lm_cnt


# pred_coord = heatmap_to_coord(heatmaps , valid_data.img_width , valid_data.img_height)

# write_point_result(pred_coord , gt_coords ,lm_cnt , params , params['valid_result_dir'])