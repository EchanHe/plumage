"""
TRAIN LAUNCHER 

"""

import tensorflow as tf

from hourglass_tiny import HourglassModel

import os
import pandas as pd
import numpy as np

import sys
#Add path of other folder and import files
dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config
from points_io import  write_pred_dataframe
from points_metrics import *
from seg_metrics import segs_eval
from points_util import heatmap_to_coord,pred_coords_to_patches
from visualize_lib import show_markups

print('--Parsing Config File')
params = process_config('config_valid.cfg')


rootdir = params['work_dir']
img_path =  params['img_folder']


df_valid = pd.read_csv(params['valid_file'])

# if params['category'] is not None:
#     params['name'] +='_' + params['category']
#     df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
# df_valid = df_valid[:10]



print("Read valid set data: ...")
valid_data = data_input.plumage_data_input(df_valid,params['batch_size'],scale = 10, state = params['data_state'],
                                         is_train=True , pre_path = img_path,is_aug=params['img_aug'] )

if params['pred_file'] == 'None':
    #Get the name of the checkpoints:
    names = os.listdir(params['saver_directory'])
    if params['category'] is not None:
        load_files = [name for name in names if params['category'] in name]
    else:
        load_files = [name for name in names if 'all' in name]

    load_file = os.path.commonprefix(load_files).split('.')[0]
    load_file = os.path.join(params['saver_directory'] , load_file)

    model = HourglassModel(img_width = params['img_width'],img_height=params['img_height'] ,img_scale = params['scale'],
                             nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                           nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
                           attention = params['mcam'],training=False, drop_rate= params['dropout_rate'],
                           lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],decay_step=params['decay_step'], name=params['name'],
                           data_stream_train=None, data_stream_valid=valid_data, data_stream_test=None,
                           logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],saver_directory=params['saver_directory'],
                           tiny= params['tiny'], w_loss=params['weighted_loss'],w_summary=True , joints= params['joint_list'],modif=False)
    model.generate_model()
    print("Load model from:",load_file)
    heatmaps = model.get_heatmaps(load = load_file)
    print("Output heatmaps result. Shape: "+ str(heatmaps.shape))


    gt_coords = valid_data.df[valid_data.coords_cols].as_matrix()
    pred_coord = heatmap_to_coord(heatmaps , valid_data.img_width , valid_data.img_height)
    
    patch_coord = pred_coords_to_patches(pred_coord)

    # write_coord(pred_coord , gt_coords,params['valid_result_dir'])
    df_pred = write_pred_dataframe(valid_data , pred_coord , params['valid_result_dir'], params['name'] , patch_coord )

    #Scaled version
    # scaled_diff_per_pt ,scaled_pck= pck_accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , pck_threshold = params['pck_threshold'],scale = 20)
    # print("scaled diff per points\n{}\nscaled pck_{}\n{}".format(scaled_diff_per_pt ,params['pck_threshold'],scaled_pck ))
else:
    df_pred = pd.read_csv(params['pred_file'])
    pred_coord = df_pred[valid_data.coords_cols].as_matrix()

    gt_coords = valid_data.df[valid_data.coords_cols].as_matrix()


lm_cnt = valid_data.lm_cnt

#PCK accuracy:
diff_per_pt ,pck= pck_accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)
print("diff per points\n{}\npck_{}\n{}".format(diff_per_pt ,params['pck_threshold'],pck ))


df_valid = df_valid.sample(n=20,random_state=3)
df_pred = df_pred.sample(n=20,random_state=3)
# df_valid = df_valid[0:20]
# df_pred = df_pred[0:20]

valid_data = data_input.plumage_data_input(df_valid, df_valid.shape[0] ,scale = 10, state = 'patches',
                                     is_train=True , pre_path = img_path,is_aug=params['img_aug'] )

pred_data = data_input.plumage_data_input(df_pred, df_pred.shape[0] ,scale = 10, state = 'patches',
                                     is_train=True , pre_path = img_path,is_aug=params['img_aug'] )

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
print("miou", segs_eval(pred_segms , gt_segms , mode="miou"))
print("correct_pred", segs_eval(pred_segms , gt_segms , mode="correct_pred" , background = 0))

print("miou", segs_eval(pred_segms , gt_segms , mode="miou" , per_class=True))
print("correct_pred", segs_eval(pred_segms , gt_segms , mode="correct_pred" , 
    background = 0 , per_class=True))


# for i in range(1, pred_mask.shape[3]):
#     gt_seg = gt_mask[...,i]
#     pred_seg = pred_mask[...,i]

#     print("{} th patch".format(i))
#     print("miou", segs_eval(pred_seg , gt_seg , mode="miou"))
#     print("correct_pred", segs_eval(pred_seg , gt_seg , mode="correct_pred" , background = 0))

#Save images codes
if params['save_img']:
    valid_data = data_input.plumage_data_input(df_valid, 10 ,scale = 1, state = params['data_state'],
                                         is_train=True , pre_path = img_path,is_aug=params['img_aug'] )

    pred_data = data_input.plumage_data_input(df_pred, 10 ,scale = 1, state = params['data_state'],
                                         is_train=True , pre_path = img_path,is_aug=params['img_aug'] )

    for i in range(0, df_valid.shape[0] , 10):
        img, gt_coord_mini, _ , _ = valid_data.get_next_batch_no_random_all_labels()
        _, pred_coord_mini, _ , _ = pred_data.get_next_batch_no_random_all_labels()
        show_markups(img , pred_coords = pred_coord_mini , gt_coords = gt_coord_mini, 
            pck_threshold = params['pck_threshold'],save_path = params['valid_imgs_dir'] + str(i//10 + 1) + "_")