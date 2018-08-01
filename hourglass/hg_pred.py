"""
TRAIN LAUNCHER 

"""

import tensorflow as tf

from hourglass_tiny import HourglassModel

import os
import pandas as pd

import sys
#Add path of other folder and import files
dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config
from points_io import  write_pred_dataframe , write_coord
from points_metrics import pck_accuracy
from points_util import heatmap_to_coord,pred_coords_to_patches

print('--Parsing Config File')
params = process_config('config_pred.cfg')


rootdir = params['work_dir']
img_path =  params['img_folder']


df_valid = pd.read_csv(params['valid_file'])
if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
# df_valid = df_valid[:10]
print("Read valid set data: ...", img_path)
valid_data = data_input.plumage_data_input(df_valid,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                         is_train=False , pre_path = img_path,is_aug=params['img_aug'] )

#Get the name of the checkpoints:
names = os.listdir(params['saver_directory'])
if params['category'] is not None:
    load_files = [name for name in names if params['category'] in name]
else:
    load_files = [name for name in names if 'all' in name]

load_file = os.path.commonprefix(load_files).split('.')[0]
load_file = os.path.join(params['saver_directory'] , load_file)
#testfile
if params['test_file'] is not None:
    pred_file = os.path.join(rootdir , params['test_file'])
    img_path = os.path.join(rootdir ,params['test_img_folder'])
    df_test = pd.read_csv(pred_file)
    # df_valid = df_valid[:1]
    print("Read test set data: ...")
    test_data = hg_data_input.hg_data_input(df_test,params['batch_size'],scale = params['scale'],
                                             is_train=False , pre_path = img_path,is_aug=params['img_aug'] )


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

pred_coord = heatmap_to_coord(heatmaps , valid_data.img_width , valid_data.img_height)
patch_coord = pred_coords_to_patches(pred_coord)

# write_coord(pred_coord , gt_coords,params['valid_result_dir'])
write_pred_dataframe(valid_data , pred_coord , params['valid_result_dir'], params['name'] , patch_coord )

