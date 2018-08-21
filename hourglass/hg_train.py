"""
TRAIN LAUNCHER 

"""


import tensorflow as tf

from hourglass_tiny import HourglassModel

import os
import pandas as pd
import sys
# import hg_data_input
dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config , save_config
from points_io import write_point_result
from points_util import heatmap_to_coord


print('--Parsing Config File')
config_name = 'config_hg.cfg'

params = process_config(os.path.join(dirname, config_name))
# save_config(os.path.join(dirname, config_name) , params['saver_directory'])


print("Read training set data: ...")

df_train = pd.read_csv(params['train_file'])
df_valid = pd.read_csv(params['valid_file'])

if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
elif params['category'] is None or params['category'] =='all':
    params['name'] +='_' + 'all'
# df=df[:1]

# df_train=df_train[:25]
input_data = data_input.plumage_data_input(df_train,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                         is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )
# df_valid = df_valid[:1]
print("Read valid set data: ...")
# df_valid = df_valid[:10]
valid_data = data_input.plumage_data_input(df_valid,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                         is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )

epochSize = input_data.df_size // params["batch_size"]
total_steps = (epochSize * params['nepochs']) //params["batch_size"]
summary_steps = epochSize // params['summary_interval']


model = HourglassModel(img_width = params['img_width'],img_height=params['img_height'] ,img_scale = params['scale'],
                         nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                       nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
                       attention = params['mcam'],training=True, drop_rate= params['dropout_rate'],
                       lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],decay_step=params['decay_step'], name=params['name'],
                       data_stream_train=input_data, data_stream_valid=valid_data, data_stream_test=None,
                       logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],saver_directory=params['saver_directory'],
                       tiny= params['tiny'], w_loss=params['weighted_loss'],w_summary=True , joints= params['joint_list'],modif=False)
model.generate_model()
load_file = None
model.training_init(nEpochs=params['nepochs'], epochSize=epochSize, saveStep=summary_steps, load = load_file)


heatmaps = model.get_heatmaps(load = load_file)
print("Output heatmaps result. Shape: "+ str(heatmaps.shape))


df_file_names = valid_data.df[valid_data.file_name_col]
gt_coords = valid_data.df[valid_data.coords_cols].as_matrix()
lm_cnt = valid_data.lm_cnt


pred_coord = heatmap_to_coord(heatmaps , valid_data.img_width , valid_data.img_height)

write_point_result(pred_coord , gt_coords ,lm_cnt , params , params['valid_result_dir'])