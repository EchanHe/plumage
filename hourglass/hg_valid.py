"""
TRAIN LAUNCHER 

"""

import configparser
import tensorflow as tf

from hourglass_tiny import HourglassModel

import os
import pandas as pd
from  plumage_util import Accuracy, heatmap_to_coord , write_coord
import sys

def process_config(conf_file):
    """
    """
    params = {}
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(conf_file)
    for section in config.sections():
        if section == 'Directory':
            for option in config.options(section):
                params[option] = config.get(section, option)
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Train':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Validation':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params

print('--Parsing Config File')
params = process_config('config_hg.cfg')


input_lib_dir= os.path.abspath(os.path.join(params['work_dir'],"input"))
sys.path.append(input_lib_dir)
import data_input

rootdir = params['work_dir']
img_path =  params['img_folder']


df_valid = pd.read_csv(params['valid_file'])
df_valid = df_valid[:12]
print("Read valid set data: ...")
valid_data = data_input.plumage_data_input(df_valid,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                         is_train=True , pre_path = img_path,is_aug=params['img_aug'] )

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
load_file = params['restore_param_file']
print("Load model from:",load_file)
heatmaps = model.get_heatmaps(load = load_file)
print("Output heatmaps result. Shape: "+ str(heatmaps.shape))


df_file_names = valid_data.df[valid_data.file_name_col]
gt_coords = valid_data.df[valid_data.coords_cols].as_matrix()
lm_cnt = valid_data.lm_cnt


pred_coord = heatmap_to_coord(heatmaps , valid_data.img_width , valid_data.img_height)
write_coord(pred_coord , gt_coords ,folder=params['valid_result_dir'])

scaled_diff_per_pt ,scaled_pck= Accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , pck_threshold = params['pck_threshold'],scale = 20)
diff_per_pt ,pck= Accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)
print("scaled diff per points\n{}\nscaled pck_{}\n{}".format(scaled_diff_per_pt ,params['pck_threshold'],scaled_pck ))
print("diff per points\n{}\npck_{}\n{}".format(diff_per_pt ,params['pck_threshold'],pck ))