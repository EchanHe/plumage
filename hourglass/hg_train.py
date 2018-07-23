"""
TRAIN LAUNCHER 

"""

import configparser
import tensorflow as tf

from hourglass_tiny import HourglassModel

import os
import pandas as pd
import sys
# import hg_data_input



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


print("Read training set data: ...")
rootdir = params['work_dir']


df_train = pd.read_csv(params['train_file'])
df_valid = pd.read_csv(params['valid_file'])

if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_train = df_train.loc[df_train.view==params["category"],:].reset_index(drop = True)
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)
# df=df[:1]

df_train=df_train[:1]
input_data = data_input.plumage_data_input(df_train,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                         is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )
df_valid = df_valid[:1]
print("Read valid set data: ...")
# df_valid = df_valid[:1]
valid_data = data_input.plumage_data_input(df_valid,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                         is_train=True , pre_path = params['img_folder'],is_aug=params['img_aug'] )

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
                       attention = params['mcam'],training=True, drop_rate= params['dropout_rate'],
                       lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],decay_step=params['decay_step'], name=params['name'],
                       data_stream_train=input_data, data_stream_valid=valid_data, data_stream_test=None,
                       logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],saver_directory=params['saver_directory'],
                       tiny= params['tiny'], w_loss=params['weighted_loss'],w_summary=True , joints= params['joint_list'],modif=False)
model.generate_model()
load_file = None
model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], load = load_file)