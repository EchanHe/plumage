"""
TRAIN LAUNCHER 

"""

import tensorflow as tf

from hourglass_tiny import HourglassModel

import os
import pandas as pd
from  plumage_util import Accuracy, heatmap_to_coord , write_coord
import sys
#Add path of other folder and import files
dirname = os.path.dirname(__file__)
input_lib_dir= os.path.abspath(os.path.join(dirname,"../input"))
util_lib_dir= os.path.abspath(os.path.join(dirname,"../util"))
sys.path.append(input_lib_dir)
sys.path.append(util_lib_dir)
import data_input
from plumage_config import process_config



print('--Parsing Config File')
params = process_config('config_valid.cfg')


rootdir = params['work_dir']
img_path =  params['img_folder']


df_valid = pd.read_csv(params['valid_file'])
if params['category'] is not None:
    params['name'] +='_' + params['category']
    df_valid = df_valid.loc[df_valid.view==params["category"],:].reset_index(drop = True)

print("Read valid set data: ...")
valid_data = data_input.plumage_data_input(df_valid,params['batch_size'],scale = params['scale'], state = params['data_state'],
                                         is_train=True , pre_path = img_path,is_aug=params['img_aug'] )

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


df_file_names = valid_data.df[valid_data.file_name_col]
gt_coords = valid_data.df[valid_data.coords_cols].as_matrix()
lm_cnt = valid_data.lm_cnt


pred_coord = heatmap_to_coord(heatmaps , valid_data.img_width , valid_data.img_height)
# write_coord(pred_coord , gt_coords ,folder=params['valid_result_dir']
result = pd.DataFrame(pred_coord ,index =df_file_names, columns = valid_data.coords_cols )
if not os.path.exists(params['valid_result_dir']):
    os.makedirs(params['valid_result_dir'])

result.to_csv(params['valid_result_dir']+params['name']+".csv")


scaled_diff_per_pt ,scaled_pck= Accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , pck_threshold = params['pck_threshold'],scale = 20)
diff_per_pt ,pck= Accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , pck_threshold = params['pck_threshold'],scale = 1)
print("scaled diff per points\n{}\nscaled pck_{}\n{}".format(scaled_diff_per_pt ,params['pck_threshold'],scaled_pck ))
print("diff per points\n{}\npck_{}\n{}".format(diff_per_pt ,params['pck_threshold'],pck ))