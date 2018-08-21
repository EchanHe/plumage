import numpy as np
import pandas as pd

import os
import json
from datetime import date
from points_metrics import pck_accuracy
### Goal: write coords only in csv
# Params: pred_coords, gt_coords [batch , lm_cnt * 2 ].
def write_coord(pred_coords , gt_coords , folder,file_name = "hg_valid_coords" ):
    pred_file_name = folder + file_name+"_pred.csv"
    gt_file_name = folder +file_name + "_gt.csv"
    print("Save VALID prediction in "+pred_file_name)
    print("save GROUND TRUTH in " + gt_file_name)
    np.savetxt(pred_file_name, pred_coords, delimiter=",")
    np.savetxt(gt_file_name, gt_coords, delimiter=",")

no_back_cols = ['throat_x', 'throat_y', 'breast_x', 'breast_y', 'belly_x',
       'belly_y', 'tail.underside_x', 'tail.underside_y', 'wing.coverts_x',
       'wing.coverts_y', 'wing.primaries.secondaries_x',
       'wing.primaries.secondaries_y',
       'poly.wing.coverts',   'poly.wing.primaries.secondaries',
     'poly.throat', 'poly.breast', 'poly.belly', 'poly.tail.underside']
no_belly_cols = ['crown_x', 'crown_y', 'nape_x',
       'nape_y', 'mantle_x', 'mantle_y', 'rump_x', 'rump_y', 'tail_x',
       'tail_y', 'wing.coverts_x',
       'wing.coverts_y', 'wing.primaries.secondaries_x',
       'wing.primaries.secondaries_y',
       'poly.crown', 'poly.nape','poly.mantle', 'poly.rump', 'poly.tail',
    'poly.wing.coverts',   'poly.wing.primaries.secondaries']
no_side_cols = ['crown_x', 'crown_y', 'nape_x',
       'nape_y', 'mantle_x', 'mantle_y', 'rump_x', 'rump_y', 'tail_x',
       'tail_y', 'throat_x', 'throat_y', 'breast_x', 'breast_y', 'belly_x',
       'belly_y', 'tail.underside_x', 'tail.underside_y',
       'poly.crown', 'poly.nape','poly.mantle', 'poly.rump', 'poly.tail',
       'poly.throat', 'poly.breast', 'poly.belly', 'poly.tail.underside']

patches_cols = ['poly.crown', 'poly.nape','poly.mantle', 'poly.rump', 'poly.tail',
     'poly.throat', 'poly.breast', 'poly.belly', 'poly.tail.underside',
     'poly.wing.coverts',   'poly.wing.primaries.secondaries']

     
def write_pred_dataframe(valid_data , pred_coord , folder,file_name , patches_coord=None ):
    """
    Goal: write prediction coordinates to DATAFRAME csv and return the panda dataframe

    params:
        valid_data: panda dataframe of validation data. used for giving file name and types
        pred_coord: prediction coordiantes shape [batch_size , 2 * landmark]
        folder: folder to save
        file_name: name of the csv file. When is None, function doesn't save the csv
        patches_coord: lists of coodinates for each patch. [batch_size, n patches] (dtype = list)
    """
    df_file_names = valid_data.df[['file.vis', 'view']]
    df_file_names = df_file_names.reset_index()
    result = pd.DataFrame(pred_coord, columns = valid_data.coords_cols )
    if not os.path.exists(folder):
        os.makedirs(folder)


    if patches_coord is None:
        result = pd.concat([df_file_names,result],axis=1)
    else:
        p_result = pd.DataFrame(patches_coord, columns = patches_cols)
        result = pd.concat([df_file_names,result,p_result],axis=1)

    result.loc[result['view']=='back', no_back_cols]=-1
    result.loc[result['view']=='belly', no_belly_cols]=-1
    result.loc[result['view']=='side', no_side_cols]=-1

    if file_name is not None:
        result.to_csv(folder+file_name+".csv" , index =False)
    return result

### Goal: write the evaluation on json
# Params: gt_df: dataframe of ground truth. pred_coords [batch , lm_cnt * 2 ].

def write_point_result(pred_coord , gt_coords,lm_cnt, params, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    pck_threshold = params['pck_threshold']
    diff_per_pt ,pck= pck_accuracy(pred_coord , gt_coords, lm_cnt=5 , 
                                            pck_threshold = params['pck_threshold'],scale = 1)
    diff_per_pt_all ,pck_all= pck_accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , 
                                            pck_threshold = params['pck_threshold'],scale = 1)
    result = {}
    result['config'] = get_info_from_params_points(params)
    result['pck-{}'.format(pck_threshold)] =pck.tolist()
    result['pixel_diff'] = diff_per_pt.tolist()
    result['mean_pck-{}'.format(pck_threshold)] =np.mean(pck)

    result['pck_all-{}'.format(pck_threshold)] =pck_all.tolist()
    result['pixel_diff_all'] = diff_per_pt_all.tolist()
    result['mean_pck_all-{}'.format(pck_threshold)] =np.mean(pck_all)

    result_name = "{}_{}.json".format(str(date.today()) ,params["name"])
    print("write into: ", result_name)
    f = open(folder+result_name,"w")
    f.write(json.dumps(result ,indent=2 , sort_keys=True))
    f.close()
### Goal: Get the dictionray from params.
# Used in write_point_result
def get_info_from_params_points(params):
    outputs = {}
    keys = ["name", "category" ,"img_aug", "nepochs" , "learning_rate","batch_size",
     "decay_step" , "decay_step" , "dropout_rate" , "nlow" ,"nstacks"]
    for key in keys:
        assert key in params.keys() , "No this keys, check the config file"
        outputs[key] = params[key]
    return outputs