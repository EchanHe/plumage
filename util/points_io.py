import numpy as np
import pandas as pd

import os
import json
from datetime import date
from points_metrics import pck_accuracy

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


     
def write_pred_dataframe(valid_data , pred_coord , folder,file_name , patches_coord=None, write_index = False ):
    """
    Goal: write prediction coordinates to DATAFRAME csv and return the panda dataframe

    params:
        valid_data: panda dataframe of validation data. used for giving file name and types
        pred_coord: prediction coordiantes shape [batch_size , 2 * landmark]
        folder: folder to save
        file_name: name of the csv file. When is None, function doesn't save the csv
        patches_coord: lists of coodinates for each patch. [batch_size, n patches] (dtype = list)
    """
    # Get the name and view from Valid data
    df_file_names = valid_data.df[['file.vis', 'view']]
    df_file_names = df_file_names.reset_index(drop=True)
    result = pd.DataFrame(pred_coord, columns = valid_data.coords_cols )
    if not os.path.exists(folder):
        os.makedirs(folder)

    gt_coords = valid_data.df[valid_data.coords_cols].values
    pred_coord[gt_coords==-1] = -1
    # Write the polygons in if there is given patches_coord
    # Other wise assign all -1.
    if patches_coord is None:
        patches_coord = np.ones((result.shape[0], len(valid_data.patches_cols))) * -1
    
    p_result = pd.DataFrame(patches_coord, columns = valid_data.patches_cols)
    result = pd.concat([df_file_names,result,p_result],axis=1)

    # result.loc[result['view']=='back', np.intersect1d(no_back_cols, result.columns)]=-1
    # result.loc[result['view']=='belly', np.intersect1d(no_belly_cols, result.columns)]=-1
    # result.loc[result['view']=='side', np.intersect1d(no_side_cols, result.columns)]=-1

    if file_name is not None:
        result.to_csv(folder+file_name+".csv" , index =write_index)
    return result

def build_result_dict(result_dict= {},name = None,
    pck = None, mean_pck = None, pck_threshold = None,
    diff_per_pt=None, mean_diff_per_pt = None,
    in_poly = None, mean_in_poly = None,
    iou = None, mean_iou = None,
    precision = None, mean_precision = None ):
    """
    Goals: write value into dictionry, the default value is None
        which the dict can be used into grid searching result.
    """
    remove_keys = ['restore_param_file', 'config_name']
    for remove_key in remove_keys:
        if remove_key in result_dict.keys():
            result_dict.pop(remove_key)

    result_dict['name'] = name
    result_dict['pck{}'.format(pck_threshold)] = pck
    result_dict['mean_pck'] = mean_pck
    result_dict['diff_per_pt'] = diff_per_pt
    result_dict['mean_diff_per_pt'] = mean_diff_per_pt
    result_dict['in_poly'] = in_poly
    result_dict['mean_in_poly'] = mean_in_poly
    result_dict['iou'] = iou
    result_dict['mean_iou'] = mean_iou
    result_dict['precision'] = precision
    result_dict['mean_precision'] = mean_precision
    result_dict = {str(k):str(v) for k,v in result_dict.items()}
    return result_dict

### Deprecate  ####

def write_point_result(pred_coord , gt_coords,lm_cnt, params, folder):
    """
    Goal: write the evaluation (PCK, pixel difference) on json

    params:
        pred_coords: predicted coordinates, shape:[batch , lm_cnt * 2]
        gt_coords: ground-truth coordinates. shape:[batch , lm_cnt * 2]
        lm_cnt: landmark count
        params: hyperparams and config dictionary
        folder: dir for saving the json
    """


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
    """
    Goal: Get useful properties of the config dictionray.
    """

    outputs = {}
    keys = ["name", "category" ,"img_aug", "nepochs" , "learning_rate","batch_size",
     "decay_step" , "decay_step" , "dropout_rate" , "nlow" ,"nstacks"]
    for key in keys:
        assert key in params.keys() , "No this keys, check the config file"
        outputs[key] = params[key]
    return outputs



def write_coord(pred_coords , gt_coords , folder,file_name = "hg_valid_coords" ):
    """
    deprecated
    Goal: write coords only in csv
    """
    pred_file_name = folder + file_name+"_pred.csv"
    gt_file_name = folder +file_name + "_gt.csv"
    print("Save VALID prediction in "+pred_file_name)
    print("save GROUND TRUTH in " + gt_file_name)
    np.savetxt(pred_file_name, pred_coords, delimiter=",")
    np.savetxt(gt_file_name, gt_coords, delimiter=",")


