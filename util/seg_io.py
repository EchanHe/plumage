"""
The 

"""

import json
import os, sys
import numpy as np
from datetime import date

def masks_to_json(masks, path ,names =None):
    """
    Write the mask to json file.

    1 layer: File name and ID
    2 layer: id of class
    3 layer: cols and rows
    4 layer: a list of col and row index

    params:
        masks [batch , height, width, n_mask]
        path: The path for saving the json file
        names: the names of each masks
    """
    
    result = {}
    for i in range(masks.shape[0]):
        mask = masks[i,...]
        if names is None:
            result[i] = mask_to_dict(mask)
        else:
            result[names[i]] = mask_to_dict(mask)

    f = open(path,"w")
    f.write(json.dumps(result))
    f.close()

def mask_to_dict(mask):
    back_ground_id = 0
    
    n_cl = mask.shape[-1]
    result ={}
    all_rows = [0] * n_cl
    all_cols = [0] * n_cl
    for c in range(n_cl):
        if c != back_ground_id:
            rows , cols = np.where(mask[...,c]==1)
            rows = rows.tolist()
            cols = cols.tolist()
            result[c] = {'rows':rows, 'cols':cols}
    return result


def read_masks_from_json(path):
    with open(path, 'r') as f:
        dict = json.load(f)
    return dict

def get_info_from_params_seg(params):
    outputs = {}
    keys = ["name", "category"  , "output_stride","img_aug", "nepochs" ,"batch_size", "learning_rate",
     "decay_step" , "decay_step" ,"lambda_l2"]
    for key in keys:
        assert key in params.keys() , "No this keys, check the config file"
        outputs[key] = params[key]
    return outputs
def write_seg_result(iou, params , folder ,cor_pred = None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    result = {}
    result['config'] = get_info_from_params_seg(params)
    result["miou"] = iou
    result["correct_predict_rate"] = cor_pred

    result_name = "protocol_result_{}_{}.json".format(str(date.today()),params["name"])
    print("write into: ", result_name)
    f = open(folder+result_name,"w")
    f.write(json.dumps(result ,indent=2 ,sort_keys=True))
    f.close()