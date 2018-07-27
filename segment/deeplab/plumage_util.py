import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
import os
import cv2

import json
from datetime import date


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

    result_name = "{}_{}.json".format(str(date.today()),params["name"])
    print("write into: ", result_name)
    f = open(folder+result_name,"w")
    f.write(json.dumps(result ,indent=2))
    f.close()