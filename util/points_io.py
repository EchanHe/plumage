import numpy as np
import pandas as pd
import cv2
import os
     
def write_pred_dataframe(valid_data , pred_coord , folder,file_name , file_col_name ,
 patches_coord=None, write_index = False , is_valid = True ):
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
    df_file_names = valid_data.df.drop(valid_data.coords_cols , axis=1 , errors = 'ignore')
    df_file_names = df_file_names.reset_index(drop=True)
    result = pd.DataFrame(pred_coord, columns = valid_data.coords_cols )
    if not os.path.exists(folder):
        os.makedirs(folder)
  
    if is_valid:
        gt_coords = valid_data.df[valid_data.coords_cols].values
        pred_coord[gt_coords==-1] = -1
    
    # Write the polygons in if there is given patches_coord
    # Other wise assign all -1.

    # if patches_coord is None:
    #     patches_coord = np.ones((result.shape[0], len(valid_data.patches_cols))) * -1    
    # p_result = pd.DataFrame(patches_coord, columns = valid_data.patches_cols)

    result = pd.concat([df_file_names,result],axis=1)

    if file_name is not None:
        result.to_csv(folder+file_name+".csv" , index =write_index)
    return result

def build_result_dict(result_dict= {},name = None,
    pck = None, mean_pck = None, pck_threshold = None,
    diff_per_pt=None, mean_diff_per_pt = None,
    in_poly = None, mean_in_poly = None,
    iou = None, mean_iou = None,
    precision = None, mean_precision = None,
    pck_50 = None, pck_150 = None, pck_200 = None, pck_300 = None ):
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

    result_dict['pck_50'] = pck_50
    result_dict['pck_150'] = pck_150
    result_dict['pck_200'] = pck_200
    result_dict['pck_300'] = pck_300
    result_dict = {str(k):str(v) for k,v in result_dict.items()}
    return result_dict

### Heatmap functions:
def plot_heatmaps(dir, heatmaps, img, file_name,  names = None, img_per_row = 5):
    if not os.path.exists(dir):
        os.makedirs(dir)

    results = np.array([], dtype=np.int64).reshape(0, img.shape[1] * img_per_row,3)
    row = np.array([], dtype=np.int64).reshape(img.shape[0],0,3)

    for i in range(heatmaps.shape[-1]):
        # Fore every pic

        heatmap = heatmaps[...,i:i+1]
        # print(np.min(img_temp),np.max(img_temp))
        heatmap = np.interp(heatmap,[np.min(heatmap),np.max(heatmap)],[0,255]).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

        heatmap = cv2.resize(heatmap, dsize=(img.shape[1], img.shape[0]),
            interpolation = cv2.INTER_NEAREST)

        dst = cv2.addWeighted(img,0.5,heatmap,0.5,0)
        if names is not None:
            cv2.putText(dst, names[i], (50,50),
             cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)

        if (i+1) % img_per_row !=0:
            row = np.hstack((row, dst))
        else:
            row = np.hstack((row, dst))
            results = np.vstack((results, row))
            row = np.array([], dtype=np.int64).reshape(img.shape[0],0,3)
    if heatmaps.shape[-1] % img_per_row !=0:      
        offset = (img_per_row-heatmaps.shape[-1] % img_per_row)
        blank = np.zeros((img.shape[0] , img.shape[1] * offset  , 3)).astype(np.uint8)

        row = np.hstack((row , blank))
        results = np.vstack((results, row))
          
    cv2.imwrite( os.path.join(dir, "{}.png".format(file_name)),results)



def save_heatmaps(dir, heatmaps, file_name,  pt_names):
    dir = os.path.join(dir, file_name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(heatmaps.shape[-1]):
        # Fore every pic
        heatmap = heatmaps[...,i:i+1]
        heatmap = np.interp(heatmap,[np.min(heatmap),np.max(heatmap)],[0,255]).astype(np.uint8)
        cv2.imwrite( os.path.join(dir, "{}.jpg".format(pt_names[i])),heatmap)


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

    print("deprecated")
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # pck_threshold = params['pck_threshold']
    # diff_per_pt ,pck= pck_accuracy(pred_coord , gt_coords, lm_cnt=5 , 
    #                                         pck_threshold = params['pck_threshold'],scale = 1)
    # diff_per_pt_all ,pck_all= pck_accuracy(pred_coord , gt_coords, lm_cnt=lm_cnt , 
    #                                         pck_threshold = params['pck_threshold'],scale = 1)
    # result = {}
    # result['config'] = get_info_from_params_points(params)
    # result['pck-{}'.format(pck_threshold)] =pck.tolist()
    # result['pixel_diff'] = diff_per_pt.tolist()
    # result['mean_pck-{}'.format(pck_threshold)] =np.mean(pck)

    # result['pck_all-{}'.format(pck_threshold)] =pck_all.tolist()
    # result['pixel_diff_all'] = diff_per_pt_all.tolist()
    # result['mean_pck_all-{}'.format(pck_threshold)] =np.mean(pck_all)

    # result_name = "{}_{}.json".format(str(date.today()) ,params["name"])
    # print("write into: ", result_name)
    # f = open(folder+result_name,"w")
    # f.write(json.dumps(result ,indent=2 , sort_keys=True))
    # f.close()

### Goal: Get the dictionray from params.
# Used in write_point_result
def get_info_from_params_points(params):
    """
    Goal: Get useful properties of the config dictionray.
    """
    print("deprecated")

    # outputs = {}
    # keys = ["name", "category" ,"img_aug", "nepochs" , "learning_rate","batch_size",
    #  "decay_step" , "decay_step" , "dropout_rate" , "nlow" ,"nstacks"]
    # for key in keys:
    #     assert key in params.keys() , "No this keys, check the config file"
    #     outputs[key] = params[key]
    # return outputs



def write_coord(pred_coords , gt_coords , folder,file_name = "hg_valid_coords" ):
    """
    deprecated
    Goal: write coords only in csv
    """
    print("deprecated")
    # pred_file_name = folder + file_name+"_pred.csv"
    # gt_file_name = folder +file_name + "_gt.csv"
    # print("Save VALID prediction in "+pred_file_name)
    # print("save GROUND TRUTH in " + gt_file_name)
    # np.savetxt(pred_file_name, pred_coords, delimiter=",")
    # np.savetxt(gt_file_name, gt_coords, delimiter=",")


