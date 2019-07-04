import numpy as np
from points_util import create_rect_on_coords_fixed_length , create_rect_on_coords_proportion_length 
import cv2
from scipy.stats import pearsonr
import collections
def pck_accuracy(pred_coords, gt_coords , lm_cnt , pck_threshold, scale = 1 ):
    """
    Goal: calculate different metrics between prediction and gt
    
    params:
        pred_coords: prediction points [batch , landmark *2 ]. eg [x1,y1, ... , xn, yn]
        gt_coords: ground truth points [batch , landmark *2 ]. eg [x1,y1, ... , xn, yn]
        lm_cnt: the landmark count
        pck_threshold: The pck threshold
        scale: scale on coords.
    return:
        pixel difference per key point 
        pck with threshold
 
    """
    df_size = gt_coords.shape[0]
    gt_coords = gt_coords.astype(float)
    gt_coords[gt_coords == -1] = np.nan
    coord_diff = np.ones((df_size,lm_cnt))
    for j in range(lm_cnt):   
        coord_diff[:,j] = np.linalg.norm(pred_coords[:,j*2:j*2+2] //scale - gt_coords[:,j*2:j*2+2] //scale , axis=1)
    acc_per_row = np.nanmean( coord_diff , axis=0)


    pck = np.zeros(lm_cnt)
    for j in range(lm_cnt):
        per_keypoint = coord_diff[:,j]
        non_nan_mask = ~np.isnan(per_keypoint)
        pck[j] = np.sum(per_keypoint[non_nan_mask]<pck_threshold) / per_keypoint[non_nan_mask].size
    return  np.round(acc_per_row , 4) , pck


def in_area_rate(pred_coords ,gt_coords, masks):
    """
    Goals: check whether the giving points
    params:
        pred_coords: prediction points [batch , landmark *2 ]. eg [x1,y1, ... , xn, yn]
        gt_coords: ground truth points [batch , landmark *2 ]. eg [x1,y1, ... , xn, yn]
        masks: The area that for check whether the poins are inside (batch, height, width, n).
    """


    df_size = masks.shape[0]
    lm_cnt =  masks.shape[-1]

    result = np.zeros((df_size, lm_cnt))
    for row in range(df_size):
        for lm in range(lm_cnt):
            if gt_coords[row, lm*2] ==-1:
                result[row, lm] = np.nan
            else:
                x = int(pred_coords[row, lm*2]) -1
                y = int(pred_coords[row, lm*2 + 1]) -1
                mask = masks[row,:,:,lm]
                result[row, lm] = (mask[y,x] == 1)
    # print(np.sum(result==True )/result.size)
    # print(np.nanmean(result , axis = 0))
    return np.nanmean(result , axis = 0)


def pixel_difference(gt_coords , pred_coords, files_names , img_folder ):
    # length_choices = np.arange(10,150,50)
    length_choices = [10,50,100,150,200]
    all_length_gt_patches = {}
    all_length_pred_patches = {}

    all_length_gt_pixel = {}
    all_length_pred_pixel = {}

    pred_coords[gt_coords==-1]=-1
    for idx,length in enumerate(length_choices):
        all_length_gt_patches[str(length)] = create_rect_on_coords(gt_coords, length,length)
        all_length_pred_patches[str(length)] = create_rect_on_coords(pred_coords, length,length)
        
        num_of_pats = all_length_pred_patches[str(length)].shape[-1]
        
        all_length_gt_pixel[str(length)] = [[]] * num_of_pats
        all_length_pred_pixel[str(length)] =[[]] * num_of_pats

    final_result = {}    

    for batch_idx in range(gt_coords.shape[0]):
        # Read img for each file names
        filename = img_folder + files_names[batch_idx]
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        
        for key in all_length_gt_patches:
            
            all_pred_patches = all_length_pred_patches[key]
            pred_patches = all_pred_patches[batch_idx]
            for idx, pat in enumerate(pred_patches):
                if pat !=-1:
                    rect_coords = eval('[' + pat +']')
                    x = np.arange(rect_coords[0],rect_coords[1])
                    y = np.arange(rect_coords[2],rect_coords[3])
                    pixels = img[np.repeat(y,len(x)) , np.tile(x, len(y)),:]
                    mean_pixel = list(np.round(np.mean(pixels,axis = 0)))
                    
                    all_length_pred_pixel[key][idx] = all_length_pred_pixel[key][idx] + (mean_pixel)

            all_gt_patches = all_length_gt_patches[key]
            gt_patches = all_gt_patches[batch_idx]
            for idx, pat in enumerate(gt_patches):
                if pat !=-1:
                    rect_coords = eval('[' + pat +']')
                    x = np.arange(rect_coords[0],rect_coords[1])
                    y = np.arange(rect_coords[2],rect_coords[3])
                    pixels = img[np.repeat(y,len(x)) , np.tile(x, len(y)),:]

                    mean_pixel = list(np.round(np.mean(pixels,axis = 0)))
                    all_length_gt_pixel[key][idx] = all_length_gt_pixel[key][idx] + (mean_pixel)
            
            difference = np.array([])
            for gt,pred in zip(all_length_gt_pixel[key], all_length_pred_pixel[key]):
    #             print("length of patches: ", key)
    #             print(len(gt),len(pred))
    #             print("correlation: ", pearsonr(gt,pred))    
                result = np.absolute(np.subtract(gt, pred))
    #             print("pixel difference: ", np.mean(result))
                difference = np.append(difference, round(np.nanmean(result),1))
            final_result["bbox_"+key] = difference

    return final_result


def pixel_difference_single(gt_coords , pred_coords,length, proportion, files_names , img_folder , height=None ):
    if height is None:
        height = length

    result_key = "Width:{}_Height:{}_Proportion:{}".format(length , height , proportion)

    if proportion is None:
        all_gt_patches = create_rect_on_coords_fixed_length(gt_coords, length,height)
        all_pred_patches = create_rect_on_coords_fixed_length(pred_coords,length,height)
    else:
        all_gt_patches = create_rect_on_coords_proportion_length(gt_coords , width =length, height=height , proportion = proportion)
        all_pred_patches = create_rect_on_coords_proportion_length(pred_coords , width =length, height=height , proportion = proportion)

    pixel_per_patch = [[]] * all_pred_patches.shape[-1]
    gt_pixel_per_patch = [[]] * all_pred_patches.shape[-1]
    names_list = np.array([])
    
    final_result = collections.defaultdict(dict)

    for batch_idx in range(gt_coords.shape[0]):
        filename = img_folder + files_names[batch_idx]
        names_list = np.append(names_list,files_names[batch_idx])
        
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        
        # get the right patches for the image.
        pred_patches = all_pred_patches[batch_idx]
        for idx, pat in enumerate(pred_patches):
            if pat !=-1:
                rect_coords = eval('[' + pat +']')
                x_min = max(0,rect_coords[0])
                x_max = min(img.shape[1]-1, rect_coords[1])

                y_min = max(0,rect_coords[2])
                y_max = min(img.shape[0]-1, rect_coords[3])

                x = np.arange(x_min, x_max, dtype = int)
                y = np.arange(y_min, y_max, dtype = int)

                pixels = img[np.repeat(y,len(x)) , np.tile(x, len(y)),:]

                mean_pixel = list(np.round(np.nanmean(pixels,axis = 0)))
                pixel_per_patch[idx] = pixel_per_patch[idx] + (mean_pixel)
        
        gt_patches = all_gt_patches[batch_idx]
        for idx, pat in enumerate(gt_patches):
            if pat !=-1:
                rect_coords = eval('[' + pat +']')
                x_min = max(0,rect_coords[0])
                x_max = min(img.shape[1]-1, rect_coords[1])

                y_min = max(0,rect_coords[2])
                y_max = min(img.shape[0]-1, rect_coords[3])

                x = np.arange(x_min, x_max, dtype = int)
                y = np.arange(y_min, y_max, dtype = int)

                pixels = img[np.repeat(y,len(x)) , np.tile(x, len(y)),:]

                mean_pixel = list(np.round(np.nanmean(pixels,axis = 0)))
                gt_pixel_per_patch[idx] = gt_pixel_per_patch[idx] + (mean_pixel)
    
    
    difference = np.array([])
    coeff_list = np.array([])
    p_value_list = np.array([])

    for gt,pred in zip(gt_pixel_per_patch, pixel_per_patch):
        result = np.absolute(np.subtract(gt, pred))
#             print("pixel difference: ", np.mean(result))
        coeff, p_value = pearsonr(gt,pred)
        coeff_list = np.append(coeff_list, coeff)
        p_value_list = np.append(p_value_list , p_value)
        difference = np.append(difference, round(np.nanmean(result),1))


    final_result[result_key]['diff'] = difference
    final_result[result_key]['coeff'] = coeff_list
    final_result[result_key]['p_value'] = p_value_list
    final_result[result_key]['gt_pixel_patches'] = gt_pixel_per_patch
    final_result[result_key]['pred_pixel_patches'] = pixel_per_patch
    final_result[result_key]['file_name'] = names_list
    return final_result




## deprecated ##


def pixel_difference_single_length(gt_coords , pred_coords,length, files_names , img_folder , height=None):
    if height is None:
        height = length
    all_gt_patches = create_rect_on_coords(gt_coords, length,height)
    all_pred_patches = create_rect_on_coords(pred_coords,length,height)


    pixel_per_patch = [[]] * all_pred_patches.shape[-1]
    gt_pixel_per_patch = [[]] * all_pred_patches.shape[-1]
    final_result = collections.defaultdict(dict)

    for batch_idx in range(gt_coords.shape[0]):
        filename = img_folder + files_names[batch_idx]
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        
        # get the right patches for the image.
        pred_patches = all_pred_patches[batch_idx]
        for idx, pat in enumerate(pred_patches):
            if pat !=-1:
                rect_coords = eval('[' + pat +']')
                x_min = max(0,rect_coords[0])
                x_max = min(img.shape[1], rect_coords[1])

                y_min = max(0,rect_coords[2])
                y_max = min(img.shape[0], rect_coords[3])

                x = np.arange(x_min, x_max)
                y = np.arange(y_min, y_max)
                pixels = img[np.repeat(y,len(x)) , np.tile(x, len(y)),:]

                mean_pixel = list(np.round(np.nanmean(pixels,axis = 0)))
                pixel_per_patch[idx] = pixel_per_patch[idx] + (mean_pixel)
        
        gt_patches = all_gt_patches[batch_idx]
        for idx, pat in enumerate(gt_patches):
            if pat !=-1:
                x_min = max(0,rect_coords[0])
                x_max = min(img.shape[1], rect_coords[1])

                y_min = max(0,rect_coords[2])
                y_max = min(img.shape[0], rect_coords[3])

                x = np.arange(x_min, x_max)
                y = np.arange(y_min, y_max)
                pixels = img[np.repeat(y,len(x)) , np.tile(x, len(y)),:]

                mean_pixel = list(np.round(np.nanmean(pixels,axis = 0)))
                gt_pixel_per_patch[idx] = gt_pixel_per_patch[idx] + (mean_pixel)
        difference = np.array([])
        coeff_list = np.array([])
        p_value_list = np.array([])
        
        for gt,pred in zip(gt_pixel_per_patch, pixel_per_patch):
            result = np.absolute(np.subtract(gt, pred))
#             print("pixel difference: ", np.mean(result))
            coeff, p_value = pearsonr(gt,pred)
            coeff_list = np.append(coeff_list, coeff)
            p_value_list = np.append(p_value_list , p_value)
            difference = np.append(difference, round(np.nanmean(result),1))
        final_result[str(length)]['diff'] = difference
        final_result[str(length)]['coeff'] = coeff_list
        final_result[str(length)]['p_value'] = p_value_list
        final_result[str(length)]['gt_pixel_patches'] = gt_pixel_per_patch
        final_result[str(length)]['pred_pixel_patches'] = pixel_per_patch
    return final_result