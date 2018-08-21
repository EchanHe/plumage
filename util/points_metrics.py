import numpy as np




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
