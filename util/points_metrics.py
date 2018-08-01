import numpy as np



#calculate different metrics between prediction and gt
# return pixel different per key point, mean of pixel different.
# return 
def pck_accuracy(pred_coords, gt_coords , lm_cnt , pck_threshold, scale = 1 ):
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