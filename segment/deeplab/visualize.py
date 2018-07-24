import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
import os
import cv2
#General function for 





#img [height, width, 3]
#coords [x,y, x,y]
def show_images_coords(img, coords):
    cols_num_per_coord = 2
    lm_cnt = coords.shape[0]//cols_num_per_coord

    for i_col in range(lm_cnt):
        x = coords[ i_col * cols_num_per_coord]
        y = coords[ i_col * cols_num_per_coord +1] 
        if x >= 0 and ~np.isnan(x):
            plt.plot(x, y, 'x' , alpha=0.8 , mew = 4 , mec = 'cyan' )

    plt.imshow(img)



##Save imgs 

#params
#   filnames: list of file names (m,1)
#   pred_coords: matrix of predicted coord (m , n*2)
#   gt_coords: matrix of Ground Truth cood (m, n*2) x1,y1,x2,y2

def save_imgs_gt_pred(file_names, pred_coord, gt_coords ,lm_cnt, cols_num_per_coord , width,height,scale ,
 images_folder , output_folder , pck_threshold =10, draw_scale = True , dis_width=256, dis_height=170):
    markersize = 30
    fontsize=12

    # cols_num_per_coord = valid_data.cols_num_per_coord

    if len(file_names) ==1:
        batch =1
        nrows=1
        ncols=1
    else:
        batch = 10
        nrows = 2
        ncols =5


    scale_width = width//scale
    scale_height = height//scale



    crop_up = (scale_height - dis_height)//2
    crop_left = (scale_width - dis_width)//2
    # crop_down = scale_height - crop_up
    # crop_right = scale_width -crop_left
    for id_df in range(0,len(file_names),batch):
        fig  = plt.figure(figsize=(100,40))  
        file_names_batch = file_names[id_df:id_df+batch]
        gt_coords_batch = gt_coords[id_df:id_df+batch,:]
        pred_coord_batch = pred_coord[id_df:id_df+batch,:]
        
        
        for id_files, file_name in enumerate(file_names_batch):
            plt.subplot(nrows, ncols, id_files+1)
            filepath = images_folder+file_name
            img =  Image.open(filepath)
            img = img.resize((scale_width,scale_height))
            img = img.crop((crop_left, crop_up, scale_width -crop_left, scale_height - crop_up))
            plt.imshow(img)
            plt.title(file_name+"\n"+str(id_df + id_files),fontsize=40)
            
            if draw_scale ==True:

                #draw the scale on images
                plt.plot([5,5], [5,5+pck_threshold], marker = 'o',color = 'white' , linewidth = 10.0)
                plt.text(10* (1 + 0.01) , 10 * (1 + 0.01)  , str(pck_threshold)+" pixels", 
                             fontsize=fontsize * 4 , bbox=dict(facecolor='white', alpha=0.4))

            for i_col in range(lm_cnt):
                x = pred_coord_batch[id_files , i_col * cols_num_per_coord] //scale
                y = pred_coord_batch[id_files , i_col * cols_num_per_coord +1] //scale

                x_gt = gt_coords_batch[id_files , i_col * cols_num_per_coord] //scale
                y_gt = gt_coords_batch[id_files , i_col * cols_num_per_coord +1] //scale
                
                y = y-crop_up
                y_gt = y_gt - crop_up


                if x_gt != -1 and ~np.isnan(x_gt):

                    plt.plot(x, y, 'x',markersize = markersize , alpha=0.8 , mew = 4 , mec = 'cyan' )
                    plt.text(x* (1 + 0.01) , y* (1 + 0.01)  , str(i_col), 
                             fontsize=fontsize , bbox=dict(facecolor='white', alpha=0.4))

                    plt.plot(x_gt, y_gt,'x',markersize = markersize , alpha=0.8, mew = 4 , mec = 'orange')
                    plt.text(x_gt* (1 + 0.01), y_gt* (1 + 0.01) , "GT_"+str(i_col), 
                             fontsize=fontsize , bbox=dict(facecolor='white', alpha=0.4))

        
        fig.savefig(output_folder + "id_df{}.jpg".format(id_df))


#calculate different metrics between prediction and gt
# return pixel different per key point, mean of pixel different.
# return 
def Accuracy(pred_coords, gt_coords , lm_cnt , pck_threshold, scale = 1 ):
    df_size = gt_coords.shape[0]
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


def heatmap_to_coord(heatmaps , ori_width , ori_height):
    df_size = heatmaps.shape[0]
    cnt_size = heatmaps.shape[3]
    output_result = np.ones((df_size,cnt_size*2))
    for i in range(df_size):
        for j in range(cnt_size):
            heat_map = heatmaps[i,:,:,j]
            ori_heatmap = cv2.resize(heat_map, dsize=(ori_width, ori_height),interpolation = cv2.INTER_NEAREST)
            map_shape = np.unravel_index(np.argmax(ori_heatmap, axis=None), ori_heatmap.shape)
            output_result[i,j*2+0] = map_shape[1] + 1
            output_result[i,j*2+1] = map_shape[0] + 1
    return output_result

#write coords only
def write_coord(pred_coords , gt_coords , folder,file_name = "hg_valid" ):
    pred_file_name = folder + file_name+"_pred.csv"
    gt_file_name = folder +file_name + "_gt.csv"
    print("Save VALID prediction in "+pred_file_name)
    print("save GROUND TRUTH in " + gt_file_name)
    np.savetxt(pred_file_name, pred_coords, delimiter=",")
    np.savetxt(gt_file_name, gt_coords, delimiter=",")

#write prediction coordinates to csv
def write_pred_dataframe(gt_df , pred_coords , folder,file_name = "hg_valid_data"):
    file_df = gt_df.iloc[:,:2]
    gt_coords = gt_df.iloc[:,2:].as_matrix()
    pred_coords[gt_coords == -1] =-1
    result_pd = pd.DataFrame(pred_coords)
    out_data = pd.concat([file_df,result_pd],axis=1)
    pred_file_name = folder + file_name+"_pred.csv"
    out_data.columns = gt_df.columns
    out_data.to_csv(pred_file_name,index=False)